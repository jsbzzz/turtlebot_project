from dataclasses import dataclass
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from person_tracker.env_bootstrap import resolve_model_path


@dataclass
class SegTrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2)
    center: Tuple[int, int]                  # (cx, cy)
    bottom_center: Tuple[int, int]           # (cx, y2) - 발바닥/하단 기준점 (거리 역산용)
    norm_offset: Tuple[float, float]         # 화면 중심 기준 정규화 오프셋 (-1.0 ~ 1.0)
    area_ratio: float                        # 화면 대비 마스크/박스 면적 비율 (0.0 ~ 1.0)
    is_target: bool
    similarity: float = 1.0                  # 타겟 외형 유사도 (0.0 ~ 1.0)
    mask_poly: Optional[np.ndarray] = None   # 세그멘테이션 폴리곤 좌표 (N, 2)


@dataclass
class PartHistogramBank:
    """
    거리(Scale) 및 신체 일부 잘림(Partial-View)에 불변하도록
    세그멘테이션 마스크 내부 픽셀을 3개 부위(상부/중부/하부) 및 전체로 나누어 저장하는 특징 뱅크
    """
    full: np.ndarray
    upper: Optional[np.ndarray]
    mid: Optional[np.ndarray]
    lower: Optional[np.ndarray]


@dataclass
class SegInferenceMetrics:
    device: str = 'cpu'
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    post_track_ms: float = 0.0
    reid_ms: float = 0.0
    total_ms: float = 0.0
    max_infer_fps: float = 0.0
    adaptive_state: str = 'IDLE'


class SegPersonTrackerCore:
    """
    YOLOv8n-seg (인스턴스 세그멘테이션 마스킹) + 부위별 2중 기억(Anchor + Recent Part Bank) 기반 추적 코어
      - 모드 1 (BYTETRACK): Seg 검출 + 기본 ByteTrack
      - 모드 2 (SEG_REID) : Seg 마스킹 + 부위별 2중 기억(전신<->하반신 불변) Re-ID
      - 모드 3 (BOTSORT)  : Seg 검출 + BoT-SORT (GMC + YOLO 특징 Re-ID)
      - 모드 4 (BOT_SEG)  : BoT-SORT + Seg 부위별 2중 기억 완전체 결합
      - 모드 5 (ADAPTIVE) : 평상시 SAFE -> 근접 시 GUARD(Seg 마스킹 방어) -> 실종 시 SEARCH 자동 스위칭
    """

    MODE_BYTETRACK = 1
    MODE_SEG_REID = 2
    MODE_BOTSORT = 3
    MODE_BOT_SEG = 4
    MODE_ADAPTIVE = 5

    MODE_NAMES = {
        1: '1:ByteTrack',
        2: '2:Seg-PartReID',
        3: '3:BoT-SORT',
        4: '4:BoT+SegReID',
        5: '5:Adaptive(Seg)',
    }

    BOTSORT_MODES = {MODE_BOTSORT, MODE_BOT_SEG}
    SEG_REID_MODES = {MODE_SEG_REID, MODE_BOT_SEG}

    def __init__(
        self,
        model_name: str = 'yolov8n-seg.pt',
        imgsz: int = 320,
        mode: int = 5,
        device: str = '',
        conf: float = 0.15,  # 하반신/발목만 보일 때도 검출을 유지하기 위해 기본 0.15 설정
        iou: float = 0.55,
    ):
        self.model = YOLO(resolve_model_path(model_name))
        self.imgsz = imgsz
        self.mode = mode
        self.conf = conf
        self.iou = iou

        if not device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.target_id: Optional[int] = None
        # 2중 기억 구조: 최초 선택 시의 원본 특징(anchor_bank) + 최근 변화 적응 특징(recent_bank)
        self.anchor_bank: Optional[PartHistogramBank] = None
        self.recent_bank: Optional[PartHistogramBank] = None
        self.last_target_bbox: Optional[Tuple[int, int, int, int]] = None
        self.pending_tracker_rebind: bool = False
        self.last_detections: List[SegTrackedPerson] = []
        self.last_frame: Optional[np.ndarray] = None
        self.metrics = SegInferenceMetrics(device=self.device)

        self.reid_match_thresh = 0.50
        self.id_switch_thresh = 0.34

        self.botsort_cfg_path = self._create_botsort_reid_config()

    def _create_botsort_reid_config(self) -> str:
        cfg_content = (
            "tracker_type: botsort\n"
            "track_high_thresh: 0.25\n"
            "track_low_thresh: 0.1\n"
            "new_track_thresh: 0.25\n"
            "track_buffer: 90\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n"
            "gmc_method: sparseOptFlow\n"
            "proximity_thresh: 0.5\n"
            "appearance_thresh: 0.65\n"
            "with_reid: True\n"
            "model: auto\n"
        )
        fd, path = tempfile.mkstemp(prefix='botsort_seg_reid_', suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(cfg_content)
        return path

    def set_mode(self, new_mode: int):
        if new_mode not in self.MODE_NAMES or new_mode == self.mode:
            return
        prev_is_botsort = (self.mode in self.BOTSORT_MODES)
        new_is_botsort = (new_mode in self.BOTSORT_MODES)
        self.mode = new_mode

        if prev_is_botsort != new_is_botsort:
            self.model.predictor = None
            if self.target_id is not None:
                self.pending_tracker_rebind = True

    def toggle_device(self) -> str:
        if torch.cuda.is_available():
            self.device = 'cpu' if 'cuda' in self.device else 'cuda:0'
            self.model.predictor = None
            if self.target_id is not None:
                self.pending_tracker_rebind = True
        else:
            self.device = 'cpu'
        return self.device

    @staticmethod
    def _calc_hsv_hist(hsv_roi: np.ndarray, mask_roi: np.ndarray) -> Optional[np.ndarray]:
        if hsv_roi.size == 0 or mask_roi.size == 0:
            return None
        if cv2.countNonZero(mask_roi) < 15:
            return None
        hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.astype(np.float32)

    def _extract_masked_part_bank(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask_poly: Optional[np.ndarray],
    ) -> Optional[PartHistogramBank]:
        """
        세그멘테이션 폴리곤으로 배경을 100% 제거한 사람 마스크를 만든 뒤,
        상부(Upper) / 중부(Mid: 골반·허벅지) / 하부(Lower: 정강이·신발) 3개 슬롯과 전체(Full)의
        거리 불변(L1 정규화) HSV 히스토그램 뱅크를 추출합니다.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(w, bbox[2]),
            min(h, bbox[3]),
        )
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 20:
            return None

        roi_bgr = frame[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 1. 세그멘테이션 폴리곤을 ROI 로컬 좌표계 이진 마스크로 렌더링
        seg_mask = np.zeros((bh, bw), dtype=np.uint8)
        if mask_poly is not None and len(mask_poly) >= 3:
            local_poly = mask_poly.astype(np.int32).copy()
            local_poly[:, 0] = np.clip(local_poly[:, 0] - x1, 0, bw - 1)
            local_poly[:, 1] = np.clip(local_poly[:, 1] - y1, 0, bh - 1)
            cv2.fillPoly(seg_mask, [local_poly], 255)
            # 경계선 1~2픽셀의 배경 번짐(Blur)까지 제거하기 위해 살짝 침식(Erode)
            if bw > 24 and bh > 40:
                kernel = np.ones((3, 3), np.uint8)
                seg_mask = cv2.erode(seg_mask, kernel, iterations=1)
        else:
            # 폴리곤이 없을 경우 중앙 영역 폴백
            seg_mask[int(bh * 0.1):int(bh * 0.9), int(bw * 0.15):int(bw * 0.85)] = 255

        # 너무 어두운 그림자(V<25) 픽셀 제외 후 세그멘테이션 마스크와 결합
        valid_color = cv2.inRange(roi_hsv, np.array([0, 20, 25]), np.array([180, 255, 255]))
        final_mask = cv2.bitwise_and(seg_mask, valid_color)
        if cv2.countNonZero(final_mask) < 20:
            final_mask = seg_mask

        full_hist = self._calc_hsv_hist(roi_hsv, final_mask)
        if full_hist is None:
            return None

        # 3등분 부위별 슬롯 추출 (0~35%: 상부, 35~70%: 중부/바지, 70~100%: 하부/정강이·신발)
        y_u = int(bh * 0.35)
        y_m = int(bh * 0.70)

        upper_hist = self._calc_hsv_hist(roi_hsv[:y_u, :], final_mask[:y_u, :])
        mid_hist = self._calc_hsv_hist(roi_hsv[y_u:y_m, :], final_mask[y_u:y_m, :])
        lower_hist = self._calc_hsv_hist(roi_hsv[y_m:, :], final_mask[y_m:, :])

        return PartHistogramBank(
            full=full_hist,
            upper=upper_hist,
            mid=mid_hist,
            lower=lower_hist,
        )

    @staticmethod
    def _hist_sim(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
        if h1 is None or h2 is None:
            return 0.0
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        return float(max(0.0, 1.0 - dist))

    def _compare_part_banks(
        self, ref: Optional[PartHistogramBank], cur: Optional[PartHistogramBank]
    ) -> float:
        """
        전신(Full) 비교뿐만 아니라, 거리가 가까워져 하반신(바지/신발)만 보이는 상황까지
        대응하기 위해 부위별 일치도와 교차 부위(Cross-Part) 유사도 중 최댓값을 계산합니다.
        """
        if ref is None or cur is None:
            return 0.0

        # 1. 정방향 부위별 매칭 (전신 대 전신, 상-상, 중-중, 하-하)
        part_scores = []
        for r_h, c_h, weight in [
            (ref.upper, cur.upper, 0.35),
            (ref.mid, cur.mid, 0.40),
            (ref.lower, cur.lower, 0.25),
        ]:
            if r_h is not None and c_h is not None:
                part_scores.append((self._hist_sim(r_h, c_h), weight))

        aligned_score = (
            sum(s * w for s, w in part_scores) / sum(w for _, w in part_scores)
            if part_scores
            else 0.0
        )
        full_score = self._hist_sim(ref.full, cur.full)

        # 2. 부분 뷰(Partial-View) 매칭:
        # 전신을 기억한 상태에서 가까이 다가와 하반신(ref.mid 또는 ref.lower)이 현재 화면 전체(cur.full)를 차지한 경우
        partial_candidates = [full_score, aligned_score]
        if ref.mid is not None:
            partial_candidates.append(self._hist_sim(ref.mid, cur.full) * 0.95)
        if ref.lower is not None:
            partial_candidates.append(self._hist_sim(ref.lower, cur.full) * 0.92)
        if ref.mid is not None and cur.upper is not None:
            # 가까이 다가와 원래 바지(ref.mid)가 현재 박스의 상단(cur.upper)에 위치하게 된 경우
            partial_candidates.append(self._hist_sim(ref.mid, cur.upper) * 0.95)

        return float(max(partial_candidates))

    def _compute_dual_memory_similarity(
        self, cur_bank: Optional[PartHistogramBank]
    ) -> float:
        """
        원본 기억(anchor_bank)과 최근 적응 기억(recent_bank) 두 곳 모두와 비교하여
        전신 -> 하반신 -> 다시 전신으로 바뀌어도 높은 유사도를 유지합니다.
        """
        if cur_bank is None:
            return 0.0
        sim_anchor = self._compare_part_banks(self.anchor_bank, cur_bank)
        sim_recent = self._compare_part_banks(self.recent_bank, cur_bank)
        return float(max(sim_anchor, sim_recent))

    def _update_recent_bank(self, cur_bank: Optional[PartHistogramBank], alpha: float = 0.04):
        """원본(anchor_bank)은 보존하고, 최근 기억(recent_bank)만 서서히 현재 보이는 모습에 적응"""
        if cur_bank is None or self.recent_bank is None:
            return
        self.recent_bank.full = (1.0 - alpha) * self.recent_bank.full + alpha * cur_bank.full
        if self.recent_bank.upper is not None and cur_bank.upper is not None:
            self.recent_bank.upper = (1.0 - alpha) * self.recent_bank.upper + alpha * cur_bank.upper
        if self.recent_bank.mid is not None and cur_bank.mid is not None:
            self.recent_bank.mid = (1.0 - alpha) * self.recent_bank.mid + alpha * cur_bank.mid
        if self.recent_bank.lower is not None and cur_bank.lower is not None:
            self.recent_bank.lower = (1.0 - alpha) * self.recent_bank.lower + alpha * cur_bank.lower

    @staticmethod
    def _bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0:
            return 0.0
        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return float(inter) / float(areaA + areaB - inter)

    @classmethod
    def _is_crowded_near_target(
        cls, target: SegTrackedPerson, detections: List[SegTrackedPerson]
    ) -> bool:
        tx1, ty1, tx2, ty2 = target.bbox
        tw = max(1, tx2 - tx1)
        tcx, tcy = target.center
        for other in detections:
            if other.track_id == target.track_id:
                continue
            if cls._bbox_iou(target.bbox, other.bbox) > 0.02:
                return True
            ocx, ocy = other.center
            if abs(tcx - ocx) < tw * 1.2 and abs(tcy - ocy) < (ty2 - ty1) * 0.8:
                return True
        return False

    def _lock_target_bank(self, frame: np.ndarray, person: SegTrackedPerson):
        bank = self._extract_masked_part_bank(frame, person.bbox, person.mask_poly)
        if bank is not None:
            self.anchor_bank = PartHistogramBank(
                full=bank.full.copy(),
                upper=bank.upper.copy() if bank.upper is not None else None,
                mid=bank.mid.copy() if bank.mid is not None else None,
                lower=bank.lower.copy() if bank.lower is not None else None,
            )
            self.recent_bank = bank

    def select_target_by_point(self, x: int, y: int) -> Optional[int]:
        for person in self.last_detections:
            x1, y1, x2, y2 = person.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.target_id = person.track_id
                self.last_target_bbox = person.bbox
                self.pending_tracker_rebind = False
                if self.last_frame is not None:
                    self._lock_target_bank(self.last_frame, person)
                return self.target_id

        self.target_id = None
        self.anchor_bank = None
        self.recent_bank = None
        self.last_target_bbox = None
        self.pending_tracker_rebind = False
        return None

    def process_frame(
        self, frame: np.ndarray, auto_lock_first: bool = False
    ) -> Tuple[List[SegTrackedPerson], Optional[SegTrackedPerson]]:
        t_start = time.perf_counter()
        self.last_frame = frame
        h, w = frame.shape[:2]

        tracker_cfg = (
            self.botsort_cfg_path if self.mode in self.BOTSORT_MODES else 'bytetrack.yaml'
        )

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            tracker=tracker_cfg,
            device=self.device,
            retina_masks=False,  # 저사양/고속 처리를 위해 기본 마스크 해상도 사용
            verbose=False,
        )
        t_after_yolo = time.perf_counter()

        detections: List[SegTrackedPerson] = []
        target_person: Optional[SegTrackedPerson] = None
        adaptive_state = 'IDLE'

        res = results[0]
        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            polys = res.masks.xy if res.masks is not None else [None] * len(ids)

            person_banks: Dict[int, Optional[PartHistogramBank]] = {}
            compute_all_banks_now = (
                self.anchor_bank is not None and self.mode in self.SEG_REID_MODES
            )

            for (x1, y1, x2, y2), track_id, poly in zip(boxes, ids, polys):
                track_id = int(track_id)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                norm_x = (cx - (w / 2.0)) / (w / 2.0)
                norm_y = (cy - (h / 2.0)) / (h / 2.0)
                area_ratio = float((x2 - x1) * (y2 - y1)) / float(w * h)

                sim = 1.0
                if compute_all_banks_now:
                    bank = self._extract_masked_part_bank(frame, bbox, poly)
                    person_banks[track_id] = bank
                    sim = self._compute_dual_memory_similarity(bank)

                person = SegTrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    center=(cx, cy),
                    bottom_center=(cx, int(y2)),
                    norm_offset=(norm_x, norm_y),
                    area_ratio=area_ratio,
                    is_target=(track_id == self.target_id),
                    similarity=sim,
                    mask_poly=poly,
                )
                detections.append(person)

            if auto_lock_first and self.target_id is None and len(detections) > 0:
                first_p = detections[0]
                self.target_id = first_p.track_id
                self.last_target_bbox = first_p.bbox
                self._lock_target_bank(frame, first_p)

            if self.target_id is not None:
                # 1. 트래커 전환 직후 1회 타겟 위치+세그멘테이션 특징 승계
                if self.pending_tracker_rebind and len(detections) > 0:
                    best_p = None
                    best_score = -1.0
                    for p in detections:
                        iou_score = (
                            self._bbox_iou(self.last_target_bbox, p.bbox)
                            if self.last_target_bbox is not None
                            else 0.0
                        )
                        sim_score = p.similarity
                        if not compute_all_banks_now and self.anchor_bank is not None:
                            b_tmp = self._extract_masked_part_bank(frame, p.bbox, p.mask_poly)
                            sim_score = self._compute_dual_memory_similarity(b_tmp)
                            p.similarity = sim_score
                        score = 0.65 * iou_score + 0.35 * sim_score
                        if score > best_score:
                            best_score = score
                            best_p = p
                    if best_p is not None and best_score >= 0.25:
                        self.target_id = best_p.track_id
                        self.pending_tracker_rebind = False

                current_target = next((p for p in detections if p.track_id == self.target_id), None)

                # 2. 모드 2 (Seg-PartReID) 및 모드 4 (BoT+SegReID): 상시 세그멘테이션 부위별 Re-ID
                if self.mode in self.SEG_REID_MODES and self.anchor_bank is not None:
                    if current_target is not None and current_target.similarity < self.id_switch_thresh:
                        current_target.is_target = False
                        current_target = None

                    if current_target is None and len(detections) > 0:
                        best_candidate = max(detections, key=lambda p: p.similarity)
                        if best_candidate.similarity >= self.reid_match_thresh:
                            self.target_id = best_candidate.track_id
                            best_candidate.is_target = True
                            current_target = best_candidate

                    if current_target is not None and current_target.similarity >= 0.62:
                        self._update_recent_bank(person_banks.get(current_target.track_id))

                # 3. 모드 5 (Adaptive Seg 스위칭): SAFE / GUARD / SEARCH
                elif self.mode == self.MODE_ADAPTIVE and self.anchor_bank is not None:
                    if current_target is None:
                        adaptive_state = 'SEARCH'
                        for p in detections:
                            bank = self._extract_masked_part_bank(frame, p.bbox, p.mask_poly)
                            p.similarity = self._compute_dual_memory_similarity(bank)
                        if len(detections) > 0:
                            best_candidate = max(detections, key=lambda p: p.similarity)
                            if best_candidate.similarity >= self.reid_match_thresh:
                                self.target_id = best_candidate.track_id
                                best_candidate.is_target = True
                                current_target = best_candidate
                                adaptive_state = 'RECOVERED'
                    elif self._is_crowded_near_target(current_target, detections):
                        adaptive_state = 'GUARD'
                        for p in detections:
                            bank = self._extract_masked_part_bank(frame, p.bbox, p.mask_poly)
                            person_banks[p.track_id] = bank
                            p.similarity = self._compute_dual_memory_similarity(bank)
                        if current_target.similarity < self.id_switch_thresh:
                            best_candidate = max(detections, key=lambda p: p.similarity)
                            if best_candidate.similarity >= self.reid_match_thresh:
                                self.target_id = best_candidate.track_id
                                current_target = best_candidate
                    else:
                        adaptive_state = 'SAFE'
                        # SAFE 상태에서도 타겟 1명에 대해서만 가볍게 최근 부위 기억(하반신 변화 등)을 업데이트
                        t_bank = self._extract_masked_part_bank(
                            frame, current_target.bbox, current_target.mask_poly
                        )
                        current_target.similarity = self._compute_dual_memory_similarity(t_bank)
                        if current_target.similarity >= 0.62:
                            self._update_recent_bank(t_bank)

            for p in detections:
                p.is_target = (p.track_id == self.target_id)
                if p.is_target:
                    target_person = p
                    self.last_target_bbox = p.bbox
        else:
            if self.mode == self.MODE_ADAPTIVE and self.target_id is not None:
                adaptive_state = 'SEARCH'

        t_end = time.perf_counter()

        spd = res.speed
        pre_ms = float(spd.get('preprocess', 0.0))
        inf_ms = float(spd.get('inference', 0.0))
        post_ms = float(spd.get('postprocess', 0.0))
        reid_ms = (t_end - t_after_yolo) * 1000.0
        total_ms = (t_end - t_start) * 1000.0

        alpha = 0.15
        m = self.metrics
        m.device = self.device.upper()
        m.preprocess_ms = (1 - alpha) * m.preprocess_ms + alpha * pre_ms if m.total_ms > 0 else pre_ms
        m.inference_ms = (1 - alpha) * m.inference_ms + alpha * inf_ms if m.total_ms > 0 else inf_ms
        m.post_track_ms = (1 - alpha) * m.post_track_ms + alpha * post_ms if m.total_ms > 0 else post_ms
        m.reid_ms = (1 - alpha) * m.reid_ms + alpha * reid_ms if m.total_ms > 0 else reid_ms
        m.total_ms = (1 - alpha) * m.total_ms + alpha * total_ms if m.total_ms > 0 else total_ms
        m.max_infer_fps = 1000.0 / max(0.1, m.total_ms)
        m.adaptive_state = adaptive_state

        self.last_detections = detections
        return detections, target_person
