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
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    center: Tuple[int, int]           # (cx, cy)
    norm_offset: Tuple[float, float]  # 화면 중심 기준 정규화 오프셋 (-1.0 ~ 1.0)
    area_ratio: float                 # 화면 대비 박스 면적 비율 (0.0 ~ 1.0)
    is_target: bool
    similarity: float = 1.0           # 타겟 외형 유사도 (0.0 ~ 1.0)


@dataclass
class InferenceMetrics:
    device: str = 'cpu'
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    post_track_ms: float = 0.0
    reid_ms: float = 0.0
    total_ms: float = 0.0
    max_infer_fps: float = 0.0
    adaptive_state: str = 'IDLE'      # Adaptive 모드 상태 표시용 (IDLE / SAFE / GUARD / SEARCH)


class PersonTrackerCore:
    """
    5가지 추적 모드 및 추론 성능(Latency Breakdown) 계측을 지원하는 YOLO 인물 추적 코어 모듈
      - 모드 1 (BYTETRACK): 기본 ByteTrack (순수 위치 기반)
      - 모드 2 (HIST_REID): ByteTrack + 상/하의 HSV 색상 히스토그램 Re-ID (초경량 외형 복구)
      - 모드 3 (BOTSORT)  : BoT-SORT + 내장 Re-ID (with_reid=True, GMC 포함)
      - 모드 4 (BOT_HIST) : BoT-SORT(GMC + 단기 특징 Re-ID) + Color-Hist(장기 색상 Re-ID) 완전체 결합
      - 모드 5 (ADAPTIVE) : 평상시 고속 추적(SAFE) -> 타인 근접 시 교차 방어(GUARD) -> 실종 시 재탐색(SEARCH) 자동 스위칭
    """

    MODE_BYTETRACK = 1
    MODE_HIST_REID = 2
    MODE_BOTSORT = 3
    MODE_BOT_HIST = 4
    MODE_ADAPTIVE = 5

    MODE_NAMES = {
        1: '1:ByteTrack',
        2: '2:ColorReID',
        3: '3:BoT-SORT',
        4: '4:BoT+Hist',
        5: '5:Adaptive',
    }

    BOTSORT_MODES = {MODE_BOTSORT, MODE_BOT_HIST}
    HIST_ACTIVE_MODES = {MODE_HIST_REID, MODE_BOT_HIST}

    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        imgsz: int = 320,
        mode: int = 5,
        device: str = '',
    ):
        self.model = YOLO(resolve_model_path(model_name))
        self.imgsz = imgsz
        self.mode = mode

        if not device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.target_id: Optional[int] = None
        self.target_hist: Optional[np.ndarray] = None
        self.last_target_bbox: Optional[Tuple[int, int, int, int]] = None
        self.pending_tracker_rebind: bool = False
        self.last_detections: List[TrackedPerson] = []
        self.last_frame: Optional[np.ndarray] = None
        self.metrics = InferenceMetrics(device=self.device)

        # 색상 히스토그램 Re-ID 임계값
        self.reid_match_thresh = 0.52
        self.id_switch_thresh = 0.35

        # BoT-SORT Re-ID 활성화용 임시 설정 파일 생성
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
        fd, path = tempfile.mkstemp(prefix='botsort_reid_', suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(cfg_content)
        return path

    def set_mode(self, new_mode: int):
        """추적 모드를 변경하고 트래커 상태를 재설정"""
        if new_mode not in self.MODE_NAMES or new_mode == self.mode:
            return
        prev_tracker_is_botsort = (self.mode in self.BOTSORT_MODES)
        new_tracker_is_botsort = (new_mode in self.BOTSORT_MODES)
        self.mode = new_mode

        if prev_tracker_is_botsort != new_tracker_is_botsort:
            self.model.predictor = None
            if self.target_id is not None:
                self.pending_tracker_rebind = True

    def toggle_device(self) -> str:
        """GPU(cuda:0)와 CPU 간 연산 장치를 실시간 전환 (저사양 시뮬레이션용)"""
        if torch.cuda.is_available():
            self.device = 'cpu' if 'cuda' in self.device else 'cuda:0'
            self.model.predictor = None
            if self.target_id is not None:
                self.pending_tracker_rebind = True
        else:
            self.device = 'cpu'
        return self.device

    def _extract_body_histogram(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 16 or bh < 32:
            return None

        cx1 = max(0, int(x1 + bw * 0.2))
        cx2 = min(w, int(x2 - bw * 0.2))
        cy1 = max(0, int(y1 + bh * 0.15))
        cy2 = min(h, int(y2 - bh * 0.1))

        if cx2 - cx1 < 8 or cy2 - cy1 < 16:
            return None

        roi = frame[cy1:cy2, cx1:cx2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 25, 30]), np.array([180, 255, 255]))

        mid_y = hsv.shape[0] // 2
        upper_hsv, lower_hsv = hsv[:mid_y, :], hsv[mid_y:, :]
        upper_mask, lower_mask = mask[:mid_y, :], mask[mid_y:, :]

        hist_u = cv2.calcHist([upper_hsv], [0, 1], upper_mask, [18, 16], [0, 180, 0, 256])
        hist_l = cv2.calcHist([lower_hsv], [0, 1], lower_mask, [18, 16], [0, 180, 0, 256])

        cv2.normalize(hist_u, hist_u, alpha=1.0, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_l, hist_l, alpha=1.0, norm_type=cv2.NORM_L1)

        return np.vstack([hist_u, hist_l]).astype(np.float32)

    @staticmethod
    def _compare_hist(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
        if hist1 is None or hist2 is None:
            return 0.0
        dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return float(max(0.0, 1.0 - dist))

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
        cls, target: TrackedPerson, detections: List[TrackedPerson]
    ) -> bool:
        """타겟 박스와 다른 사람의 박스가 겹치거나 가로 거리가 가까워 교차(Occlusion) 위험이 있는지 판별"""
        tx1, ty1, tx2, ty2 = target.bbox
        tw = max(1, tx2 - tx1)
        tcx, tcy = target.center

        for other in detections:
            if other.track_id == target.track_id:
                continue
            if cls._bbox_iou(target.bbox, other.bbox) > 0.02:
                return True
            ocx, ocy = other.center
            # 타겟 가로 폭의 1.2배 이내로 다른 사람이 접근하면 GUARD 상태로 전환
            if abs(tcx - ocx) < tw * 1.2 and abs(tcy - ocy) < (ty2 - ty1) * 0.8:
                return True
        return False

    def select_target_by_point(self, x: int, y: int) -> Optional[int]:
        for person in self.last_detections:
            x1, y1, x2, y2 = person.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.target_id = person.track_id
                self.last_target_bbox = person.bbox
                self.pending_tracker_rebind = False
                if self.last_frame is not None:
                    self.target_hist = self._extract_body_histogram(self.last_frame, person.bbox)
                return self.target_id

        self.target_id = None
        self.target_hist = None
        self.last_target_bbox = None
        self.pending_tracker_rebind = False
        return None

    def process_frame(
        self, frame: np.ndarray, auto_lock_first: bool = False
    ) -> Tuple[List[TrackedPerson], Optional[TrackedPerson]]:
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
            tracker=tracker_cfg,
            device=self.device,
            verbose=False,
            conf=0.8
        )
        t_after_yolo = time.perf_counter()

        detections: List[TrackedPerson] = []
        target_person: Optional[TrackedPerson] = None
        adaptive_state = 'IDLE'

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            if auto_lock_first and self.target_id is None and len(ids) > 0:
                self.target_id = int(ids[0])
                self.last_target_bbox = tuple(int(v) for v in boxes[0])
                self.target_hist = self._extract_body_histogram(frame, self.last_target_bbox)

            person_hists: Dict[int, Optional[np.ndarray]] = {}
            compute_all_hist_now = (
                self.target_hist is not None and self.mode in self.HIST_ACTIVE_MODES
            )

            for (x1, y1, x2, y2), track_id in zip(boxes, ids):
                track_id = int(track_id)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                norm_x = (cx - (w / 2.0)) / (w / 2.0)
                norm_y = (cy - (h / 2.0)) / (h / 2.0)
                area_ratio = float((x2 - x1) * (y2 - y1)) / float(w * h)

                sim = 1.0
                if compute_all_hist_now:
                    hist = self._extract_body_histogram(frame, bbox)
                    person_hists[track_id] = hist
                    sim = self._compare_hist(self.target_hist, hist)

                person = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    center=(cx, cy),
                    norm_offset=(norm_x, norm_y),
                    area_ratio=area_ratio,
                    is_target=(track_id == self.target_id),
                    similarity=sim,
                )
                detections.append(person)

            if self.target_id is not None:
                # 1. 트래커 엔진 전환 직후 1회 타겟 승계 (IoU + 색상 결합)
                if self.pending_tracker_rebind and len(detections) > 0:
                    best_p = None
                    best_score = -1.0
                    for p in detections:
                        iou_score = (
                            self._bbox_iou(self.last_target_bbox, p.bbox)
                            if self.last_target_bbox is not None
                            else 0.0
                        )
                        hist_score = p.similarity
                        if not compute_all_hist_now and self.target_hist is not None:
                            h_tmp = self._extract_body_histogram(frame, p.bbox)
                            hist_score = self._compare_hist(self.target_hist, h_tmp)
                            p.similarity = hist_score
                        score = 0.65 * iou_score + 0.35 * hist_score
                        if score > best_score:
                            best_score = score
                            best_p = p
                    if best_p is not None and best_score >= 0.25:
                        self.target_id = best_p.track_id
                        self.pending_tracker_rebind = False

                current_target = next((p for p in detections if p.track_id == self.target_id), None)

                # 2. 모드 2 (ColorReID) 및 모드 4 (BoT+Hist): 상시 Color-Hist 방어 및 복구
                if self.mode in self.HIST_ACTIVE_MODES and self.target_hist is not None:
                    if current_target is not None and current_target.similarity < self.id_switch_thresh:
                        current_target.is_target = False
                        current_target = None

                    if current_target is None and len(detections) > 0:
                        best_candidate = max(detections, key=lambda p: p.similarity)
                        if best_candidate.similarity >= self.reid_match_thresh:
                            self.target_id = best_candidate.track_id
                            best_candidate.is_target = True
                            current_target = best_candidate

                    if current_target is not None and current_target.similarity >= 0.65:
                        cur_hist = person_hists.get(current_target.track_id)
                        if cur_hist is not None:
                            self.target_hist = 0.97 * self.target_hist + 0.03 * cur_hist

                # 3. 모드 5 (Adaptive 스위칭): 상황에 따라 SAFE / GUARD / SEARCH 자동 전환
                elif self.mode == self.MODE_ADAPTIVE and self.target_hist is not None:
                    if current_target is None:
                        # [SEARCH 상태] 타겟이 화면에서 사라짐 -> 즉시 전체 인원 Re-ID 탐색 가동
                        adaptive_state = 'SEARCH'
                        for p in detections:
                            hist = self._extract_body_histogram(frame, p.bbox)
                            p.similarity = self._compare_hist(self.target_hist, hist)
                        if len(detections) > 0:
                            best_candidate = max(detections, key=lambda p: p.similarity)
                            if best_candidate.similarity >= self.reid_match_thresh:
                                self.target_id = best_candidate.track_id
                                best_candidate.is_target = True
                                current_target = best_candidate
                                adaptive_state = 'RECOVERED'
                    elif self._is_crowded_near_target(current_target, detections):
                        # [GUARD 상태] 다른 사람이 타겟 근처로 접근/교차 -> ID 스위치 방어 가동
                        adaptive_state = 'GUARD'
                        for p in detections:
                            hist = self._extract_body_histogram(frame, p.bbox)
                            person_hists[p.track_id] = hist
                            p.similarity = self._compare_hist(self.target_hist, hist)

                        # 만약 겹침 직후 ID가 뺏겨서 유사도가 급락하고 직전 위치와도 멀어졌다면 원래 사람으로 복구
                        if current_target.similarity < self.id_switch_thresh:
                            best_candidate = max(detections, key=lambda p: p.similarity)
                            if best_candidate.similarity >= self.reid_match_thresh:
                                self.target_id = best_candidate.track_id
                                current_target = best_candidate
                    else:
                        # [SAFE 상태] 주변 방해 없음 -> 타겟 외형比較 생략하고 순수 궤적만 고속 추적
                        adaptive_state = 'SAFE'

            for p in detections:
                p.is_target = (p.track_id == self.target_id)
                if p.is_target:
                    target_person = p
                    self.last_target_bbox = p.bbox
        else:
            if self.mode == self.MODE_ADAPTIVE and self.target_id is not None:
                adaptive_state = 'SEARCH'

        t_end = time.perf_counter()

        spd = results[0].speed
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
