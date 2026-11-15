from dataclasses import dataclass
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from person_tracker.env_bootstrap import resolve_model_path
from person_tracker.feature_extractors import (
    BaseFeatureExtractor,
    BBoxFeatureExtractor,
    PartFeatureBank,
    PoseKeypointFeatureExtractor,
    SegMaskFeatureExtractor,
)


@dataclass
class UnifiedTrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    bottom_center: Tuple[int, int]
    norm_offset: Tuple[float, float]
    area_ratio: float
    is_target: bool
    similarity: float = 1.0
    mask_poly: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None  # (17, 3) [x, y, conf]
    visible_parts: Tuple[bool, bool, bool] = (True, True, True)


@dataclass
class MultiTrackerMetrics:
    model_label: str = 'yolo26n-pose.pt'
    task: str = 'pose'
    device: str = 'CPU'
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    post_track_ms: float = 0.0
    reid_ms: float = 0.0
    total_ms: float = 0.0
    max_infer_fps: float = 0.0
    adaptive_state: str = 'IDLE'


class MultiModelTrackerCore:
    """
    3종 YOLO 버전(YOLOv8n / YOLO11n / YOLO26n) x 3종 태스크(Detect / Segment / Pose) = 9개 모델과
    5가지 추적 알고리즘을 자유롭게 실시간 교차 조합할 수 있는 모듈형 코어
    """

    FAMILIES = {
        1: ('v8n', 'yolov8n'),
        2: ('11n', 'yolo11n'),
        3: ('26n', 'yolo26n'),
    }

    TASKS = {
        1: ('Det', '', 'detect'),
        2: ('Seg', '-seg', 'segment'),
        3: ('Pose', '-pose', 'pose'),
    }

    MODE_BYTETRACK = 1
    MODE_TASK_REID = 2
    MODE_BOTSORT = 3
    MODE_BOT_REID = 4
    MODE_ADAPTIVE = 5

    MODE_NAMES = {
        1: '1:ByteTrack',
        2: '2:Task-ReID',
        3: '3:BoT-SORT',
        4: '4:BoT+ReID',
        5: '5:Adaptive',
    }

    BOTSORT_MODES = {MODE_BOTSORT, MODE_BOT_REID}
    REID_ACTIVE_MODES = {MODE_TASK_REID, MODE_BOT_REID}

    def __init__(
        self,
        family_idx: int = 3,   # 기본: 3 (YOLO26n)
        task_idx: int = 3,     # 기본: 3 (Pose)
        mode: int = 5,         # 기본: 5 (Adaptive)
        imgsz: int = 320,
        device: str = '',
        conf: float = 0.15,
        iou: float = 0.55,
    ):
        self.family_idx = family_idx if family_idx in self.FAMILIES else 3
        self.task_idx = task_idx if task_idx in self.TASKS else 3
        self.mode = mode if mode in self.MODE_NAMES else 5
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        if not device:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 태스크별 특징 추출기 플러그인 매핑
        self.extractors: Dict[str, BaseFeatureExtractor] = {
            'detect': BBoxFeatureExtractor(),
            'segment': SegMaskFeatureExtractor(),
            'pose': PoseKeypointFeatureExtractor(),
        }

        # 모델 캐시 (한 번 로드한 모델은 메모리에 유지하여 즉시 전환)
        self.model_cache: Dict[str, YOLO] = {}
        self.botsort_cfg_path = self._create_botsort_reid_config()

        self.target_id: Optional[int] = None
        self.anchor_bank: Optional[PartFeatureBank] = None
        self.recent_bank: Optional[PartFeatureBank] = None
        self.last_target_bbox: Optional[Tuple[int, int, int, int]] = None
        self.pending_tracker_rebind: bool = False
        self.last_detections: List[UnifiedTrackedPerson] = []
        self.last_frame: Optional[np.ndarray] = None

        self.reid_match_thresh = 0.50
        self.id_switch_thresh = 0.34

        self.current_model_name = self._build_model_filename(self.family_idx, self.task_idx)
        self.model = self._get_or_load_model(self.current_model_name)
        self.metrics = MultiTrackerMetrics(
            model_label=self.current_model_name,
            task=self.current_task_name,
            device=self.device.upper(),
        )

    @classmethod
    def _build_model_filename(cls, family_idx: int, task_idx: int) -> str:
        _, prefix = cls.FAMILIES[family_idx]
        _, suffix, _ = cls.TASKS[task_idx]
        return f'{prefix}{suffix}.pt'

    @property
    def current_task_name(self) -> str:
        return self.TASKS[self.task_idx][2]

    @property
    def current_extractor(self) -> BaseFeatureExtractor:
        return self.extractors[self.current_task_name]

    def _get_or_load_model(self, model_filename: str) -> YOLO:
        if model_filename not in self.model_cache:
            resolved = resolve_model_path(model_filename)
            self.model_cache[model_filename] = YOLO(resolved)
        return self.model_cache[model_filename]

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
        fd, path = tempfile.mkstemp(prefix='botsort_multi_', suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(cfg_content)
        return path

    def set_model(self, family_idx: Optional[int] = None, task_idx: Optional[int] = None) -> str:
        """YOLO 버전(v8n/11n/26n) 또는 태스크(Det/Seg/Pose)를 실시간 교체하고 타겟을 승계"""
        new_f = family_idx if family_idx in self.FAMILIES else self.family_idx
        new_t = task_idx if task_idx in self.TASKS else self.task_idx
        if new_f == self.family_idx and new_t == self.task_idx:
            return self.current_model_name

        self.family_idx = new_f
        self.task_idx = new_t
        self.current_model_name = self._build_model_filename(self.family_idx, self.task_idx)
        self.model = self._get_or_load_model(self.current_model_name)
        self.model.predictor = None
        if self.target_id is not None:
            self.pending_tracker_rebind = True
        return self.current_model_name

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
        cls, target: UnifiedTrackedPerson, detections: List[UnifiedTrackedPerson]
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

    def _lock_target_bank(self, frame: np.ndarray, person: UnifiedTrackedPerson):
        bank = self.current_extractor.extract(
            frame, person.bbox, person.mask_poly, person.keypoints
        )
        if bank is not None:
            self.anchor_bank = bank.copy()
            self.recent_bank = bank.copy()
            person.visible_parts = bank.visible_parts

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
    ) -> Tuple[List[UnifiedTrackedPerson], Optional[UnifiedTrackedPerson]]:
        t_start = time.perf_counter()
        self.last_frame = frame
        h, w = frame.shape[:2]

        tracker_cfg = (
            self.botsort_cfg_path if self.mode in self.BOTSORT_MODES else 'bytetrack.yaml'
        )

        track_kwargs = dict(
            persist=True,
            classes=[0],
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            tracker=tracker_cfg,
            device=self.device,
            verbose=False,
        )
        if self.current_task_name == 'segment':
            track_kwargs['retina_masks'] = False

        results = self.model.track(frame, **track_kwargs)
        t_after_yolo = time.perf_counter()

        detections: List[UnifiedTrackedPerson] = []
        target_person: Optional[UnifiedTrackedPerson] = None
        adaptive_state = 'IDLE'
        extractor = self.current_extractor

        res = results[0]
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)
            n_det = len(ids)

            polys = (
                res.masks.xy
                if (self.current_task_name == 'segment' and res.masks is not None)
                else [None] * n_det
            )
            kpts = (
                res.keypoints.data.cpu().numpy()
                if (self.current_task_name == 'pose' and res.keypoints is not None)
                else [None] * n_det
            )

            person_banks: Dict[int, Optional[PartFeatureBank]] = {}
            compute_all_now = (
                self.anchor_bank is not None and self.mode in self.REID_ACTIVE_MODES
            )

            for (x1, y1, x2, y2), track_id, poly, kp in zip(boxes, ids, polys, kpts):
                track_id = int(track_id)
                bbox = (int(x1), int(y1), int(x2), int(y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                norm_x = (cx - (w / 2.0)) / (w / 2.0)
                norm_y = (cy - (h / 2.0)) / (h / 2.0)
                area_ratio = float((x2 - x1) * (y2 - y1)) / float(w * h)

                sim = 1.0
                vis_parts = (True, True, True)
                if compute_all_now:
                    bank = extractor.extract(frame, bbox, poly, kp)
                    person_banks[track_id] = bank
                    if bank is not None:
                        vis_parts = bank.visible_parts
                    sim = extractor.compute_dual_similarity(
                        self.anchor_bank, self.recent_bank, bank
                    )

                person = UnifiedTrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    center=(cx, cy),
                    bottom_center=(cx, int(y2)),
                    norm_offset=(norm_x, norm_y),
                    area_ratio=area_ratio,
                    is_target=(track_id == self.target_id),
                    similarity=sim,
                    mask_poly=poly,
                    keypoints=kp,
                    visible_parts=vis_parts,
                )
                detections.append(person)

            if auto_lock_first and self.target_id is None and len(detections) > 0:
                first_p = detections[0]
                self.target_id = first_p.track_id
                self.last_target_bbox = first_p.bbox
                self._lock_target_bank(frame, first_p)

            if self.target_id is not None:
                # 1. 모델/태스크/트래커 전환 직후 1회 타겟 자동 승계 (위치 IoU 65% + 특징 유사도 35%)
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
                        if not compute_all_now and self.anchor_bank is not None:
                            b_tmp = extractor.extract(frame, p.bbox, p.mask_poly, p.keypoints)
                            sim_score = extractor.compute_dual_similarity(
                                self.anchor_bank, self.recent_bank, b_tmp
                            )
                            p.similarity = sim_score
                        score = 0.65 * iou_score + 0.35 * sim_score
                        if score > best_score:
                            best_score = score
                            best_p = p
                    if best_p is not None and best_score >= 0.25:
                        self.target_id = best_p.track_id
                        self.pending_tracker_rebind = False
                        # 태스크가 바뀐 경우 새 태스크 추출기로 최근 뱅크 갱신
                        new_bank = extractor.extract(
                            frame, best_p.bbox, best_p.mask_poly, best_p.keypoints
                        )
                        if new_bank is not None:
                            self.recent_bank = new_bank.copy()

                current_target = next(
                    (p for p in detections if p.track_id == self.target_id), None
                )

                # 2. 모드 2 (Task-ReID) & 모드 4 (BoT+ReID)
                if self.mode in self.REID_ACTIVE_MODES and self.anchor_bank is not None:
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
                        extractor.update_recent_bank(
                            self.recent_bank, person_banks.get(current_target.track_id)
                        )

                # 3. 모드 5 (Adaptive 스위칭)
                elif self.mode == self.MODE_ADAPTIVE and self.anchor_bank is not None:
                    if current_target is None:
                        adaptive_state = 'SEARCH'
                        for p in detections:
                            b = extractor.extract(frame, p.bbox, p.mask_poly, p.keypoints)
                            if b is not None:
                                p.visible_parts = b.visible_parts
                            p.similarity = extractor.compute_dual_similarity(
                                self.anchor_bank, self.recent_bank, b
                            )
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
                            b = extractor.extract(frame, p.bbox, p.mask_poly, p.keypoints)
                            person_banks[p.track_id] = b
                            if b is not None:
                                p.visible_parts = b.visible_parts
                            p.similarity = extractor.compute_dual_similarity(
                                self.anchor_bank, self.recent_bank, b
                            )
                        if current_target.similarity < self.id_switch_thresh:
                            best_candidate = max(detections, key=lambda p: p.similarity)
                            if best_candidate.similarity >= self.reid_match_thresh:
                                self.target_id = best_candidate.track_id
                                current_target = best_candidate
                    else:
                        adaptive_state = 'SAFE'
                        t_bank = extractor.extract(
                            frame,
                            current_target.bbox,
                            current_target.mask_poly,
                            current_target.keypoints,
                        )
                        if t_bank is not None:
                            current_target.visible_parts = t_bank.visible_parts
                        current_target.similarity = extractor.compute_dual_similarity(
                            self.anchor_bank, self.recent_bank, t_bank
                        )
                        if current_target.similarity >= 0.62:
                            extractor.update_recent_bank(self.recent_bank, t_bank)

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
        m.model_label = self.current_model_name
        m.task = self.current_task_name.upper()
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
