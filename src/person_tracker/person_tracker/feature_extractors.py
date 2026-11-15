from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class PartFeatureBank:
    """
    모든 태스크(Detect / Segment / Pose)가 공통으로 반환하는 3부위 + 전체 특징 뱅크
      - full : 전체 가시 영역 히스토그램
      - upper: 상반신(어깨~골반 / 상의) 히스토그램
      - mid  : 중하반신(골반~무릎 / 바지) 히스토그램
      - lower: 최하단(무릎~발목 / 정강이·신발) 히스토그램
    """
    full: np.ndarray
    upper: Optional[np.ndarray]
    mid: Optional[np.ndarray]
    lower: Optional[np.ndarray]
    visible_parts: Tuple[bool, bool, bool] = (True, True, True)

    def copy(self) -> 'PartFeatureBank':
        return PartFeatureBank(
            full=self.full.copy(),
            upper=self.upper.copy() if self.upper is not None else None,
            mid=self.mid.copy() if self.mid is not None else None,
            lower=self.lower.copy() if self.lower is not None else None,
            visible_parts=self.visible_parts,
        )


class BaseFeatureExtractor:
    """태스크별 특징 추출기의 공통 베이스 클래스 (Strategy Pattern)"""

    @staticmethod
    def calc_hsv_hist(hsv_roi: np.ndarray, mask_roi: np.ndarray) -> Optional[np.ndarray]:
        if hsv_roi.size == 0 or mask_roi.size == 0:
            return None
        if cv2.countNonZero(mask_roi) < 12:
            return None
        hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.astype(np.float32)

    @staticmethod
    def hist_sim(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
        if h1 is None or h2 is None:
            return 0.0
        dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        return float(max(0.0, 1.0 - dist))

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask_poly: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[PartFeatureBank]:
        raise NotImplementedError

    def compare_banks(
        self, ref: Optional[PartFeatureBank], cur: Optional[PartFeatureBank]
    ) -> float:
        """
        현재 보이는 부위(visible_parts)끼리 1:1 매칭을 우선 수행하고,
        가까이 다가와 하반신만 보이는 경우 교차 부위(Cross-Part) 매칭까지 수행합니다.
        """
        if ref is None or cur is None:
            return 0.0

        part_scores = []
        for r_h, c_h, weight in [
            (ref.upper, cur.upper, 0.35),
            (ref.mid, cur.mid, 0.40),
            (ref.lower, cur.lower, 0.25),
        ]:
            if r_h is not None and c_h is not None:
                part_scores.append((self.hist_sim(r_h, c_h), weight))

        aligned_score = (
            sum(s * w for s, w in part_scores) / sum(w for _, w in part_scores)
            if part_scores
            else 0.0
        )
        full_score = self.hist_sim(ref.full, cur.full)

        candidates = [full_score, aligned_score]
        if ref.mid is not None:
            candidates.append(self.hist_sim(ref.mid, cur.full) * 0.95)
        if ref.lower is not None:
            candidates.append(self.hist_sim(ref.lower, cur.full) * 0.92)
        if ref.mid is not None and cur.upper is not None:
            candidates.append(self.hist_sim(ref.mid, cur.upper) * 0.95)

        return float(max(candidates))

    def compute_dual_similarity(
        self,
        anchor_bank: Optional[PartFeatureBank],
        recent_bank: Optional[PartFeatureBank],
        cur_bank: Optional[PartFeatureBank],
    ) -> float:
        if cur_bank is None:
            return 0.0
        return float(
            max(
                self.compare_banks(anchor_bank, cur_bank),
                self.compare_banks(recent_bank, cur_bank),
            )
        )

    @staticmethod
    def update_recent_bank(
        recent_bank: Optional[PartFeatureBank],
        cur_bank: Optional[PartFeatureBank],
        alpha: float = 0.04,
    ):
        if recent_bank is None or cur_bank is None:
            return
        recent_bank.full = (1.0 - alpha) * recent_bank.full + alpha * cur_bank.full
        if recent_bank.upper is not None and cur_bank.upper is not None:
            recent_bank.upper = (1.0 - alpha) * recent_bank.upper + alpha * cur_bank.upper
        if recent_bank.mid is not None and cur_bank.mid is not None:
            recent_bank.mid = (1.0 - alpha) * recent_bank.mid + alpha * cur_bank.mid
        if recent_bank.lower is not None and cur_bank.lower is not None:
            recent_bank.lower = (1.0 - alpha) * recent_bank.lower + alpha * cur_bank.lower


class BBoxFeatureExtractor(BaseFeatureExtractor):
    """[Detect 태스크용] 바운딩 박스 내부 중앙 크롭 + 3등분 특징 추출기"""

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask_poly: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[PartFeatureBank]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 16 or bh < 28:
            return None

        cx1 = max(0, int(x1 + bw * 0.20))
        cx2 = min(w, int(x2 - bw * 0.20))
        cy1 = max(0, int(y1 + bh * 0.12))
        cy2 = min(h, int(y2 - bh * 0.08))
        if cx2 - cx1 < 8 or cy2 - cy1 < 16:
            return None

        roi_hsv = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_hsv, np.array([0, 22, 28]), np.array([180, 255, 255]))

        full_hist = self.calc_hsv_hist(roi_hsv, mask)
        if full_hist is None:
            return None

        rh = roi_hsv.shape[0]
        y_u, y_m = int(rh * 0.38), int(rh * 0.72)
        upper_h = self.calc_hsv_hist(roi_hsv[:y_u, :], mask[:y_u, :])
        mid_h = self.calc_hsv_hist(roi_hsv[y_u:y_m, :], mask[y_u:y_m, :])
        lower_h = self.calc_hsv_hist(roi_hsv[y_m:, :], mask[y_m:, :])

        return PartFeatureBank(
            full=full_hist,
            upper=upper_h,
            mid=mid_h,
            lower=lower_h,
            visible_parts=(upper_h is not None, mid_h is not None, lower_h is not None),
        )


class SegMaskFeatureExtractor(BaseFeatureExtractor):
    """[Segment 태스크용] 폴리곤 실루엣 마스킹(배경 0%) + 3부위 특징 추출기"""

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask_poly: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[PartFeatureBank]:
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

        roi_hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        seg_mask = np.zeros((bh, bw), dtype=np.uint8)

        if mask_poly is not None and len(mask_poly) >= 3:
            local_poly = mask_poly.astype(np.int32).copy()
            local_poly[:, 0] = np.clip(local_poly[:, 0] - x1, 0, bw - 1)
            local_poly[:, 1] = np.clip(local_poly[:, 1] - y1, 0, bh - 1)
            cv2.fillPoly(seg_mask, [local_poly], 255)
            if bw > 24 and bh > 40:
                seg_mask = cv2.erode(seg_mask, np.ones((3, 3), np.uint8), iterations=1)
        else:
            seg_mask[int(bh * 0.1):int(bh * 0.9), int(bw * 0.15):int(bw * 0.85)] = 255

        valid_color = cv2.inRange(roi_hsv, np.array([0, 20, 25]), np.array([180, 255, 255]))
        final_mask = cv2.bitwise_and(seg_mask, valid_color)
        if cv2.countNonZero(final_mask) < 20:
            final_mask = seg_mask

        full_hist = self.calc_hsv_hist(roi_hsv, final_mask)
        if full_hist is None:
            return None

        y_u, y_m = int(bh * 0.35), int(bh * 0.70)
        upper_h = self.calc_hsv_hist(roi_hsv[:y_u, :], final_mask[:y_u, :])
        mid_h = self.calc_hsv_hist(roi_hsv[y_u:y_m, :], final_mask[y_u:y_m, :])
        lower_h = self.calc_hsv_hist(roi_hsv[y_m:, :], final_mask[y_m:, :])

        return PartFeatureBank(
            full=full_hist,
            upper=upper_h,
            mid=mid_h,
            lower=lower_h,
            visible_parts=(upper_h is not None, mid_h is not None, lower_h is not None),
        )


class PoseKeypointFeatureExtractor(BaseFeatureExtractor):
    """
    [Pose 태스크용] COCO 17개 관절 좌표 및 가시성(conf)을 기반으로
    실제 해부학적 부위(어깨~골반=상의, 골반~무릎=허벅지, 무릎~발목=정강이)만 정확히 마스킹해 추출하는 추출기
    """

    # COCO Keypoint 인덱스:
    # 5:L-Shoulder, 6:R-Shoulder, 11:L-Hip, 12:R-Hip, 13:L-Knee, 14:R-Knee, 15:L-Ankle, 16:R-Ankle
    KP_CONF_THRESH = 0.35

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask_poly: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> Optional[PartFeatureBank]:
        if keypoints is None or len(keypoints) < 17:
            return BBoxFeatureExtractor().extract(frame, bbox)

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

        roi_hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        valid_color = cv2.inRange(roi_hsv, np.array([0, 20, 25]), np.array([180, 255, 255]))

        # 로컬 ROI 좌표로 변환된 키포인트 (17, 3) -> [lx, ly, conf]
        kp = keypoints.copy()
        kp[:, 0] = np.clip(kp[:, 0] - x1, 0, bw - 1)
        kp[:, 1] = np.clip(kp[:, 1] - y1, 0, bh - 1)

        def is_vis(idx: int) -> bool:
            return float(kp[idx, 2]) >= self.KP_CONF_THRESH

        def pt(idx: int) -> Tuple[int, int]:
            return (int(kp[idx, 0]), int(kp[idx, 1]))

        limb_thickness = max(6, int(bw * 0.22))

        # 1. 상의(Upper/Torso) 마스크: 어깨(5,6)와 골반(11,12)을 잇는 몸통 다각형 또는 두꺼운 골격선
        upper_mask = np.zeros((bh, bw), dtype=np.uint8)
        torso_pts = [pt(i) for i in (5, 6, 12, 11) if is_vis(i)]
        if len(torso_pts) >= 3:
            cv2.fillPoly(upper_mask, [np.array(torso_pts, dtype=np.int32)], 255)
        elif (is_vis(5) or is_vis(6)) and (is_vis(11) or is_vis(12)):
            p_sh = pt(5) if is_vis(5) else pt(6)
            p_hip = pt(11) if is_vis(11) else pt(12)
            cv2.line(upper_mask, p_sh, p_hip, 255, thickness=limb_thickness * 2)

        # 2. 중하반신(Mid/Thighs) 마스크: 골반(11->13, 12->14)에서 무릎까지 선분 스트립
        mid_mask = np.zeros((bh, bw), dtype=np.uint8)
        for hip_i, knee_i in ((11, 13), (12, 14)):
            if is_vis(hip_i) and is_vis(knee_i):
                cv2.line(mid_mask, pt(hip_i), pt(knee_i), 255, thickness=limb_thickness)

        # 3. 최하단(Lower/Calves & Shoes) 마스크: 무릎(13->15, 14->16)에서 발목까지 선분 스트립
        lower_mask = np.zeros((bh, bw), dtype=np.uint8)
        for knee_i, ank_i in ((13, 15), (14, 16)):
            if is_vis(knee_i) and is_vis(ank_i):
                cv2.line(lower_mask, pt(knee_i), pt(ank_i), 255, thickness=limb_thickness)
            elif is_vis(ank_i):
                cv2.circle(lower_mask, pt(ank_i), limb_thickness, 255, -1)

        upper_mask = cv2.bitwise_and(upper_mask, valid_color)
        mid_mask = cv2.bitwise_and(mid_mask, valid_color)
        lower_mask = cv2.bitwise_and(lower_mask, valid_color)

        upper_h = self.calc_hsv_hist(roi_hsv, upper_mask)
        mid_h = self.calc_hsv_hist(roi_hsv, mid_mask)
        lower_h = self.calc_hsv_hist(roi_hsv, lower_mask)

        # 전체(Full) 마스크는 관절로 확인된 모든 부위의 합집합
        combined_mask = cv2.bitwise_or(upper_mask, cv2.bitwise_or(mid_mask, lower_mask))
        if cv2.countNonZero(combined_mask) < 20:
            # 관절이 너무 안 보일 때는 중앙 영역 폴백
            return BBoxFeatureExtractor().extract(frame, bbox)

        full_h = self.calc_hsv_hist(roi_hsv, combined_mask)
        if full_h is None:
            return BBoxFeatureExtractor().extract(frame, bbox)

        return PartFeatureBank(
            full=full_h,
            upper=upper_h,
            mid=mid_h,
            lower=lower_h,
            visible_parts=(upper_h is not None, mid_h is not None, lower_h is not None),
        )
