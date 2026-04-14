"""Heuristics for selecting one best target from many YOLO detections."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math


@dataclass
class DetectionCandidate:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    score: float = 0.0
    score_details: dict[str, float] = field(default_factory=dict)

    @property
    def center_xy(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(a_xyxy, b_xyxy):
    ax1, ay1, ax2, ay2 = a_xyxy
    bx1, by1, bx2, by2 = b_xyxy
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / max(1.0, union)


class TargetSelector:
    """Rank detections using confidence, image position, size, and temporal consistency."""

    def __init__(
        self,
        preferred_class_names,
        use_all_classes_as_candidates=False,
        other_class_score=0.15,
        confidence_weight=0.55,
        area_weight=0.15,
        center_weight=0.10,
        temporal_weight=0.20,
        same_class_bonus=0.10,
    ):
        self._preferred_class_names = {str(name).lower() for name in preferred_class_names}
        self._use_all_classes_as_candidates = bool(use_all_classes_as_candidates)
        self._other_class_score = float(other_class_score)
        self._confidence_weight = float(confidence_weight)
        self._area_weight = float(area_weight)
        self._center_weight = float(center_weight)
        self._temporal_weight = float(temporal_weight)
        self._same_class_bonus = float(same_class_bonus)
        self._previous = None

    def reset(self):
        self._previous = None

    def _class_prior(self, class_name):
        if class_name.lower() in self._preferred_class_names:
            return 1.0
        if self._use_all_classes_as_candidates:
            return self._other_class_score
        return None

    def _score_candidate(self, candidate, image_shape):
        img_h, img_w = image_shape
        class_prior = self._class_prior(candidate.class_name)
        if class_prior is None:
            return None

        cx, cy = candidate.center_xy
        img_cx = 0.5 * img_w
        img_cy = 0.5 * img_h
        diag = max(1.0, math.hypot(img_w, img_h))
        center_dist = math.hypot(cx - img_cx, cy - img_cy) / diag
        center_score = max(0.0, 1.0 - 2.0 * center_dist)

        area_norm = candidate.area / max(1.0, img_w * img_h)
        area_score = min(1.0, math.sqrt(max(0.0, area_norm)))

        temporal_score = 0.0
        same_class_score = 0.0
        if self._previous is not None:
            prev_cx, prev_cy = self._previous.center_xy
            move_dist = math.hypot(cx - prev_cx, cy - prev_cy) / diag
            move_score = max(0.0, 1.0 - 3.0 * move_dist)
            iou_score = _bbox_iou(candidate.bbox_xyxy, self._previous.bbox_xyxy)
            temporal_score = max(move_score, iou_score)
            if candidate.class_id == self._previous.class_id:
                same_class_score = self._same_class_bonus

        details = {
            "prior": class_prior,
            "conf": self._confidence_weight * candidate.confidence,
            "area": self._area_weight * area_score,
            "center": self._center_weight * center_score,
            "temp": self._temporal_weight * temporal_score,
            "same": same_class_score,
        }
        score = sum(details.values())
        return score, details

    def select(self, candidates, image_shape):
        best = None
        ranked = []
        for candidate in candidates:
            scored_result = self._score_candidate(candidate, image_shape=image_shape)
            if scored_result is None:
                continue
            score, details = scored_result
            scored = replace(candidate, score=score, score_details=details)
            ranked.append(scored)
            if best is None or scored.score > best.score:
                best = scored

        ranked.sort(key=lambda cand: cand.score, reverse=True)
        if best is not None:
            self._previous = best
        return best, ranked
