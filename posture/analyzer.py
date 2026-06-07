from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .config import MonitorConfig
from .models import DepthMeasurement, FeatureFrame, PostureResult
from .pose import LEFT_EAR, LEFT_SHOULDER, NOSE, RIGHT_EAR, RIGHT_SHOULDER
from .turtle_neck_filter import TurtleNeckFilterResult, compare_to_baseline


class PostureAnalyzer:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.baseline: Optional[FeatureFrame] = None
        self.baseline_frames: List[FeatureFrame] = []
        self.depth_baseline: Optional[Dict[str, float]] = None
        self.depth_baseline_frames: Dict[str, List[float]] = {"head": [], "shoulder": [], "chest": []}
        self.neck_baseline_vectors: List[np.ndarray] = []
        self.neck_baseline_vector: Optional[np.ndarray] = None
        self.nose_depth_baseline_frames: List[float] = []
        self.nose_depth_baseline: Optional[float] = None
        self.depth_baseline_shoulder_y: List[float] = []
        self.depth_baseline_started_at: Optional[float] = None
        self.depth_baseline_latest_at: Optional[float] = None
        self.baseline_shoulder_y: Optional[float] = None
        self.bad_since: Optional[float] = None
        self.last_bad_at: Optional[float] = None
        self.last_bad_reason: Optional[str] = None
        self.out_of_range_since: Optional[float] = None
        self.missing_frames = 0
        self.lock = threading.Lock()

    def add_baseline_frame(self, keypoints: np.ndarray) -> bool:
        features = self.extract_features(keypoints)
        if features is None:
            return False
        self.baseline_frames.append(features)
        return True

    def finalize_baseline(self) -> bool:
        if not self.baseline_frames:
            return False
        self.baseline = FeatureFrame(
            timestamp=time.time(),
            head_x=float(np.mean([f.head_x for f in self.baseline_frames])),
            shoulder_x=float(np.mean([f.shoulder_x for f in self.baseline_frames])),
            shoulder_tilt=float(np.mean([f.shoulder_tilt for f in self.baseline_frames])),
            shoulder_angle_deg=self._mean_angle_deg([f.shoulder_angle_deg for f in self.baseline_frames]),
            min_confidence=float(np.mean([f.min_confidence for f in self.baseline_frames])),
        )
        return True

    def add_depth_baseline_frame(
        self,
        measurements: List[DepthMeasurement],
        neck_result: Optional[TurtleNeckFilterResult] = None,
    ) -> bool:
        by_roi = {measurement.roi: measurement for measurement in measurements}
        required = ("head", "shoulder", "chest")
        if any(name not in by_roi or by_roi[name].median_m is None for name in required):
            return False

        now = time.time()
        if self.depth_baseline_started_at is None:
            self.depth_baseline_started_at = now
        self.depth_baseline_latest_at = now
        for name in required:
            self.depth_baseline_frames[name].append(float(by_roi[name].median_m))
        self.depth_baseline_shoulder_y.append(float(by_roi["shoulder"].cy))
        if neck_result is not None and neck_result.valid:
            self.neck_baseline_vectors.append(
                np.array(
                    [
                        neck_result.vector_x_m,
                        neck_result.vector_y_m,
                        neck_result.vector_z_m,
                    ],
                    dtype=float,
                )
            )
            if neck_result.nose_depth_m is not None:
                self.nose_depth_baseline_frames.append(float(neck_result.nose_depth_m))
        return True

    def depth_baseline_valid_seconds(self) -> float:
        if self.depth_baseline_started_at is None or self.depth_baseline_latest_at is None:
            return 0.0
        return max(0.0, self.depth_baseline_latest_at - self.depth_baseline_started_at)

    def finalize_depth_baseline(self) -> bool:
        required = ("head", "shoulder", "chest")
        if any(not self.depth_baseline_frames[name] for name in required):
            return False
        self.depth_baseline = {
            name: float(np.median(self.depth_baseline_frames[name]))
            for name in required
        }
        self.baseline_shoulder_y = float(np.median(self.depth_baseline_shoulder_y))
        if self.neck_baseline_vectors:
            self.neck_baseline_vector = np.median(np.stack(self.neck_baseline_vectors, axis=0), axis=0)
        if self.nose_depth_baseline_frames:
            self.nose_depth_baseline = float(np.median(self.nose_depth_baseline_frames))
        return True

    def analyze(self, keypoints: Optional[np.ndarray]) -> PostureResult:
        now = time.time()
        if self.baseline is None:
            return PostureResult(now, "unknown", "baseline_not_ready", False, 0.0, None)
        if keypoints is None:
            return self.handle_out_of_range(now, "person_missing_or_range_left")

        features = self.extract_features(keypoints)
        if features is None:
            return self.handle_out_of_range(now, "keypoint_confidence_low_or_range_left")

        self.missing_frames = 0
        self.out_of_range_since = None
        reason = self._classify_pose(features)
        if reason == "normal":
            self._clear_bad()
            return PostureResult(now, "normal", "normal", False, 0.0, features)

        self._mark_bad(now, reason)
        elapsed = self._bad_elapsed(now)
        return PostureResult(now, "bad", reason, elapsed >= self.config.bad_duration_seconds, elapsed, features)

    def analyze_depth(
        self,
        measurements: List[DepthMeasurement],
        features: Optional[FeatureFrame],
        neck_result: Optional[TurtleNeckFilterResult] = None,
    ) -> PostureResult:
        now = time.time()
        if self.depth_baseline is None:
            return PostureResult(now, "unknown", "depth_baseline_not_ready", False, 0.0, features, measurements, None)

        by_roi = {measurement.roi: measurement for measurement in measurements}
        required = ("head", "shoulder", "chest")
        if any(name not in by_roi or by_roi[name].median_m is None for name in required):
            return self.handle_out_of_range(now, "missing_depth_measurement")

        self.missing_frames = 0
        self.out_of_range_since = None
        deltas = {
            name: self.depth_baseline[name] - float(by_roi[name].median_m)
            for name in required
        }
        shoulder_y_base = self.baseline_shoulder_y if self.baseline_shoulder_y is not None else by_roi["shoulder"].cy
        shoulder_y_delta = abs(float(by_roi["shoulder"].cy) - float(shoulder_y_base))
        pose_shoulder_tilt_delta = self._pose_shoulder_tilt_delta(features)
        pose_shoulder_angle_delta = self._pose_shoulder_angle_delta(features)
        if neck_result is not None and self.neck_baseline_vector is not None:
            neck_result = compare_to_baseline(neck_result, self.neck_baseline_vector)
        nose_depth_delta = self._nose_depth_delta(neck_result)
        reason = self._classify_depth(
            deltas,
            shoulder_y_delta,
            pose_shoulder_tilt_delta,
            pose_shoulder_angle_delta,
            neck_result,
            nose_depth_delta,
        )

        if reason == "normal":
            held = self._hold_recent_bad(now, features, measurements, deltas, neck_result, nose_depth_delta)
            if held is not None:
                return held
            self._clear_bad()
            return self._make_result(now, "normal", "normal", False, 0.0, features, measurements, deltas, neck_result, nose_depth_delta)

        self._mark_bad(now, reason)
        elapsed = self._bad_elapsed(now)
        return self._make_result(
            now,
            "bad",
            reason,
            elapsed >= self.config.bad_duration_seconds,
            elapsed,
            features,
            measurements,
            deltas,
            neck_result,
            nose_depth_delta,
        )

    def extract_features(self, keypoints: np.ndarray) -> Optional[FeatureFrame]:
        required = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]
        if any(keypoints[i][2] < self.config.min_keypoint_confidence for i in required):
            return None

        head_points = [keypoints[NOSE]]
        for index in (LEFT_EAR, RIGHT_EAR):
            if keypoints[index][2] >= self.config.min_keypoint_confidence:
                head_points.append(keypoints[index])

        shoulder_l = keypoints[LEFT_SHOULDER]
        shoulder_r = keypoints[RIGHT_SHOULDER]
        min_conf = min(float(keypoints[i][2]) for i in required)

        return FeatureFrame(
            timestamp=time.time(),
            head_x=float(np.mean([point[0] for point in head_points])),
            shoulder_x=float((shoulder_l[0] + shoulder_r[0]) / 2.0),
            shoulder_tilt=float(shoulder_l[1] - shoulder_r[1]),
            shoulder_angle_deg=self._shoulder_angle_deg(shoulder_l, shoulder_r),
            min_confidence=min_conf,
        )

    def handle_out_of_range(self, now: float, reason: str) -> PostureResult:
        self.missing_frames += 1
        if self.out_of_range_since is None:
            self.out_of_range_since = now

        elapsed = max(0.0, now - self.out_of_range_since)
        if elapsed >= self.config.out_of_range_tolerance_seconds:
            self._clear_bad()
            return PostureResult(now, "out_of_range", reason, False, 0.0, None)
        return PostureResult(now, "unknown", "waiting_for_valid_pose", False, self._bad_elapsed(now), None)

    def _classify_pose(self, features: FeatureFrame) -> str:
        assert self.baseline is not None
        head_shift = abs(features.head_x - self.baseline.head_x)
        torso_shift = abs(features.shoulder_x - self.baseline.shoulder_x)
        shoulder_tilt_delta = abs(features.shoulder_tilt - self.baseline.shoulder_tilt)
        shoulder_angle_delta = self._pose_shoulder_angle_delta(features)

        if self._is_shoulder_tilt(shoulder_tilt_delta, shoulder_angle_delta):
            return "shoulder_tilt"
        if head_shift >= self.config.head_shift_px and torso_shift < self.config.torso_shift_px:
            return "turtle_neck"
        return "normal"

    def _classify_depth(
        self,
        deltas: Dict[str, float],
        shoulder_y_delta: float,
        pose_shoulder_tilt_delta: Optional[float] = None,
        pose_shoulder_angle_delta: Optional[float] = None,
        neck_result: Optional[TurtleNeckFilterResult] = None,
        nose_depth_delta_m: Optional[float] = None,
    ) -> str:
        head = deltas["head"]
        shoulder = deltas["shoulder"]

        shoulder_angle_is_stable = (
            pose_shoulder_angle_delta is None
            or pose_shoulder_angle_delta < self.config.shoulder_tilt_angle_deg
        )

        if (
            pose_shoulder_tilt_delta is not None
            and self._is_shoulder_tilt(pose_shoulder_tilt_delta, pose_shoulder_angle_delta)
        ):
            return "shoulder_tilt"

        # Weighted turtle-neck score keeps the stable pre-coordinate-logger signal
        # while absorbing small ROI/keypoint jitter in individual measurements.
        turtle_neck_score = self._turtle_neck_score(head, shoulder, pose_shoulder_angle_delta)
        if (
            shoulder_angle_is_stable
            and turtle_neck_score >= self.config.turtle_neck_score_threshold
        ):
            return "turtle_neck"
        if (
            shoulder_angle_is_stable
            and nose_depth_delta_m is not None
            and nose_depth_delta_m <= -self.config.turtle_neck_nose_depth_change_m
            and abs(shoulder) <= self.config.torso_stable_threshold_m
        ):
            return "turtle_neck"
        return "normal"

    def _is_shoulder_tilt(
        self,
        shoulder_tilt_delta_px: float,
        shoulder_angle_delta_deg: Optional[float],
    ) -> bool:
        if shoulder_angle_delta_deg is None:
            return False
        if shoulder_angle_delta_deg >= self.config.shoulder_tilt_angle_deg:
            return True
        return (
            shoulder_tilt_delta_px >= self.config.shoulder_tilt_px
            and shoulder_angle_delta_deg >= self.config.shoulder_tilt_combo_angle_deg
        )

    def _turtle_neck_score(
        self,
        head_delta_m: float,
        shoulder_delta_m: float,
        pose_shoulder_angle_delta: Optional[float],
    ) -> float:
        head_score = self._clamp01(head_delta_m / self.config.head_depth_threshold_m)
        shoulder_score = self._clamp01(
            1.0 - abs(shoulder_delta_m) / self.config.torso_depth_threshold_m
        )
        shoulder_angle = 0.0 if pose_shoulder_angle_delta is None else pose_shoulder_angle_delta
        shoulder_angle_score = self._clamp01(
            1.0 - shoulder_angle / self.config.shoulder_tilt_angle_deg
        )
        return (
            self.config.turtle_neck_head_weight * head_score
            + self.config.turtle_neck_shoulder_weight * shoulder_score
            + self.config.turtle_neck_shoulder_angle_weight * shoulder_angle_score
        )

    def _make_result(
        self,
        timestamp: float,
        state: str,
        reason: str,
        alarm: bool,
        elapsed_bad_seconds: float,
        features: Optional[FeatureFrame],
        measurements: List[DepthMeasurement],
        deltas: Dict[str, float],
        neck_result: Optional[TurtleNeckFilterResult],
        nose_depth_delta_m: Optional[float] = None,
    ) -> PostureResult:
        return PostureResult(
            timestamp,
            state,
            reason,
            alarm,
            elapsed_bad_seconds,
            features,
            measurements,
            deltas,
            None if neck_result is None else neck_result.angle_delta_deg,
            None if neck_result is None else neck_result.pitch_deg,
            None if neck_result is None else neck_result.vector_x_m,
            None if neck_result is None else neck_result.vector_y_m,
            None if neck_result is None else neck_result.vector_z_m,
            None if neck_result is None else neck_result.nose_depth_m,
            None if neck_result is None else neck_result.neck_depth_m,
            nose_depth_delta_m,
            self._pose_shoulder_tilt_delta(features),
            self._pose_shoulder_angle_delta(features),
            None if neck_result is None else neck_result.nose_valid_ratio,
            None if neck_result is None else neck_result.neck_valid_ratio,
        )

    def _mark_bad(self, now: float, reason: str) -> None:
        if self.bad_since is None:
            self.bad_since = now
        self.last_bad_at = now
        self.last_bad_reason = reason

    def _hold_recent_bad(
        self,
        now: float,
        features: Optional[FeatureFrame],
        measurements: List[DepthMeasurement],
        deltas: Dict[str, float],
        neck_result: Optional[TurtleNeckFilterResult],
        nose_depth_delta_m: Optional[float] = None,
    ) -> Optional[PostureResult]:
        if self.bad_since is None or self.last_bad_at is None:
            return None
        if now - self.last_bad_at > self.config.bad_recovery_tolerance_seconds:
            return None

        elapsed = self._bad_elapsed(now)
        return self._make_result(
            now,
            "bad",
            self.last_bad_reason or "held_bad",
            elapsed >= self.config.bad_duration_seconds,
            elapsed,
            features,
            measurements,
            deltas,
            neck_result,
            nose_depth_delta_m,
        )

    def _nose_depth_delta(self, neck_result: Optional[TurtleNeckFilterResult]) -> Optional[float]:
        if (
            neck_result is None
            or neck_result.nose_depth_m is None
            or self.nose_depth_baseline is None
        ):
            return None
        return float(neck_result.nose_depth_m - self.nose_depth_baseline)

    def _pose_shoulder_tilt_delta(self, features: Optional[FeatureFrame]) -> Optional[float]:
        if features is None or self.baseline is None:
            return None
        return abs(float(features.shoulder_tilt) - float(self.baseline.shoulder_tilt))

    def _pose_shoulder_angle_delta(self, features: Optional[FeatureFrame]) -> Optional[float]:
        if features is None or self.baseline is None:
            return None
        return abs(self._normalize_angle_delta_deg(features.shoulder_angle_deg - self.baseline.shoulder_angle_deg))

    @staticmethod
    def _shoulder_angle_deg(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
        dy = float(right_shoulder[1] - left_shoulder[1])
        dx = float(right_shoulder[0] - left_shoulder[0])
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _normalize_angle_delta_deg(delta: float) -> float:
        return (float(delta) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _mean_angle_deg(cls, angles: List[float]) -> float:
        if not angles:
            return 0.0
        radians = np.radians(np.array(angles, dtype=float))
        sin_mean = float(np.mean(np.sin(radians)))
        cos_mean = float(np.mean(np.cos(radians)))
        return math.degrees(math.atan2(sin_mean, cos_mean))

    def _clear_bad(self) -> None:
        self.bad_since = None
        self.last_bad_at = None
        self.last_bad_reason = None

    def _bad_elapsed(self, now: float) -> float:
        if self.bad_since is None:
            return 0.0
        return max(0.0, now - self.bad_since)
