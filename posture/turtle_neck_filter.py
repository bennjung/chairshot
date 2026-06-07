from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KeypointDepth3D:
    x_px: float
    y_px: float
    depth_x: int
    depth_y: int
    depth_m: Optional[float]
    valid_ratio: float
    confidence: float


@dataclass
class TurtleNeckVector:
    vector_m: np.ndarray
    pitch_deg: Optional[float]


@dataclass
class TurtleNeckFilterResult:
    valid: bool
    reason: str
    angle_delta_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    vector_x_m: Optional[float] = None
    vector_y_m: Optional[float] = None
    vector_z_m: Optional[float] = None
    nose_depth_m: Optional[float] = None
    neck_depth_m: Optional[float] = None
    nose_valid_ratio: Optional[float] = None
    neck_valid_ratio: Optional[float] = None


NOSE_VALID_RATIO_THRESHOLD = 0.25
NECK_VALID_RATIO_THRESHOLD = 0.90


def pixel_to_camera_m(
    x_px: float,
    y_px: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x_m = (x_px - cx) * depth_m / fx
    y_m = (y_px - cy) * depth_m / fy
    return np.array([x_m, y_m, depth_m], dtype=float)


def angle_between_3d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    cos_theta = float(np.dot(a, b) / denom)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def pitch_from_vertical(vector_m: np.ndarray) -> Optional[float]:
    vertical = abs(float(vector_m[1]))
    forward = abs(float(vector_m[2]))
    if vertical <= 1e-9 and forward <= 1e-9:
        return None
    return math.degrees(math.atan2(forward, vertical))


def build_neck_vector(
    nose: KeypointDepth3D,
    neck: KeypointDepth3D,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    nose_valid_ratio_threshold: float = NOSE_VALID_RATIO_THRESHOLD,
    neck_valid_ratio_threshold: float = NECK_VALID_RATIO_THRESHOLD,
) -> TurtleNeckFilterResult:
    if nose.depth_m is None:
        return TurtleNeckFilterResult(False, "missing_nose_depth")
    if neck.depth_m is None:
        return TurtleNeckFilterResult(False, "missing_neck_depth")
    if nose.valid_ratio < nose_valid_ratio_threshold:
        return TurtleNeckFilterResult(
            False,
            "nose_depth_valid_ratio_low",
            nose_depth_m=nose.depth_m,
            neck_depth_m=neck.depth_m,
            nose_valid_ratio=nose.valid_ratio,
            neck_valid_ratio=neck.valid_ratio,
        )
    if neck.valid_ratio < neck_valid_ratio_threshold:
        return TurtleNeckFilterResult(
            False,
            "neck_depth_valid_ratio_low",
            nose_depth_m=nose.depth_m,
            neck_depth_m=neck.depth_m,
            nose_valid_ratio=nose.valid_ratio,
            neck_valid_ratio=neck.valid_ratio,
        )

    nose_3d = pixel_to_camera_m(nose.depth_x, nose.depth_y, nose.depth_m, fx, fy, cx, cy)
    neck_3d = pixel_to_camera_m(neck.depth_x, neck.depth_y, neck.depth_m, fx, fy, cx, cy)
    vector_m = nose_3d - neck_3d
    pitch_deg = pitch_from_vertical(vector_m)

    return TurtleNeckFilterResult(
        True,
        "valid",
        pitch_deg=pitch_deg,
        vector_x_m=float(vector_m[0]),
        vector_y_m=float(vector_m[1]),
        vector_z_m=float(vector_m[2]),
        nose_depth_m=nose.depth_m,
        neck_depth_m=neck.depth_m,
        nose_valid_ratio=nose.valid_ratio,
        neck_valid_ratio=neck.valid_ratio,
    )


def compare_to_baseline(
    current: TurtleNeckFilterResult,
    baseline_vector_m: np.ndarray,
) -> TurtleNeckFilterResult:
    if not current.valid:
        return current
    vector = np.array(
        [
            current.vector_x_m,
            current.vector_y_m,
            current.vector_z_m,
        ],
        dtype=float,
    )
    angle_delta_deg = angle_between_3d(baseline_vector_m, vector)
    current.angle_delta_deg = angle_delta_deg
    return current


def is_turtle_neck_candidate(
    result: TurtleNeckFilterResult,
    head_depth_delta_m: float,
    shoulder_depth_delta_m: float,
    head_depth_threshold_m: float,
    torso_depth_threshold_m: float,
    angle_delta_threshold_deg: float = 15.0,
) -> bool:
    if not result.valid or result.angle_delta_deg is None:
        return False
    return (
        head_depth_delta_m >= head_depth_threshold_m
        and shoulder_depth_delta_m < torso_depth_threshold_m
        and result.angle_delta_deg >= angle_delta_threshold_deg
    )
