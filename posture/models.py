from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FeatureFrame:
    timestamp: float
    head_x: float
    shoulder_x: float
    shoulder_tilt: float
    shoulder_angle_deg: float
    min_confidence: float


@dataclass
class RoiBox:
    timestamp: float
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


@dataclass
class DepthMeasurement:
    timestamp: float
    roi_timestamp: float
    roi: str
    cx: int
    cy: int
    sample_size: int
    confidence: float
    valid_ratio: float
    median_m: Optional[float]
    center_m: Optional[float]


@dataclass
class PostureResult:
    timestamp: float
    state: str
    reason: str
    alarm: bool
    elapsed_bad_seconds: float
    features: Optional[FeatureFrame]
    depth_measurements: Optional[List[DepthMeasurement]] = None
    depth_deltas: Optional[Dict[str, float]] = None
    neck_angle_delta_deg: Optional[float] = None
    neck_pitch_deg: Optional[float] = None
    neck_vector_x_m: Optional[float] = None
    neck_vector_y_m: Optional[float] = None
    neck_vector_z_m: Optional[float] = None
    nose_depth_m: Optional[float] = None
    neck_depth_m: Optional[float] = None
    nose_depth_delta_m: Optional[float] = None
    pose_shoulder_tilt_delta_px: Optional[float] = None
    pose_shoulder_angle_delta_deg: Optional[float] = None
    nose_valid_ratio: Optional[float] = None
    neck_valid_ratio: Optional[float] = None


@dataclass
class PosePacket:
    timestamp: float
    boxes: Any
    scores: Any
    keypoints: Any
