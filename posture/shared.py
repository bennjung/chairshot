from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .models import DepthMeasurement, FeatureFrame, PosePacket, PostureResult, RoiBox


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_pose_packet: Optional[PosePacket] = None
    latest_selected_keypoints: Optional[Any] = None
    latest_roi_boxes: List[RoiBox] = field(default_factory=list)
    latest_feature_frame: Optional[FeatureFrame] = None
    latest_result: Optional[PostureResult] = None
    latest_depth_measurements: List[DepthMeasurement] = field(default_factory=list)
    latest_coord_depth_measurements: List[DepthMeasurement] = field(default_factory=list)
    baseline_active: bool = False
    baseline_depth_active: bool = False
    baseline_pose_seconds: float = 0.0
    baseline_depth_seconds: float = 0.0
    baseline_target_seconds: float = 0.0
