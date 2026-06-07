from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MonitorConfig:
    baseline_seconds: float = 5.0
    bad_duration_seconds: float = 5.0
    bad_recovery_tolerance_seconds: float = 1.0
    min_keypoint_confidence: float = 0.25
    head_shift_px: float = 35.0
    torso_shift_px: float = 30.0
    shoulder_tilt_px: float = 22.0
    shoulder_tilt_combo_angle_deg: float = 8.0
    shoulder_tilt_angle_deg: float = 12.0
    out_of_range_tolerance_seconds: float = 0.5
    valid_missing_frames: int = 15
    depth_enabled: bool = True
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    ai_width: int = 640
    ai_height: int = 480
    min_depth_m: float = 0.15
    max_depth_m: float = 3.0
    head_depth_roi_size: int = 40
    body_depth_roi_size: int = 60
    keypoint_depth_sample_size: int = 21
    nose_depth_valid_ratio_threshold: float = 0.25
    neck_depth_valid_ratio_threshold: float = 0.90
    neck_angle_threshold_deg: float = 15.0
    turtle_neck_nose_depth_change_m: float = 0.05
    turtle_neck_score_threshold: float = 0.65
    turtle_neck_head_weight: float = 0.60
    turtle_neck_shoulder_weight: float = 0.25
    turtle_neck_shoulder_angle_weight: float = 0.15
    head_depth_threshold_m: float = 0.07
    torso_depth_threshold_m: float = 0.05
    torso_stable_threshold_m: float = 0.015
    roi_box_padding: int = 25
    chest_height: int = 110
    led_switch_delay_seconds: float = 1.0
    event_publish_stable_seconds: float = 5.0
    green_led_pin: Optional[int] = None
    yellow_led_pin: Optional[int] = None
    red_led_pin: Optional[int] = None
    buzzer_pin: Optional[int] = None

    @classmethod
    def from_file(cls, path: Path) -> "MonitorConfig":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})
