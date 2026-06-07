from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .config import MonitorConfig
from .models import DepthMeasurement, RoiBox


def map_roi_to_depth(roi: RoiBox, sample_size: int, config: MonitorConfig) -> Tuple[int, int, int, int, int, int]:
    scale_x = config.depth_width / config.ai_width
    scale_y = config.depth_height / config.ai_height
    cx = int(np.clip(round(roi.cx * scale_x), 0, config.depth_width - 1))
    cy = int(np.clip(round(roi.cy * scale_y), 0, config.depth_height - 1))
    half = max(1, sample_size // 2)
    x1 = int(np.clip(cx - half, 0, config.depth_width - 1))
    x2 = int(np.clip(cx + half, 0, config.depth_width - 1))
    y1 = int(np.clip(cy - half, 0, config.depth_height - 1))
    y2 = int(np.clip(cy + half, 0, config.depth_height - 1))
    if x2 <= x1:
        x2 = min(config.depth_width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(config.depth_height - 1, y1 + 1)
    return x1, y1, x2, y2, cx, cy


def measure_depth(depth_m: np.ndarray, roi: RoiBox, config: MonitorConfig) -> DepthMeasurement:
    sample_size = config.head_depth_roi_size if roi.name == "head" else config.body_depth_roi_size
    x1, y1, x2, y2, cx, cy = map_roi_to_depth(roi, sample_size, config)
    crop = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    valid = crop[(crop >= config.min_depth_m) & (crop <= config.max_depth_m)]
    center = float(depth_m[cy, cx])
    center_m = center if config.min_depth_m <= center <= config.max_depth_m else None
    return DepthMeasurement(
        timestamp=time.time(),
        roi_timestamp=roi.timestamp,
        roi=roi.name,
        cx=cx,
        cy=cy,
        sample_size=sample_size,
        confidence=roi.confidence,
        valid_ratio=float(valid.size / crop.size) if crop.size else 0.0,
        median_m=float(np.median(valid)) if valid.size else None,
        center_m=center_m,
    )


def measure_keypoint_depth(
    depth_m: np.ndarray,
    name: str,
    x_px: float,
    y_px: float,
    confidence: float,
    config: MonitorConfig,
) -> DepthMeasurement:
    scale_x = config.depth_width / config.ai_width
    scale_y = config.depth_height / config.ai_height
    cx = int(np.clip(round(x_px * scale_x), 0, config.depth_width - 1))
    cy = int(np.clip(round(y_px * scale_y), 0, config.depth_height - 1))
    sample_size = config.keypoint_depth_sample_size
    half = max(1, sample_size // 2)
    x1 = int(np.clip(cx - half, 0, config.depth_width - 1))
    x2 = int(np.clip(cx + half, 0, config.depth_width - 1))
    y1 = int(np.clip(cy - half, 0, config.depth_height - 1))
    y2 = int(np.clip(cy + half, 0, config.depth_height - 1))
    crop = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    valid = crop[(crop >= config.min_depth_m) & (crop <= config.max_depth_m)]
    center = float(depth_m[cy, cx])
    center_m = center if config.min_depth_m <= center <= config.max_depth_m else None
    return DepthMeasurement(
        timestamp=time.time(),
        roi_timestamp=time.time(),
        roi=name,
        cx=cx,
        cy=cy,
        sample_size=sample_size,
        confidence=confidence,
        valid_ratio=float(valid.size / crop.size) if crop.size else 0.0,
        median_m=float(np.median(valid)) if valid.size else None,
        center_m=center_m,
    )
