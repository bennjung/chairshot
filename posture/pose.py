from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from .config import MonitorConfig
from .models import RoiBox

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def select_person(keypoints: Optional[np.ndarray], scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None or len(keypoints) == 0:
        return None
    if scores is None or len(scores) == 0:
        return keypoints[0]
    return keypoints[int(np.argmax(scores))]


def valid_point(keypoints: np.ndarray, index: int, config: MonitorConfig) -> bool:
    return float(keypoints[index][2]) >= config.min_keypoint_confidence


def make_roi_box(
    name: str,
    points: List[np.ndarray],
    width: int,
    height: int,
    padding: int,
    timestamp: float,
    config: MonitorConfig,
) -> Optional[RoiBox]:
    valid = [point for point in points if float(point[2]) >= config.min_keypoint_confidence]
    if not valid:
        return None

    xs = [float(point[0]) for point in valid]
    ys = [float(point[1]) for point in valid]
    confidence = float(np.mean([float(point[2]) for point in valid]))
    return RoiBox(
        timestamp=timestamp,
        name=name,
        x1=clamp(min(xs) - padding, 0, width - 1),
        y1=clamp(min(ys) - padding, 0, height - 1),
        x2=clamp(max(xs) + padding, 0, width - 1),
        y2=clamp(max(ys) + padding, 0, height - 1),
        confidence=confidence,
    )


def extract_roi_boxes(keypoints: np.ndarray, frame_shape: Tuple[int, int, int], config: MonitorConfig) -> List[RoiBox]:
    height, width = frame_shape[:2]
    timestamp = time.time()
    boxes: List[RoiBox] = []

    head_points = [
        keypoints[index]
        for index in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
        if valid_point(keypoints, index, config)
    ]
    head_box = make_roi_box("head", head_points, width, height, config.roi_box_padding, timestamp, config)
    if head_box is not None:
        boxes.append(head_box)

    if valid_point(keypoints, LEFT_SHOULDER, config) and valid_point(keypoints, RIGHT_SHOULDER, config):
        shoulder_l = keypoints[LEFT_SHOULDER]
        shoulder_r = keypoints[RIGHT_SHOULDER]
        shoulder_box = make_roi_box(
            "shoulder",
            [shoulder_l, shoulder_r],
            width,
            height,
            config.roi_box_padding,
            timestamp,
            config,
        )
        if shoulder_box is not None:
            boxes.append(shoulder_box)
            chest_y1 = shoulder_box.y2
            chest_y2 = clamp(chest_y1 + config.chest_height, 0, height - 1)
            boxes.append(
                RoiBox(
                    timestamp=timestamp,
                    name="chest",
                    x1=shoulder_box.x1,
                    y1=chest_y1,
                    x2=shoulder_box.x2,
                    y2=chest_y2,
                    confidence=shoulder_box.confidence,
                )
            )

    return boxes
