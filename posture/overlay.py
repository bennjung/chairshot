from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .config import MonitorConfig
from .models import DepthMeasurement, PostureResult, RoiBox
from .pose import LEFT_SHOULDER, NOSE, RIGHT_SHOULDER


def draw_core_keypoints_overlay(
    image: np.ndarray,
    keypoints: Optional[np.ndarray],
    config: MonitorConfig,
) -> None:
    if keypoints is None:
        return

    points = [
        ("nose", NOSE, (0, 255, 255)),
        ("L shoulder", LEFT_SHOULDER, (0, 255, 0)),
        ("R shoulder", RIGHT_SHOULDER, (0, 255, 0)),
    ]
    valid_xy = {}
    for label, index, color in points:
        x, y, confidence = keypoints[index]
        x_i = int(np.clip(round(float(x)), 0, image.shape[1] - 1))
        y_i = int(np.clip(round(float(y)), 0, image.shape[0] - 1))
        is_valid = float(confidence) >= config.min_keypoint_confidence
        point_color = color if is_valid else (120, 120, 120)
        cv2.circle(image, (x_i, y_i), 7, point_color, -1)
        cv2.circle(image, (x_i, y_i), 9, (0, 0, 0), 2)
        cv2.putText(
            image,
            f"{label} {float(confidence):.2f}",
            (x_i + 10, max(18, y_i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            point_color,
            1,
            cv2.LINE_AA,
        )
        if is_valid:
            valid_xy[index] = (x_i, y_i)

    if LEFT_SHOULDER in valid_xy and RIGHT_SHOULDER in valid_xy:
        cv2.line(image, valid_xy[LEFT_SHOULDER], valid_xy[RIGHT_SHOULDER], (0, 255, 0), 2)
        shoulder_mid = (
            (valid_xy[LEFT_SHOULDER][0] + valid_xy[RIGHT_SHOULDER][0]) // 2,
            (valid_xy[LEFT_SHOULDER][1] + valid_xy[RIGHT_SHOULDER][1]) // 2,
        )
        cv2.circle(image, shoulder_mid, 5, (0, 255, 0), -1)
        if NOSE in valid_xy:
            cv2.line(image, valid_xy[NOSE], shoulder_mid, (0, 255, 255), 2)


def draw_baseline_timer_overlay(
    image: np.ndarray,
    active: bool,
    pose_seconds: float,
    depth_seconds: float,
    target_seconds: float,
    depth_active: bool,
) -> None:
    if not active or target_seconds <= 0:
        return

    pose_done = min(max(pose_seconds, 0.0), target_seconds)
    lines = [f"BASE POSE {pose_done:>4.1f}/{target_seconds:.1f}s"]
    if depth_active:
        depth_done = min(max(depth_seconds, 0.0), target_seconds)
        lines.append(f"BASE DEPTH {depth_done:>4.1f}/{target_seconds:.1f}s")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 10
    line_gap = 8
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + padding * 2
    height = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1) + padding * 2
    x1 = image.shape[1] - width - 12
    y1 = 12
    x2 = image.shape[1] - 12
    y2 = y1 + height

    cv2.rectangle(image, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    y = y1 + padding + sizes[0][1]
    for line, size in zip(lines, sizes):
        cv2.putText(
            image,
            line,
            (x2 - padding - size[0], y),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += size[1] + line_gap


def draw_status_overlay(image: np.ndarray, result: Optional[PostureResult]) -> None:
    if result is None:
        return

    if result.alarm:
        color = (0, 0, 255)
    elif result.state == "bad":
        color = (0, 165, 255)
    elif result.state == "normal":
        color = (0, 255, 0)
    else:
        color = (180, 180, 180)

    lines = [
        f"STATE: {result.state}",
        f"REASON: {result.reason}",
        f"BAD: {result.elapsed_bad_seconds:.1f}s ALARM: {int(result.alarm)}",
    ]
    if result.neck_angle_delta_deg is not None:
        lines.append(f"NECK ANGLE: {result.neck_angle_delta_deg:.1f}deg")
    if result.nose_depth_delta_m is not None:
        lines.append(f"NOSE DELTA: {result.nose_depth_delta_m:.3f}m")
    if result.pose_shoulder_tilt_delta_px is not None:
        lines.append(f"SH TILT: {result.pose_shoulder_tilt_delta_px:.1f}px")
    if result.pose_shoulder_angle_delta_deg is not None:
        lines.append(f"SH ANGLE: {result.pose_shoulder_angle_delta_deg:.1f}deg")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    padding = 10
    line_gap = 8
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + padding * 2
    height = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1) + padding * 2
    x1, y1 = 12, 12
    x2, y2 = x1 + width, y1 + height
    cv2.rectangle(image, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    y = y1 + padding + sizes[0][1]
    for line in lines:
        cv2.putText(image, line, (x1 + padding, y), font, scale, color, thickness, cv2.LINE_AA)
        y += sizes[0][1] + line_gap


def draw_roi_overlay(image: np.ndarray, roi_boxes: List[RoiBox]) -> None:
    colors = {
        "head": (0, 255, 255),
        "shoulder": (0, 255, 0),
        "chest": (255, 128, 0),
        "nose": (0, 128, 255),
        "neck": (255, 0, 255),
    }
    for box in roi_boxes:
        color = colors.get(box.name, (255, 255, 255))
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        cv2.circle(image, (box.cx, box.cy), 5, color, -1)
        cv2.putText(
            image,
            f"{box.name} ai=({box.cx},{box.cy})",
            (box.x1, max(18, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_depth_overlay(image: np.ndarray, measurements: List[DepthMeasurement], config: MonitorConfig) -> None:
    if not measurements:
        return

    scale_x = image.shape[1] / config.depth_width
    scale_y = image.shape[0] / config.depth_height
    for measurement in measurements:
        color = (0, 0, 255) if measurement.median_m is None else (255, 255, 0)
        x = int(np.clip(round(measurement.cx * scale_x), 0, image.shape[1] - 1))
        y = int(np.clip(round(measurement.cy * scale_y), 0, image.shape[0] - 1))
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 16, 2)
        depth_text = "invalid" if measurement.median_m is None else f"{measurement.median_m:.3f}m"
        cv2.putText(
            image,
            f"{measurement.roi} depth={depth_text}",
            (x + 8, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
