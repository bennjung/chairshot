#!/usr/bin/env python3
"""
Intel RealSense D435 depth frame smoke test.

Development-only hardware smoke test.

This script verifies that pyrealsense2 can open the D435 depth stream and read
valid depth values. It does not use the Raspberry Pi AI Camera.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is not installed. Install Intel RealSense SDK/python binding first."
    ) from exc


@dataclass
class DepthStats:
    valid_ratio: float
    center_m: Optional[float]
    roi_median_m: Optional[float]
    min_m: Optional[float]
    max_m: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense D435 depth frame test")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--roi-size", type=int, default=40, help="Center ROI size in pixels")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV depth preview")
    return parser.parse_args()


def depth_stats(depth_m: np.ndarray, roi_size: int) -> DepthStats:
    height, width = depth_m.shape
    center_y = height // 2
    center_x = width // 2
    center = float(depth_m[center_y, center_x])
    center_m = center if center > 0 else None

    half = max(1, roi_size // 2)
    y1 = max(0, center_y - half)
    y2 = min(height, center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(width, center_x + half)
    roi = depth_m[y1:y2, x1:x2]
    valid = depth_m[depth_m > 0]
    roi_valid = roi[roi > 0]

    return DepthStats(
        valid_ratio=float(valid.size / depth_m.size),
        center_m=center_m,
        roi_median_m=float(np.median(roi_valid)) if roi_valid.size else None,
        min_m=float(np.min(valid)) if valid.size else None,
        max_m=float(np.max(valid)) if valid.size else None,
    )


def format_meters(value: Optional[float]) -> str:
    if value is None:
        return "invalid"
    return f"{value:.3f}m"


def maybe_show_preview(depth_m: np.ndarray) -> bool:
    import cv2

    depth_mm = np.clip(depth_m * 1000.0, 0, 3000).astype(np.uint16)
    depth_8u = cv2.convertScaleAbs(depth_mm, alpha=255.0 / 3000.0)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    cv2.imshow("D435 depth", depth_color)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def main() -> None:
    args = parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print(f"Starting D435 depth stream: {args.width}x{args.height}@{args.fps}")
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")

    start = time.time()
    frame_count = 0
    last_print = 0.0

    try:
        while time.time() - start < args.seconds:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                print("No depth frame")
                continue

            frame_count += 1
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            now = time.time()
            if now - last_print >= 1.0:
                stats = depth_stats(depth_m, args.roi_size)
                elapsed = max(now - start, 1e-6)
                print(
                    "depth "
                    f"frames={frame_count} fps={frame_count / elapsed:.1f} "
                    f"valid={stats.valid_ratio * 100:.1f}% "
                    f"center={format_meters(stats.center_m)} "
                    f"roi_median={format_meters(stats.roi_median_m)} "
                    f"min={format_meters(stats.min_m)} "
                    f"max={format_meters(stats.max_m)}",
                    flush=True,
                )
                last_print = now

            if args.preview and not maybe_show_preview(depth_m):
                break
    finally:
        pipeline.stop()
        if args.preview:
            import cv2

            cv2.destroyAllWindows()
        print("D435 depth stream stopped.")


if __name__ == "__main__":
    main()
