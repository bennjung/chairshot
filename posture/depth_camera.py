from __future__ import annotations

import sys
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from .config import MonitorConfig


class D435DepthCamera:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.pipeline = None
        self.depth_scale: Optional[float] = None
        self.depth_intrinsics = None
        self.started = False

    @property
    def available(self) -> bool:
        return rs is not None

    def start(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.depth,
            self.config.depth_width,
            self.config.depth_height,
            rs.format.z16,
            self.config.depth_fps,
        )
        profile = self.pipeline.start(rs_config)
        self.started = True
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        print(
            f"D435 depth stream started: {self.config.depth_width}x{self.config.depth_height}@{self.config.depth_fps}, "
            f"scale={self.depth_scale}",
            flush=True,
        )

    def wait_depth_m(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        if self.pipeline is None or self.depth_scale is None:
            return None
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        depth_raw = np.asanyarray(depth_frame.get_data())
        return depth_raw.astype(np.float32) * self.depth_scale

    def stop(self) -> None:
        if self.pipeline is not None and self.started:
            try:
                self.pipeline.stop()
            except Exception as exc:
                print(f"D435 depth stream stop failed: {exc}", file=sys.stderr, flush=True)
        self.started = False
        self.pipeline = None
        self.depth_intrinsics = None


def report_missing_realsense() -> None:
    print("pyrealsense2 is not installed; depth posture monitoring disabled.", file=sys.stderr, flush=True)
