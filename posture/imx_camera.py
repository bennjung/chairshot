from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    from picamera2 import CompletedRequest, MappedArray, Picamera2
    from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
    from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
except ImportError as exc:
    print(
        "Picamera2 IMX500 dependencies are missing. Run this on Raspberry Pi OS "
        "with imx500-all and Picamera2 installed.",
        file=sys.stderr,
    )
    print(f"Original import error: {exc}", file=sys.stderr)
    raise

from .config import MonitorConfig
from .models import PosePacket
from .overlay import draw_baseline_timer_overlay, draw_core_keypoints_overlay, draw_depth_overlay, draw_roi_overlay, draw_status_overlay
from .shared import SharedState

MODEL_INPUT_H_W = (480, 640)


def load_labels(intrinsics: NetworkIntrinsics, labels_path: Optional[str]) -> None:
    if labels_path:
        with open(labels_path, "r", encoding="utf-8") as f:
            intrinsics.labels = f.read().splitlines()
    elif intrinsics.labels is None:
        fallback = Path(__file__).resolve().parent.parent / "assets" / "coco_labels.txt"
        if fallback.exists():
            intrinsics.labels = fallback.read_text(encoding="utf-8").splitlines()
        else:
            intrinsics.labels = ["person"]


class Imx500Camera:
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str],
        fps: Optional[int],
        detection_threshold: float,
        no_preview: bool,
        no_overlay: bool,
        config: MonitorConfig,
        shared: SharedState,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.no_preview = no_preview
        self.no_overlay = no_overlay
        self.config = config
        self.shared = shared
        self.last_boxes = None
        self.last_scores = None
        self.last_keypoints = None

        self.imx500 = IMX500(model_path)
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "pose estimation"
        elif self.intrinsics.task != "pose estimation":
            raise SystemExit("Network is not a pose estimation task")

        load_labels(self.intrinsics, labels_path)
        if self.intrinsics.inference_rate is None:
            self.intrinsics.inference_rate = fps or 10
        elif fps is not None:
            self.intrinsics.inference_rate = fps
        self.intrinsics.update_with_defaults()

        self.picam2 = Picamera2(self.imx500.camera_num)

    def print_intrinsics(self) -> None:
        print(self.intrinsics)

    def start(self) -> None:
        preview_config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.intrinsics.inference_rate},
            buffer_count=12,
        )
        print("Loading IMX500 network firmware...")
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(preview_config, show_preview=not self.no_preview)
        self.imx500.set_auto_aspect_ratio()
        self.picam2.pre_callback = self.pre_callback

    def stop(self) -> None:
        self.picam2.pre_callback = None
        try:
            self.picam2.stop()
        except Exception as exc:
            print(f"IMX500 camera stop failed: {exc}", file=sys.stderr, flush=True)

    def parse_outputs(self, metadata: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)
        if np_outputs is not None:
            keypoints, scores, boxes = postprocess_higherhrnet(
                outputs=np_outputs,
                img_size=MODEL_INPUT_H_W,
                img_w_pad=(0, 0),
                img_h_pad=(0, 0),
                detection_threshold=self.detection_threshold,
                network_postprocess=True,
            )
            if scores is not None and len(scores) > 0:
                self.last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
                self.last_boxes = [np.array(box) for box in boxes]
                self.last_scores = np.array(scores)
            else:
                self.last_keypoints = None
                self.last_boxes = None
                self.last_scores = None
        return self.last_boxes, self.last_scores, self.last_keypoints

    def pre_callback(self, request: CompletedRequest) -> None:
        boxes, scores, keypoints = self.parse_outputs(request.get_metadata())
        with self.shared.lock:
            self.shared.latest_pose_packet = PosePacket(time.time(), boxes, scores, keypoints)
            selected_keypoints = self.shared.latest_selected_keypoints
            roi_boxes = list(self.shared.latest_roi_boxes)
            depth_measurements = list(self.shared.latest_depth_measurements)
            result = self.shared.latest_result
            baseline_active = self.shared.baseline_active
            baseline_pose_seconds = self.shared.baseline_pose_seconds
            baseline_depth_seconds = self.shared.baseline_depth_seconds
            baseline_target_seconds = self.shared.baseline_target_seconds
            baseline_depth_active = self.shared.baseline_depth_active

        if self.no_preview or self.no_overlay:
            return

        with MappedArray(request, "main") as mapped:
            draw_core_keypoints_overlay(mapped.array, selected_keypoints, self.config)
            draw_roi_overlay(mapped.array, roi_boxes)
            draw_depth_overlay(mapped.array, depth_measurements, self.config)
            draw_status_overlay(mapped.array, result)
            draw_baseline_timer_overlay(
                mapped.array,
                baseline_active,
                baseline_pose_seconds,
                baseline_depth_seconds,
                baseline_target_seconds,
                baseline_depth_active,
            )
