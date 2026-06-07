#!/usr/bin/env python3
"""Development-only isolated IMX500 + D435 turtle-neck detector test."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit("pyrealsense2 is not installed. Install RealSense SDK/python binding first.") from exc

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


MODEL_INPUT_H_W = (480, 640)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


@dataclass
class Snapshot:
    timestamp: float
    keypoints: np.ndarray

    @property
    def nose(self) -> np.ndarray:
        return self.keypoints[NOSE]

    @property
    def left_shoulder(self) -> np.ndarray:
        return self.keypoints[LEFT_SHOULDER]

    @property
    def right_shoulder(self) -> np.ndarray:
        return self.keypoints[RIGHT_SHOULDER]

    @property
    def neck_x(self) -> float:
        return float((self.left_shoulder[0] + self.right_shoulder[0]) / 2.0)

    @property
    def neck_y(self) -> float:
        return float((self.left_shoulder[1] + self.right_shoulder[1]) / 2.0)

    @property
    def neck_confidence(self) -> float:
        return float(min(self.left_shoulder[2], self.right_shoulder[2]))

    @property
    def shoulder_angle_deg(self) -> float:
        dy = float(self.right_shoulder[1] - self.left_shoulder[1])
        dx = float(self.right_shoulder[0] - self.left_shoulder[0])
        return math.degrees(math.atan2(dy, dx))

    @property
    def shoulder_tilt_px(self) -> float:
        return float(self.left_shoulder[1] - self.right_shoulder[1])


@dataclass
class DepthSample:
    x: int
    y: int
    depth_m: Optional[float]
    center_m: Optional[float]
    valid_ratio: float


@dataclass
class Metrics:
    timestamp: float
    valid: bool
    reason: str
    turtle_neck: bool
    nose_delta_m: Optional[float]
    shoulder_delta_m: Optional[float]
    neck_angle_delta_deg: Optional[float]
    shoulder_angle_delta_deg: Optional[float]
    shoulder_tilt_delta_px: Optional[float]
    nose: DepthSample
    neck: DepthSample
    left_shoulder: DepthSample
    right_shoulder: DepthSample
    neck_vector_m: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated turtle-neck condition test")
    parser.add_argument("--model", default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.30)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--ai-width", type=int, default=640)
    parser.add_argument("--ai-height", type=int, default=480)
    parser.add_argument("--sample-size", type=int, default=21)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-min-samples", type=int)
    parser.add_argument("--neck-angle-threshold-deg", type=float, default=15.0)
    parser.add_argument("--nose-delta-threshold-m", type=float, default=0.05)
    parser.add_argument("--nose-delta-direction", choices=("positive", "negative"), default="positive")
    parser.add_argument("--shoulder-delta-threshold-m", type=float, default=0.05)
    parser.add_argument("--shoulder-angle-threshold-deg", type=float, default=12.0)
    parser.add_argument("--shoulder-tilt-threshold-px", type=float, default=22.0)
    parser.add_argument("--nose-valid-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--neck-valid-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--log-csv", type=Path, default=Path("turtleneck-test.csv"))
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-interval", type=float, default=0.5)
    return parser.parse_args()


def normalize_angle_delta_deg(delta: float) -> float:
    return (float(delta) + 180.0) % 360.0 - 180.0


def angle_between_3d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    cos_theta = float(np.dot(a, b) / denom)
    return math.degrees(math.acos(float(np.clip(cos_theta, -1.0, 1.0))))


def mean_angle_deg(angles: list[float]) -> float:
    radians = np.radians(np.array(angles, dtype=float))
    return math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))


def ai_to_depth(x: float, y: float, args: argparse.Namespace) -> tuple[int, int]:
    depth_x = int(np.clip(round(x * args.depth_width / args.ai_width), 0, args.depth_width - 1))
    depth_y = int(np.clip(round(y * args.depth_height / args.ai_height), 0, args.depth_height - 1))
    return depth_x, depth_y


def sample_depth(depth_m: np.ndarray, ai_x: float, ai_y: float, args: argparse.Namespace) -> DepthSample:
    depth_x, depth_y = ai_to_depth(ai_x, ai_y, args)
    half = max(1, args.sample_size // 2)
    x1 = int(np.clip(depth_x - half, 0, args.depth_width - 1))
    x2 = int(np.clip(depth_x + half, 0, args.depth_width - 1))
    y1 = int(np.clip(depth_y - half, 0, args.depth_height - 1))
    y2 = int(np.clip(depth_y + half, 0, args.depth_height - 1))
    crop = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    valid = crop[(crop >= args.min_depth_m) & (crop <= args.max_depth_m)]
    center = float(depth_m[depth_y, depth_x])
    return DepthSample(
        x=depth_x,
        y=depth_y,
        depth_m=float(np.median(valid)) if valid.size else None,
        center_m=center if args.min_depth_m <= center <= args.max_depth_m else None,
        valid_ratio=float(valid.size / crop.size) if crop.size else 0.0,
    )


def pixel_to_camera_m(x: int, y: int, depth_m: float, intrinsics: Any) -> np.ndarray:
    x_m = (float(x) - float(intrinsics.ppx)) * depth_m / float(intrinsics.fx)
    y_m = (float(y) - float(intrinsics.ppy)) * depth_m / float(intrinsics.fy)
    return np.array([x_m, y_m, depth_m], dtype=float)


def select_person(keypoints: Optional[np.ndarray], scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None or len(keypoints) == 0:
        return None
    if scores is None or len(scores) == 0:
        return keypoints[0]
    return keypoints[int(np.argmax(scores))]


def snapshot_from_person(person: Optional[np.ndarray], min_confidence: float) -> Optional[Snapshot]:
    if person is None:
        return None
    required = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)
    if any(float(person[index][2]) < min_confidence for index in required):
        return None
    return Snapshot(time.time(), person)


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "timestamp",
                "valid",
                "reason",
                "turtle_neck",
                "nose_delta_m",
                "shoulder_delta_m",
                "neck_angle_delta_deg",
                "shoulder_angle_delta_deg",
                "shoulder_tilt_delta_px",
                "nose_depth_m",
                "nose_valid_ratio",
                "neck_depth_m",
                "neck_valid_ratio",
                "left_shoulder_depth_m",
                "left_shoulder_valid_ratio",
                "right_shoulder_depth_m",
                "right_shoulder_valid_ratio",
                "neck_vector_x_m",
                "neck_vector_y_m",
                "neck_vector_z_m",
            ]
        )
        self.rows_written = 0

    def write(self, metrics: Metrics) -> None:
        vector = metrics.neck_vector_m
        self.writer.writerow(
            [
                f"{metrics.timestamp:.6f}",
                int(metrics.valid),
                metrics.reason,
                int(metrics.turtle_neck),
                self._fmt(metrics.nose_delta_m, 4),
                self._fmt(metrics.shoulder_delta_m, 4),
                self._fmt(metrics.neck_angle_delta_deg, 3),
                self._fmt(metrics.shoulder_angle_delta_deg, 3),
                self._fmt(metrics.shoulder_tilt_delta_px, 2),
                self._fmt(metrics.nose.depth_m, 4),
                f"{metrics.nose.valid_ratio:.3f}",
                self._fmt(metrics.neck.depth_m, 4),
                f"{metrics.neck.valid_ratio:.3f}",
                self._fmt(metrics.left_shoulder.depth_m, 4),
                f"{metrics.left_shoulder.valid_ratio:.3f}",
                self._fmt(metrics.right_shoulder.depth_m, 4),
                f"{metrics.right_shoulder.valid_ratio:.3f}",
                "" if vector is None else f"{float(vector[0]):.6f}",
                "" if vector is None else f"{float(vector[1]):.6f}",
                "" if vector is None else f"{float(vector[2]):.6f}",
            ]
        )
        self.rows_written += 1

    def close(self) -> None:
        self.file.flush()
        self.file.close()
        print(f"Log saved to {self.path} ({self.rows_written} rows)", flush=True)

    @staticmethod
    def _fmt(value: Optional[float], digits: int) -> str:
        return "" if value is None else f"{value:.{digits}f}"


class TurtleNeckTest:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.latest_lock = threading.Lock()
        self.latest_snapshot: Optional[Snapshot] = None
        self.latest_metrics: Optional[Metrics] = None
        self.last_boxes = None
        self.last_scores = None
        self.last_keypoints = None
        self.last_print = 0.0
        self.baseline_ready = False
        self.baseline_nose_m: Optional[float] = None
        self.baseline_shoulder_m: Optional[float] = None
        self.baseline_neck_vector: Optional[np.ndarray] = None
        self.baseline_shoulder_angle_deg: Optional[float] = None
        self.baseline_shoulder_tilt_px: Optional[float] = None
        self.logger = CsvLogger(args.log_csv)

        self.imx500 = IMX500(args.model)
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "pose estimation"
        elif self.intrinsics.task != "pose estimation":
            raise SystemExit("Network is not a pose estimation task")
        if self.intrinsics.inference_rate is None:
            self.intrinsics.inference_rate = args.fps or 10
        elif args.fps is not None:
            self.intrinsics.inference_rate = args.fps
        self.intrinsics.update_with_defaults()
        self.picam2 = Picamera2(self.imx500.camera_num)

        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)
        self.align = None
        self.depth_intrinsics = None

    def parse_outputs(self, metadata: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)
        if np_outputs is not None:
            keypoints, scores, boxes = postprocess_higherhrnet(
                outputs=np_outputs,
                img_size=MODEL_INPUT_H_W,
                img_w_pad=(0, 0),
                img_h_pad=(0, 0),
                detection_threshold=self.args.detection_threshold,
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
        _, scores, keypoints = self.parse_outputs(request.get_metadata())
        snapshot = snapshot_from_person(select_person(keypoints, scores), self.args.min_keypoint_confidence)
        with self.latest_lock:
            self.latest_snapshot = snapshot
            metrics = self.latest_metrics
        if self.args.no_preview:
            return
        with MappedArray(request, "main") as mapped:
            self.draw_overlay(mapped.array, snapshot, metrics)

    def start(self) -> None:
        profile = self.pipeline.start(self.rs_config)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.align = rs.align(rs.stream.depth)

        preview_config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.intrinsics.inference_rate},
            buffer_count=12,
        )
        print("Loading IMX500 network firmware...")
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(preview_config, show_preview=not self.args.no_preview)
        self.imx500.set_auto_aspect_ratio()
        self.picam2.pre_callback = self.pre_callback

    def stop(self) -> None:
        self.picam2.pre_callback = None
        try:
            self.picam2.stop()
        except Exception as exc:
            print(f"IMX500 stop failed: {exc}", file=sys.stderr, flush=True)
        try:
            self.pipeline.stop()
        except Exception as exc:
            print(f"D435 stop failed: {exc}", file=sys.stderr, flush=True)
        self.logger.close()

    def wait_depth_m(self) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames(500)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)

    def collect_baseline(self) -> None:
        print(f"Hold correct sitting posture for {self.args.baseline_seconds:.1f}s baseline.")
        deadline = time.time() + max(30.0, self.args.baseline_seconds * 6.0)
        target_samples = self.args.baseline_min_samples or max(10, int(round(self.args.baseline_seconds * self.args.depth_fps)))
        valid_samples = 0
        nose_values: list[float] = []
        shoulder_values: list[float] = []
        vectors: list[np.ndarray] = []
        shoulder_angles: list[float] = []
        shoulder_tilts: list[float] = []
        last_print = 0.0

        while time.time() < deadline:
            depth_m = self.wait_depth_m()
            if depth_m is None:
                continue
            with self.latest_lock:
                snapshot = self.latest_snapshot
            metrics = self.compute_metrics(depth_m, snapshot, baseline=False)
            if metrics.valid and metrics.nose.depth_m is not None and metrics.neck_vector_m is not None:
                valid_samples += 1
                nose_values.append(metrics.nose.depth_m)
                shoulder_values.append(metrics.neck.depth_m)
                vectors.append(metrics.neck_vector_m)
                assert snapshot is not None
                shoulder_angles.append(snapshot.shoulder_angle_deg)
                shoulder_tilts.append(snapshot.shoulder_tilt_px)

            valid_seconds = valid_samples / float(max(1, self.args.depth_fps))
            now = time.time()
            if now - last_print >= 1.0:
                print(
                    f"baseline valid={valid_samples}/{target_samples} "
                    f"({valid_seconds:.1f}/{self.args.baseline_seconds:.1f}s) "
                    f"last_reason={metrics.reason} nose_vr={metrics.nose.valid_ratio:.2f} neck_vr={metrics.neck.valid_ratio:.2f}",
                    flush=True,
                )
                last_print = now
            if valid_samples >= target_samples:
                break

        if valid_samples < target_samples or not nose_values or not shoulder_values or not vectors:
            raise SystemExit("Could not record turtle-neck baseline.")
        self.baseline_nose_m = float(np.median(nose_values))
        self.baseline_shoulder_m = float(np.median(shoulder_values))
        self.baseline_neck_vector = np.median(np.stack(vectors, axis=0), axis=0)
        self.baseline_shoulder_angle_deg = mean_angle_deg(shoulder_angles)
        self.baseline_shoulder_tilt_px = float(np.median(shoulder_tilts))
        self.baseline_ready = True
        print(
            "Baseline "
            f"nose={self.baseline_nose_m:.3f}m shoulder={self.baseline_shoulder_m:.3f}m "
            f"shoulder_angle={self.baseline_shoulder_angle_deg:.1f}deg",
            flush=True,
        )

    def compute_metrics(self, depth_m: np.ndarray, snapshot: Optional[Snapshot], baseline: bool = True) -> Metrics:
        empty = DepthSample(0, 0, None, None, 0.0)
        if snapshot is None:
            return Metrics(time.time(), False, "missing_keypoints", False, None, None, None, None, None, empty, empty, empty, empty, None)

        nose = sample_depth(depth_m, float(snapshot.nose[0]), float(snapshot.nose[1]), self.args)
        neck = sample_depth(depth_m, snapshot.neck_x, snapshot.neck_y, self.args)
        left = sample_depth(depth_m, float(snapshot.left_shoulder[0]), float(snapshot.left_shoulder[1]), self.args)
        right = sample_depth(depth_m, float(snapshot.right_shoulder[0]), float(snapshot.right_shoulder[1]), self.args)

        shoulder_depth_values = [sample.depth_m for sample in (left, right, neck) if sample.depth_m is not None]
        if shoulder_depth_values:
            neck = DepthSample(neck.x, neck.y, float(np.median(shoulder_depth_values)), neck.center_m, neck.valid_ratio)

        if nose.depth_m is None:
            return Metrics(time.time(), False, "missing_nose_depth", False, None, None, None, None, None, nose, neck, left, right, None)
        if neck.depth_m is None:
            return Metrics(time.time(), False, "missing_neck_depth", False, None, None, None, None, None, nose, neck, left, right, None)
        if nose.valid_ratio < self.args.nose_valid_ratio_threshold:
            return Metrics(time.time(), False, "nose_valid_ratio_low", False, None, None, None, None, None, nose, neck, left, right, None)
        if neck.valid_ratio < self.args.neck_valid_ratio_threshold:
            return Metrics(time.time(), False, "neck_valid_ratio_low", False, None, None, None, None, None, nose, neck, left, right, None)

        assert self.depth_intrinsics is not None
        nose_3d = pixel_to_camera_m(nose.x, nose.y, nose.depth_m, self.depth_intrinsics)
        neck_3d = pixel_to_camera_m(neck.x, neck.y, neck.depth_m, self.depth_intrinsics)
        vector = nose_3d - neck_3d

        nose_delta = None
        shoulder_delta = None
        angle_delta = None
        shoulder_angle_delta = None
        shoulder_tilt_delta = None
        if baseline and self.baseline_ready:
            assert self.baseline_nose_m is not None
            assert self.baseline_shoulder_m is not None
            assert self.baseline_neck_vector is not None
            assert self.baseline_shoulder_angle_deg is not None
            assert self.baseline_shoulder_tilt_px is not None
            nose_delta = nose.depth_m - self.baseline_nose_m
            shoulder_delta = self.baseline_shoulder_m - neck.depth_m
            angle_delta = angle_between_3d(self.baseline_neck_vector, vector)
            shoulder_angle_delta = abs(normalize_angle_delta_deg(snapshot.shoulder_angle_deg - self.baseline_shoulder_angle_deg))
            shoulder_tilt_delta = abs(snapshot.shoulder_tilt_px - self.baseline_shoulder_tilt_px)

        nose_delta_ok = False
        if nose_delta is not None:
            if self.args.nose_delta_direction == "negative":
                nose_delta_ok = nose_delta <= -abs(self.args.nose_delta_threshold_m)
            else:
                nose_delta_ok = nose_delta >= abs(self.args.nose_delta_threshold_m)

        turtle = (
            angle_delta is not None
            and angle_delta >= self.args.neck_angle_threshold_deg
            and nose_delta_ok
            and shoulder_delta is not None
            and shoulder_delta < self.args.shoulder_delta_threshold_m
            and shoulder_angle_delta is not None
            and shoulder_angle_delta < self.args.shoulder_angle_threshold_deg
            and shoulder_tilt_delta is not None
            and shoulder_tilt_delta < self.args.shoulder_tilt_threshold_px
        )
        return Metrics(time.time(), True, "valid", turtle, nose_delta, shoulder_delta, angle_delta, shoulder_angle_delta, shoulder_tilt_delta, nose, neck, left, right, vector)

    def draw_overlay(self, image: np.ndarray, snapshot: Optional[Snapshot], metrics: Optional[Metrics]) -> None:
        if snapshot is not None:
            points = [
                ("nose", snapshot.nose, (0, 255, 255)),
                ("L sh", snapshot.left_shoulder, (0, 255, 0)),
                ("R sh", snapshot.right_shoulder, (0, 255, 0)),
            ]
            for label, point, color in points:
                x, y = int(round(float(point[0]))), int(round(float(point[1])))
                cv2.circle(image, (x, y), 7, color, -1)
                cv2.putText(image, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            neck = (int(round(snapshot.neck_x)), int(round(snapshot.neck_y)))
            nose = (int(round(float(snapshot.nose[0]))), int(round(float(snapshot.nose[1]))))
            left = (int(round(float(snapshot.left_shoulder[0]))), int(round(float(snapshot.left_shoulder[1]))))
            right = (int(round(float(snapshot.right_shoulder[0]))), int(round(float(snapshot.right_shoulder[1]))))
            cv2.line(image, left, right, (0, 255, 0), 2)
            cv2.line(image, neck, nose, (0, 255, 255), 2)

        if metrics is None:
            return
        color = (0, 0, 255) if metrics.turtle_neck else (0, 255, 0)
        lines = [
            f"TURTLE: {int(metrics.turtle_neck)} {metrics.reason}",
            f"nose_d={self._fmt(metrics.nose_delta_m)}m shoulder_d={self._fmt(metrics.shoulder_delta_m)}m",
            f"neck_a={self._fmt(metrics.neck_angle_delta_deg)}deg sh_a={self._fmt(metrics.shoulder_angle_delta_deg)}deg",
            f"nose_z={self._fmt(metrics.nose.depth_m)}m vr={metrics.nose.valid_ratio:.2f}",
        ]
        y = 28
        for line in lines:
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            y += 26

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        return "na" if value is None else f"{value:.3f}"

    def run(self) -> None:
        self.start()
        try:
            self.collect_baseline()
            print("Monitoring turtle-neck only. Ctrl+C to stop.")
            while True:
                depth_m = self.wait_depth_m()
                if depth_m is None:
                    continue
                with self.latest_lock:
                    snapshot = self.latest_snapshot
                metrics = self.compute_metrics(depth_m, snapshot, baseline=True)
                with self.latest_lock:
                    self.latest_metrics = metrics
                self.logger.write(metrics)
                now = time.time()
                if now - self.last_print >= self.args.print_interval:
                    print(
                        f"turtle={metrics.turtle_neck} reason={metrics.reason} "
                        f"nose_d={self._fmt(metrics.nose_delta_m)}m "
                        f"shoulder_d={self._fmt(metrics.shoulder_delta_m)}m "
                        f"neck_a={self._fmt(metrics.neck_angle_delta_deg)}deg "
                        f"sh_a={self._fmt(metrics.shoulder_angle_delta_deg)}deg "
                        f"nose_vr={metrics.nose.valid_ratio:.2f}",
                        flush=True,
                    )
                    self.last_print = now
        except KeyboardInterrupt:
            print("\nStopping turtle-neck test.")
        finally:
            self.stop()


def main() -> None:
    TurtleNeckTest(parse_args()).run()


if __name__ == "__main__":
    main()
