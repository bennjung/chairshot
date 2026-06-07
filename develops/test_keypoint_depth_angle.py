#!/usr/bin/env python3
"""Development-only IMX500 keypoint depth/angle measurement test."""

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
class KeypointSnapshot:
    timestamp: float
    nose_x: float
    nose_y: float
    nose_conf: float
    left_shoulder_x: float
    left_shoulder_y: float
    left_shoulder_conf: float
    right_shoulder_x: float
    right_shoulder_y: float
    right_shoulder_conf: float

    @property
    def neck_x(self) -> float:
        return (self.left_shoulder_x + self.right_shoulder_x) / 2.0

    @property
    def neck_y(self) -> float:
        return (self.left_shoulder_y + self.right_shoulder_y) / 2.0

    @property
    def neck_conf(self) -> float:
        return min(self.left_shoulder_conf, self.right_shoulder_conf)


@dataclass
class DepthPoint:
    depth_x: int
    depth_y: int
    depth_m: Optional[float]
    valid_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IMX500 keypoint + D435 depth 3D angle test")
    parser.add_argument("--model", default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.30)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--ai-width", type=int, default=640)
    parser.add_argument("--ai-height", type=int, default=480)
    parser.add_argument("--depth-sample-size", type=int, default=21)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--log-csv", type=Path, default=Path("keypoint_depth_angle_log.csv"))
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-interval", type=float, default=0.5)
    return parser.parse_args()


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "timestamp",
                "kp_age_s",
                "baseline_ready",
                "nose_ai_x",
                "nose_ai_y",
                "nose_conf",
                "nose_depth_x",
                "nose_depth_y",
                "nose_depth_m",
                "nose_valid_ratio",
                "left_shoulder_ai_x",
                "left_shoulder_ai_y",
                "left_shoulder_conf",
                "left_shoulder_depth_x",
                "left_shoulder_depth_y",
                "left_shoulder_depth_m",
                "left_shoulder_valid_ratio",
                "right_shoulder_ai_x",
                "right_shoulder_ai_y",
                "right_shoulder_conf",
                "right_shoulder_depth_x",
                "right_shoulder_depth_y",
                "right_shoulder_depth_m",
                "right_shoulder_valid_ratio",
                "neck_ai_x",
                "neck_ai_y",
                "neck_conf",
                "neck_depth_x",
                "neck_depth_y",
                "neck_depth_m",
                "neck_valid_ratio",
                "neck_vec_x_m",
                "neck_vec_y_m",
                "neck_vec_z_m",
                "neck_vec_norm_m",
                "neck_angle_baseline_deg",
                "neck_pitch_deg",
                "valid",
            ]
        )
        self.file.flush()
        self.rows_written = 0

    def write(self, row: list[Any]) -> None:
        self.writer.writerow(row)
        self.rows_written += 1

    def close(self) -> None:
        self.file.flush()
        self.file.close()
        print(f"Log saved to {self.path} ({self.rows_written} rows)", flush=True)


args = parse_args()
latest_lock = threading.Lock()
latest_keypoints: Optional[KeypointSnapshot] = None
latest_depth_metrics: Optional[Dict[str, Any]] = None

last_boxes = None
last_scores = None
last_keypoints = None
last_print = 0.0

imx500 = IMX500(args.model)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "pose estimation"
elif intrinsics.task != "pose estimation":
    print("Network is not a pose estimation task", file=sys.stderr)
    sys.exit(1)

if intrinsics.inference_rate is None:
    intrinsics.inference_rate = args.fps or 10
elif args.fps is not None:
    intrinsics.inference_rate = args.fps
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)


def ai_output_tensor_parse(metadata: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=MODEL_INPUT_H_W,
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=args.detection_threshold,
            network_postprocess=True,
        )
        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(box) for box in boxes]
            last_scores = np.array(scores)
        else:
            last_keypoints = None
            last_boxes = None
            last_scores = None
    return last_boxes, last_scores, last_keypoints


def select_person(keypoints: Optional[np.ndarray], scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None or len(keypoints) == 0:
        return None
    if scores is None or len(scores) == 0:
        return keypoints[0]
    return keypoints[int(np.argmax(scores))]


def valid_point(keypoints: np.ndarray, index: int) -> bool:
    return float(keypoints[index][2]) >= args.min_keypoint_confidence


def snapshot_from_person(person: Optional[np.ndarray]) -> Optional[KeypointSnapshot]:
    if person is None:
        return None
    if not all(valid_point(person, index) for index in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)):
        return None
    nose = person[NOSE]
    left_shoulder = person[LEFT_SHOULDER]
    right_shoulder = person[RIGHT_SHOULDER]
    return KeypointSnapshot(
        timestamp=time.time(),
        nose_x=float(nose[0]),
        nose_y=float(nose[1]),
        nose_conf=float(nose[2]),
        left_shoulder_x=float(left_shoulder[0]),
        left_shoulder_y=float(left_shoulder[1]),
        left_shoulder_conf=float(left_shoulder[2]),
        right_shoulder_x=float(right_shoulder[0]),
        right_shoulder_y=float(right_shoulder[1]),
        right_shoulder_conf=float(right_shoulder[2]),
    )


def ai_to_depth(x: float, y: float) -> tuple[int, int]:
    depth_x = int(np.clip(round(x * args.depth_width / args.ai_width), 0, args.depth_width - 1))
    depth_y = int(np.clip(round(y * args.depth_height / args.ai_height), 0, args.depth_height - 1))
    return depth_x, depth_y


def sample_depth(depth_m: np.ndarray, ai_x: float, ai_y: float) -> DepthPoint:
    depth_x, depth_y = ai_to_depth(ai_x, ai_y)
    half = max(1, args.depth_sample_size // 2)
    x1 = int(np.clip(depth_x - half, 0, args.depth_width - 1))
    x2 = int(np.clip(depth_x + half, 0, args.depth_width - 1))
    y1 = int(np.clip(depth_y - half, 0, args.depth_height - 1))
    y2 = int(np.clip(depth_y + half, 0, args.depth_height - 1))
    crop = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    valid = crop[(crop >= args.min_depth_m) & (crop <= args.max_depth_m)]
    return DepthPoint(
        depth_x=depth_x,
        depth_y=depth_y,
        depth_m=float(np.median(valid)) if valid.size else None,
        valid_ratio=float(valid.size / crop.size) if crop.size else 0.0,
    )


def pixel_to_camera_m(depth_intrinsics: Any, depth_x: int, depth_y: int, depth_m: float) -> np.ndarray:
    x_m = (float(depth_x) - float(depth_intrinsics.ppx)) * depth_m / float(depth_intrinsics.fx)
    y_m = (float(depth_y) - float(depth_intrinsics.ppy)) * depth_m / float(depth_intrinsics.fy)
    return np.array([x_m, y_m, depth_m], dtype=float)


def angle_between_3d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    cos_theta = float(np.dot(a, b) / denom)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def pitch_from_vertical(vector_m: np.ndarray) -> Optional[float]:
    vertical = abs(float(vector_m[1]))
    forward = abs(float(vector_m[2]))
    if vertical <= 1e-9 and forward <= 1e-9:
        return None
    return math.degrees(math.atan2(forward, vertical))


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def draw_overlay(image: np.ndarray, snapshot: Optional[KeypointSnapshot], metrics: Optional[Dict[str, Any]]) -> None:
    if snapshot is None:
        cv2.putText(image, "keypoints invalid", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return

    nose = (round(snapshot.nose_x), round(snapshot.nose_y))
    left_shoulder = (round(snapshot.left_shoulder_x), round(snapshot.left_shoulder_y))
    right_shoulder = (round(snapshot.right_shoulder_x), round(snapshot.right_shoulder_y))
    neck = (round(snapshot.neck_x), round(snapshot.neck_y))

    cv2.line(image, left_shoulder, right_shoulder, (0, 255, 0), 3)
    cv2.line(image, neck, nose, (0, 255, 255), 3)
    cv2.circle(image, nose, 8, (0, 255, 255), -1)
    cv2.circle(image, neck, 8, (255, 0, 255), -1)
    cv2.circle(image, left_shoulder, 7, (0, 255, 0), -1)
    cv2.circle(image, right_shoulder, 7, (0, 255, 0), -1)

    text = "waiting depth"
    if metrics is not None:
        angle = metrics.get("neck_angle_baseline_deg")
        pitch = metrics.get("neck_pitch_deg")
        nose_depth = metrics.get("nose_depth_m")
        neck_depth = metrics.get("neck_depth_m")
        base = "ready" if metrics.get("baseline_ready") else "baseline"
        text = (
            f"{base} nose_z={fmt(nose_depth, 3)}m neck_z={fmt(neck_depth, 3)}m "
            f"angle_d={fmt(angle, 1)}deg pitch={fmt(pitch, 1)}deg"
        )
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


def pre_callback(request: CompletedRequest) -> None:
    _, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    person = select_person(keypoints, scores)
    snapshot = snapshot_from_person(person)
    with latest_lock:
        global latest_keypoints
        if snapshot is not None:
            latest_keypoints = snapshot
        draw_snapshot = latest_keypoints
        draw_metrics = latest_depth_metrics

    with MappedArray(request, "main") as mapped:
        draw_overlay(mapped.array, draw_snapshot, draw_metrics)


def build_row(
    snapshot: KeypointSnapshot,
    nose: DepthPoint,
    left: DepthPoint,
    right: DepthPoint,
    neck: DepthPoint,
    vector: np.ndarray,
    angle_deg: Optional[float],
    pitch_deg: Optional[float],
    baseline_ready: bool,
) -> list[Any]:
    norm = float(np.linalg.norm(vector))
    return [
        f"{time.time():.6f}",
        f"{time.time() - snapshot.timestamp:.4f}",
        int(baseline_ready),
        f"{snapshot.nose_x:.2f}",
        f"{snapshot.nose_y:.2f}",
        f"{snapshot.nose_conf:.3f}",
        nose.depth_x,
        nose.depth_y,
        fmt(nose.depth_m),
        f"{nose.valid_ratio:.3f}",
        f"{snapshot.left_shoulder_x:.2f}",
        f"{snapshot.left_shoulder_y:.2f}",
        f"{snapshot.left_shoulder_conf:.3f}",
        left.depth_x,
        left.depth_y,
        fmt(left.depth_m),
        f"{left.valid_ratio:.3f}",
        f"{snapshot.right_shoulder_x:.2f}",
        f"{snapshot.right_shoulder_y:.2f}",
        f"{snapshot.right_shoulder_conf:.3f}",
        right.depth_x,
        right.depth_y,
        fmt(right.depth_m),
        f"{right.valid_ratio:.3f}",
        f"{snapshot.neck_x:.2f}",
        f"{snapshot.neck_y:.2f}",
        f"{snapshot.neck_conf:.3f}",
        neck.depth_x,
        neck.depth_y,
        fmt(neck.depth_m),
        f"{neck.valid_ratio:.3f}",
        f"{vector[0]:.6f}",
        f"{vector[1]:.6f}",
        f"{vector[2]:.6f}",
        f"{norm:.6f}",
        fmt(angle_deg),
        fmt(pitch_deg),
        1,
    ]


def start_depth_pipeline() -> tuple[Any, float, Any]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrinsics = depth_stream.get_intrinsics()
    print(
        f"D435 started: {args.depth_width}x{args.depth_height}@{args.depth_fps}, "
        f"scale={depth_scale}, fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}",
        flush=True,
    )
    return pipeline, depth_scale, depth_intrinsics


def main() -> None:
    logger = CsvLogger(args.log_csv)
    pipeline = None
    baseline_vectors: list[np.ndarray] = []
    baseline_vector: Optional[np.ndarray] = None
    baseline_started_at: Optional[float] = None
    frame_count = 0
    last_status = 0.0

    preview_config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )

    try:
        print("Loading IMX500 network firmware...")
        imx500.show_network_fw_progress_bar()
        picam2.start(preview_config, show_preview=not args.no_preview)
        imx500.set_auto_aspect_ratio()
        picam2.pre_callback = pre_callback

        pipeline, depth_scale, depth_intrinsics = start_depth_pipeline()
        print(
            f"Hold a correct posture for {args.baseline_seconds:.1f}s to record baseline 3D neck vector.",
            flush=True,
        )

        while True:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            with latest_lock:
                snapshot = latest_keypoints

            if snapshot is None:
                continue

            nose_depth = sample_depth(depth_m, snapshot.nose_x, snapshot.nose_y)
            left_depth = sample_depth(depth_m, snapshot.left_shoulder_x, snapshot.left_shoulder_y)
            right_depth = sample_depth(depth_m, snapshot.right_shoulder_x, snapshot.right_shoulder_y)
            neck_depth = sample_depth(depth_m, snapshot.neck_x, snapshot.neck_y)

            shoulder_depth_values = [
                value
                for value in (left_depth.depth_m, right_depth.depth_m, neck_depth.depth_m)
                if value is not None
            ]
            if nose_depth.depth_m is None or not shoulder_depth_values:
                continue
            neck_depth_m = float(np.median(shoulder_depth_values))
            neck_depth = DepthPoint(neck_depth.depth_x, neck_depth.depth_y, neck_depth_m, neck_depth.valid_ratio)

            nose_3d = pixel_to_camera_m(depth_intrinsics, nose_depth.depth_x, nose_depth.depth_y, nose_depth.depth_m)
            neck_3d = pixel_to_camera_m(depth_intrinsics, neck_depth.depth_x, neck_depth.depth_y, neck_depth.depth_m)
            vector = nose_3d - neck_3d

            now = time.time()
            if baseline_started_at is None:
                baseline_started_at = now
            if baseline_vector is None:
                baseline_vectors.append(vector)
                baseline_elapsed = now - baseline_started_at
                if baseline_elapsed >= args.baseline_seconds:
                    baseline_vector = np.median(np.stack(baseline_vectors, axis=0), axis=0)
                    print(f"Baseline vector recorded: {baseline_vector}", flush=True)

            angle_deg = angle_between_3d(baseline_vector, vector) if baseline_vector is not None else None
            pitch_deg = pitch_from_vertical(vector)
            baseline_ready = baseline_vector is not None

            logger.write(
                build_row(
                    snapshot,
                    nose_depth,
                    left_depth,
                    right_depth,
                    neck_depth,
                    vector,
                    angle_deg,
                    pitch_deg,
                    baseline_ready,
                )
            )
            frame_count += 1

            metrics = {
                "baseline_ready": baseline_ready,
                "nose_depth_m": nose_depth.depth_m,
                "neck_depth_m": neck_depth.depth_m,
                "neck_angle_baseline_deg": angle_deg,
                "neck_pitch_deg": pitch_deg,
            }
            with latest_lock:
                global latest_depth_metrics
                latest_depth_metrics = metrics

            if now - last_status >= args.print_interval:
                if baseline_ready:
                    print(
                        f"angle_delta={fmt(angle_deg, 2)}deg pitch={fmt(pitch_deg, 2)}deg "
                        f"nose_z={nose_depth.depth_m:.3f}m neck_z={neck_depth.depth_m:.3f}m "
                        f"vec=({vector[0]:.3f},{vector[1]:.3f},{vector[2]:.3f})",
                        flush=True,
                    )
                else:
                    elapsed = 0.0 if baseline_started_at is None else now - baseline_started_at
                    print(
                        f"baseline {elapsed:.1f}/{args.baseline_seconds:.1f}s "
                        f"nose_z={nose_depth.depth_m:.3f}m neck_z={neck_depth.depth_m:.3f}m",
                        flush=True,
                    )
                last_status = now
    except KeyboardInterrupt:
        print("\nStopping keypoint depth angle test.", flush=True)
    finally:
        logger.close()
        picam2.pre_callback = None
        try:
            picam2.stop()
        except Exception as exc:
            print(f"IMX500 stop failed: {exc}", file=sys.stderr, flush=True)
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"D435 stop failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
