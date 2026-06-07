#!/usr/bin/env python3
"""Development-only virtual neck preview for IMX500 nose/shoulder keypoints."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
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


MODEL_INPUT_H_W = (480, 640)

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw virtual neck skeleton from IMX500 pose keypoints")
    parser.add_argument("--model", default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.30)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-interval", type=float, default=0.5)
    return parser.parse_args()


class CsvLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.file = None
        self.writer = None
        if path is not None:
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(
                [
                    "timestamp",
                    "nose_x",
                    "nose_y",
                    "nose_conf",
                    "neck_x",
                    "neck_y",
                    "neck_conf",
                    "left_shoulder_x",
                    "left_shoulder_y",
                    "left_shoulder_conf",
                    "right_shoulder_x",
                    "right_shoulder_y",
                    "right_shoulder_conf",
                    "nose_neck_dx",
                    "nose_neck_dy",
                    "nose_neck_angle_deg",
                    "valid",
                ]
            )
            self.file.flush()

    def write(self, metrics: Optional[Dict[str, float]]) -> None:
        if self.writer is None:
            return
        if metrics is None:
            self.writer.writerow([time.time()] + [""] * 15 + [0])
            return
        self.writer.writerow(
            [
                time.time(),
                f"{metrics['nose_x']:.2f}",
                f"{metrics['nose_y']:.2f}",
                f"{metrics['nose_conf']:.3f}",
                f"{metrics['neck_x']:.2f}",
                f"{metrics['neck_y']:.2f}",
                f"{metrics['neck_conf']:.3f}",
                f"{metrics['left_shoulder_x']:.2f}",
                f"{metrics['left_shoulder_y']:.2f}",
                f"{metrics['left_shoulder_conf']:.3f}",
                f"{metrics['right_shoulder_x']:.2f}",
                f"{metrics['right_shoulder_y']:.2f}",
                f"{metrics['right_shoulder_conf']:.3f}",
                f"{metrics['nose_neck_dx']:.2f}",
                f"{metrics['nose_neck_dy']:.2f}",
                f"{metrics['nose_neck_angle_deg']:.2f}",
                1,
            ]
        )

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None


args = parse_args()
logger = CsvLogger(args.log_csv)

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


def virtual_neck_metrics(person: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if person is None:
        return None
    if not all(valid_point(person, index) for index in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)):
        return None

    nose = person[NOSE]
    left_shoulder = person[LEFT_SHOULDER]
    right_shoulder = person[RIGHT_SHOULDER]
    neck_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
    neck_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
    neck_conf = float(min(left_shoulder[2], right_shoulder[2]))
    dx = float(nose[0] - neck_x)
    dy = float(nose[1] - neck_y)
    angle = math.degrees(math.atan2(dx, -dy)) if dx or dy else 0.0

    return {
        "nose_x": float(nose[0]),
        "nose_y": float(nose[1]),
        "nose_conf": float(nose[2]),
        "neck_x": neck_x,
        "neck_y": neck_y,
        "neck_conf": neck_conf,
        "left_shoulder_x": float(left_shoulder[0]),
        "left_shoulder_y": float(left_shoulder[1]),
        "left_shoulder_conf": float(left_shoulder[2]),
        "right_shoulder_x": float(right_shoulder[0]),
        "right_shoulder_y": float(right_shoulder[1]),
        "right_shoulder_conf": float(right_shoulder[2]),
        "nose_neck_dx": dx,
        "nose_neck_dy": dy,
        "nose_neck_angle_deg": angle,
    }


def draw_virtual_neck(image: np.ndarray, metrics: Optional[Dict[str, float]]) -> None:
    if metrics is None:
        cv2.putText(
            image,
            "virtual neck: invalid keypoints",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    nose = (round(metrics["nose_x"]), round(metrics["nose_y"]))
    neck = (round(metrics["neck_x"]), round(metrics["neck_y"]))
    left_shoulder = (round(metrics["left_shoulder_x"]), round(metrics["left_shoulder_y"]))
    right_shoulder = (round(metrics["right_shoulder_x"]), round(metrics["right_shoulder_y"]))

    cv2.line(image, left_shoulder, right_shoulder, (0, 255, 0), 3)
    cv2.line(image, neck, nose, (0, 255, 255), 3)
    cv2.circle(image, nose, 8, (0, 255, 255), -1)
    cv2.circle(image, neck, 8, (255, 0, 255), -1)
    cv2.circle(image, left_shoulder, 7, (0, 255, 0), -1)
    cv2.circle(image, right_shoulder, 7, (0, 255, 0), -1)

    cv2.putText(image, "nose", (nose[0] + 10, nose[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "virtual neck", (neck[0] + 10, neck[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"dx={metrics['nose_neck_dx']:.1f}px dy={metrics['nose_neck_dy']:.1f}px angle={metrics['nose_neck_angle_deg']:.1f}deg",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def pre_callback(request: CompletedRequest) -> None:
    global last_print
    _, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    person = select_person(keypoints, scores)
    metrics = virtual_neck_metrics(person)

    with MappedArray(request, "main") as mapped:
        draw_virtual_neck(mapped.array, metrics)

    logger.write(metrics)
    now = time.time()
    if now - last_print >= args.print_interval:
        if metrics is None:
            print("virtual_neck: invalid", flush=True)
        else:
            print(
                "virtual_neck: "
                f"nose=({metrics['nose_x']:.1f},{metrics['nose_y']:.1f}) "
                f"neck=({metrics['neck_x']:.1f},{metrics['neck_y']:.1f}) "
                f"dx={metrics['nose_neck_dx']:.1f}px "
                f"dy={metrics['nose_neck_dy']:.1f}px "
                f"angle={metrics['nose_neck_angle_deg']:.1f}deg "
                f"conf={metrics['neck_conf']:.2f}",
                flush=True,
            )
        last_print = now


def main() -> None:
    preview_config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )

    print("Loading IMX500 network firmware...")
    imx500.show_network_fw_progress_bar()
    picam2.start(preview_config, show_preview=not args.no_preview)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = pre_callback

    print("Drawing virtual neck skeleton. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping virtual neck test.")
    finally:
        logger.close()
        picam2.pre_callback = None
        picam2.stop()


if __name__ == "__main__":
    main()
