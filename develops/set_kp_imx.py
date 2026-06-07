#!/usr/bin/env python3
"""
IMX500 pose keypoint box extractor.

Development-only ROI/keypoint inspection utility.

Use this before depth integration to inspect the AI Camera coordinates for the
regions that will later be mapped to D435 depth coordinates:
- head box
- shoulder box
- chest box

The script draws ROI boxes on the AI Camera preview and optionally writes their
pixel coordinates to CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import CompletedRequest, MappedArray, Picamera2
    from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
    from picamera2.devices.imx500.postprocess import COCODrawer
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
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


@dataclass
class RoiBox:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


class CsvLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.file = None
        self.writer = None
        if path is not None:
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(["timestamp", "roi", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
            self.file.flush()

    def write(self, boxes: List[RoiBox]) -> None:
        if self.writer is None:
            return
        now = time.time()
        for box in boxes:
            self.writer.writerow(
                [now, box.name, box.x1, box.y1, box.x2, box.y2, box.cx, box.cy, f"{box.confidence:.3f}"]
            )

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract IMX500 pose ROI boxes")
    parser.add_argument("--model", default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.30)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.30)
    parser.add_argument("--box-padding", type=int, default=25)
    parser.add_argument("--chest-height", type=int, default=110)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--print-interval", type=float, default=1.0)
    return parser.parse_args()


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

drawer = COCODrawer(intrinsics.labels or ["person"], imx500, needs_rescale_coords=False)
picam2 = Picamera2(imx500.camera_num)


def clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def valid_point(keypoints: np.ndarray, index: int) -> bool:
    return float(keypoints[index][2]) >= args.min_keypoint_confidence


def make_box(name: str, points: List[np.ndarray], width: int, height: int, padding: int) -> Optional[RoiBox]:
    valid = [p for p in points if float(p[2]) >= args.min_keypoint_confidence]
    if not valid:
        return None
    xs = [float(p[0]) for p in valid]
    ys = [float(p[1]) for p in valid]
    conf = float(np.mean([float(p[2]) for p in valid]))
    return RoiBox(
        name=name,
        x1=clamp(min(xs) - padding, 0, width - 1),
        y1=clamp(min(ys) - padding, 0, height - 1),
        x2=clamp(max(xs) + padding, 0, width - 1),
        y2=clamp(max(ys) + padding, 0, height - 1),
        confidence=conf,
    )


def extract_roi_boxes(keypoints: np.ndarray, frame_shape: Tuple[int, int, int]) -> List[RoiBox]:
    height, width = frame_shape[:2]
    padding = args.box_padding
    boxes: List[RoiBox] = []

    head_points = [
        keypoints[index]
        for index in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
        if valid_point(keypoints, index)
    ]
    head_box = make_box("head", head_points, width, height, padding)
    if head_box is not None:
        boxes.append(head_box)

    if valid_point(keypoints, LEFT_SHOULDER) and valid_point(keypoints, RIGHT_SHOULDER):
        shoulder_l = keypoints[LEFT_SHOULDER]
        shoulder_r = keypoints[RIGHT_SHOULDER]
        shoulder_box = make_box("shoulder", [shoulder_l, shoulder_r], width, height, padding)
        if shoulder_box is not None:
            boxes.append(shoulder_box)

            chest_y1 = shoulder_box.y2
            chest_y2 = clamp(chest_y1 + args.chest_height, 0, height - 1)
            chest_box = RoiBox(
                name="chest",
                x1=shoulder_box.x1,
                y1=chest_y1,
                x2=shoulder_box.x2,
                y2=chest_y2,
                confidence=shoulder_box.confidence,
            )
            boxes.append(chest_box)

    return boxes


def select_person(keypoints: Optional[np.ndarray], scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None or len(keypoints) == 0:
        return None
    if scores is None or len(scores) == 0:
        return keypoints[0]
    return keypoints[int(np.argmax(scores))]


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
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
        else:
            last_keypoints = None
            last_boxes = None
            last_scores = None
    return last_boxes, last_scores, last_keypoints


def draw_roi_boxes(image: np.ndarray, roi_boxes: List[RoiBox]) -> None:
    colors = {
        "head": (0, 255, 255),
        "shoulder": (0, 255, 0),
        "chest": (255, 128, 0),
    }
    for box in roi_boxes:
        color = colors.get(box.name, (255, 255, 255))
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        cv2.circle(image, (box.cx, box.cy), 4, color, -1)
        cv2.putText(
            image,
            f"{box.name} ({box.cx},{box.cy})",
            (box.x1, max(18, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def print_boxes(roi_boxes: List[RoiBox]) -> None:
    if not roi_boxes:
        print("roi: none", flush=True)
        return
    parts = [
        f"{b.name}=({b.x1},{b.y1})-({b.x2},{b.y2}) center=({b.cx},{b.cy}) conf={b.confidence:.2f}"
        for b in roi_boxes
    ]
    print("roi: " + " | ".join(parts), flush=True)


def pre_callback(request: CompletedRequest) -> None:
    global last_print
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    person = select_person(keypoints, scores)

    with MappedArray(request, "main") as mapped:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(
                mapped.array,
                boxes,
                scores,
                np.zeros(scores.shape),
                keypoints,
                args.detection_threshold,
                args.min_keypoint_confidence,
                request.get_metadata(),
                picam2,
                "main",
            )

        roi_boxes = extract_roi_boxes(person, mapped.array.shape) if person is not None else []
        draw_roi_boxes(mapped.array, roi_boxes)
        logger.write(roi_boxes)

        now = time.time()
        if now - last_print >= args.print_interval:
            print_boxes(roi_boxes)
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

    print("Extracting AI Camera ROI boxes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ROI extractor.")
    finally:
        logger.close()
        picam2.stop()


if __name__ == "__main__":
    main()
