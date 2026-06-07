from __future__ import annotations

# Development-only coordinate accuracy logger.

import csv
from pathlib import Path
from typing import Any, Iterable, List, Optional

from posture.config import MonitorConfig
from posture.models import DepthMeasurement, PostureResult, RoiBox


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
ROI_NAMES = ["head", "shoulder", "chest"]
KEYPOINT_DEPTH_NAMES = ["nose", "neck", "left_shoulder_kp", "right_shoulder_kp"]


class CoordinateAccuracyLogger:
    """Wide CSV logger for posture accuracy analysis.

    The ordinary posture log stores classifier-oriented features. This logger
    stores raw pose/depth/angle evidence so bad posture labels can be audited.
    """

    def __init__(self, path: Optional[Path], config: MonitorConfig) -> None:
        self.path = path
        self.config = config
        self.file = None
        self.writer = None
        self.rows_written = 0
        if path is not None:
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(self._header())
            self.file.flush()

    def write(
        self,
        result: PostureResult,
        keypoints: Optional[Any],
        roi_boxes: Iterable[RoiBox],
        coord_depth_measurements: Iterable[DepthMeasurement],
    ) -> None:
        if self.writer is None:
            return

        features = result.features
        roi_by_name = {box.name: box for box in roi_boxes}
        result_depth_by_name = {
            measurement.roi: measurement
            for measurement in (result.depth_measurements or [])
        }
        coord_depth_by_name = {
            measurement.roi: measurement
            for measurement in coord_depth_measurements
        }
        depth_by_name = {**result_depth_by_name, **coord_depth_by_name}
        deltas = result.depth_deltas or {}

        row: List[object] = [
            f"{result.timestamp:.6f}",
            result.state,
            result.reason,
            int(result.alarm),
            f"{result.elapsed_bad_seconds:.3f}",
            "" if "head" not in deltas else f"{deltas['head']:.4f}",
            "" if "shoulder" not in deltas else f"{deltas['shoulder']:.4f}",
            "" if "chest" not in deltas else f"{deltas['chest']:.4f}",
            "" if result.nose_depth_delta_m is None else f"{result.nose_depth_delta_m:.4f}",
            "" if result.neck_angle_delta_deg is None else f"{result.neck_angle_delta_deg:.3f}",
            "" if result.neck_pitch_deg is None else f"{result.neck_pitch_deg:.3f}",
            "" if result.neck_vector_x_m is None else f"{result.neck_vector_x_m:.6f}",
            "" if result.neck_vector_y_m is None else f"{result.neck_vector_y_m:.6f}",
            "" if result.neck_vector_z_m is None else f"{result.neck_vector_z_m:.6f}",
            "" if features is None else f"{features.head_x:.2f}",
            "" if features is None else f"{features.shoulder_x:.2f}",
            "" if features is None else f"{features.shoulder_tilt:.2f}",
            "" if features is None else f"{features.shoulder_angle_deg:.3f}",
            "" if result.pose_shoulder_tilt_delta_px is None else f"{result.pose_shoulder_tilt_delta_px:.2f}",
            "" if result.pose_shoulder_angle_delta_deg is None else f"{result.pose_shoulder_angle_delta_deg:.3f}",
            "" if features is None else f"{features.min_confidence:.3f}",
            *self._neck_point_columns(keypoints),
        ]

        for name in COCO_KEYPOINT_NAMES:
            row.extend(self._keypoint_columns(keypoints, name))

        for name in ROI_NAMES:
            row.extend(self._roi_columns(roi_by_name.get(name), depth_by_name.get(name)))

        for name in KEYPOINT_DEPTH_NAMES:
            row.extend(self._depth_columns(depth_by_name.get(name)))

        self.writer.writerow(row)
        self.rows_written += 1

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            print(f"Coordinate log saved to {self.path} ({self.rows_written} rows)", flush=True)
            self.file = None
            self.writer = None

    def _header(self) -> List[str]:
        header = [
            "timestamp",
            "state",
            "reason",
            "alarm",
            "elapsed_bad_seconds",
            "head_delta_m",
            "shoulder_delta_m",
            "chest_delta_m",
            "nose_delta_m",
            "neck_angle_delta_deg",
            "neck_pitch_deg",
            "neck_vector_x_m",
            "neck_vector_y_m",
            "neck_vector_z_m",
            "feature_head_x",
            "feature_shoulder_x",
            "feature_shoulder_tilt_px",
            "feature_shoulder_angle_deg",
            "feature_shoulder_tilt_delta_px",
            "feature_shoulder_angle_delta_deg",
            "feature_min_confidence",
            "virtual_neck_x",
            "virtual_neck_y",
            "virtual_neck_confidence",
        ]
        for name in COCO_KEYPOINT_NAMES:
            header.extend([f"kp_{name}_x", f"kp_{name}_y", f"kp_{name}_confidence"])
        for name in ROI_NAMES:
            header.extend(
                [
                    f"roi_{name}_x1",
                    f"roi_{name}_y1",
                    f"roi_{name}_x2",
                    f"roi_{name}_y2",
                    f"roi_{name}_ai_cx",
                    f"roi_{name}_ai_cy",
                    f"roi_{name}_confidence",
                    f"roi_{name}_depth_x",
                    f"roi_{name}_depth_y",
                    f"roi_{name}_depth_m",
                    f"roi_{name}_center_depth_m",
                    f"roi_{name}_valid_ratio",
                    f"roi_{name}_sample_size",
                ]
            )
        for name in KEYPOINT_DEPTH_NAMES:
            header.extend(
                [
                    f"{name}_depth_x",
                    f"{name}_depth_y",
                    f"{name}_depth_m",
                    f"{name}_center_depth_m",
                    f"{name}_valid_ratio",
                    f"{name}_sample_size",
                    f"{name}_confidence",
                ]
            )
        return header

    @staticmethod
    def _keypoint_columns(keypoints: Optional[Any], name: str) -> List[str]:
        index = COCO_KEYPOINT_NAMES.index(name)
        if keypoints is None or len(keypoints) <= index:
            return ["", "", ""]
        point = keypoints[index]
        return [f"{float(point[0]):.2f}", f"{float(point[1]):.2f}", f"{float(point[2]):.3f}"]

    @staticmethod
    def _neck_point_columns(keypoints: Optional[Any]) -> List[str]:
        if keypoints is None or len(keypoints) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
            return ["", "", ""]
        left = keypoints[LEFT_SHOULDER]
        right = keypoints[RIGHT_SHOULDER]
        neck_x = (float(left[0]) + float(right[0])) / 2.0
        neck_y = (float(left[1]) + float(right[1])) / 2.0
        confidence = min(float(left[2]), float(right[2]))
        return [f"{neck_x:.2f}", f"{neck_y:.2f}", f"{confidence:.3f}"]

    @staticmethod
    def _roi_columns(roi: Optional[RoiBox], measurement: Optional[DepthMeasurement]) -> List[str]:
        if roi is None:
            roi_columns = ["", "", "", "", "", "", ""]
        else:
            roi_columns = [
                str(roi.x1),
                str(roi.y1),
                str(roi.x2),
                str(roi.y2),
                str(roi.cx),
                str(roi.cy),
                f"{roi.confidence:.3f}",
            ]
        return roi_columns + CoordinateAccuracyLogger._depth_columns(measurement)[:6]

    @staticmethod
    def _depth_columns(measurement: Optional[DepthMeasurement]) -> List[str]:
        if measurement is None:
            return ["", "", "", "", "", "", ""]
        return [
            str(measurement.cx),
            str(measurement.cy),
            "" if measurement.median_m is None else f"{measurement.median_m:.4f}",
            "" if measurement.center_m is None else f"{measurement.center_m:.4f}",
            f"{measurement.valid_ratio:.3f}",
            str(measurement.sample_size),
            f"{measurement.confidence:.3f}",
        ]
