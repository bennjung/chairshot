from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from .models import DepthMeasurement, PostureResult


class CsvLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self.file = None
        self.writer = None
        self.rows_written = 0
        if path is not None:
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow(
                [
                    "timestamp",
                    "state",
                    "reason",
                    "alarm",
                    "elapsed_bad_seconds",
                    "head_x",
                    "shoulder_x",
                    "shoulder_tilt",
                    "shoulder_angle_deg",
                    "min_confidence",
                    "head_m",
                    "shoulder_m",
                    "chest_m",
                    "head_delta_m",
                    "shoulder_delta_m",
                    "chest_delta_m",
                    "nose_m",
                    "neck_m",
                    "nose_delta_m",
                    "pose_shoulder_tilt_delta_px",
                    "pose_shoulder_angle_delta_deg",
                    "nose_valid_ratio",
                    "neck_valid_ratio",
                    "neck_angle_delta_deg",
                    "neck_pitch_deg",
                    "neck_vector_x_m",
                    "neck_vector_y_m",
                    "neck_vector_z_m",
                ]
            )
            self.file.flush()

    def write(self, result: PostureResult) -> None:
        if self.writer is None:
            return
        features = result.features
        depth_by_roi = {
            measurement.roi: measurement
            for measurement in (result.depth_measurements or [])
        }
        deltas = result.depth_deltas or {}
        self.writer.writerow(
            [
                result.timestamp,
                result.state,
                result.reason,
                int(result.alarm),
                f"{result.elapsed_bad_seconds:.3f}",
                "" if features is None else f"{features.head_x:.2f}",
                "" if features is None else f"{features.shoulder_x:.2f}",
                "" if features is None else f"{features.shoulder_tilt:.2f}",
                "" if features is None else f"{features.shoulder_angle_deg:.2f}",
                "" if features is None else f"{features.min_confidence:.3f}",
                self._format_depth(depth_by_roi.get("head")),
                self._format_depth(depth_by_roi.get("shoulder")),
                self._format_depth(depth_by_roi.get("chest")),
                "" if "head" not in deltas else f"{deltas['head']:.4f}",
                "" if "shoulder" not in deltas else f"{deltas['shoulder']:.4f}",
                "" if "chest" not in deltas else f"{deltas['chest']:.4f}",
                self._format_depth(depth_by_roi.get("nose")),
                self._format_depth(depth_by_roi.get("neck")),
                "" if result.nose_depth_delta_m is None else f"{result.nose_depth_delta_m:.4f}",
                "" if result.pose_shoulder_tilt_delta_px is None else f"{result.pose_shoulder_tilt_delta_px:.2f}",
                "" if result.pose_shoulder_angle_delta_deg is None else f"{result.pose_shoulder_angle_delta_deg:.2f}",
                "" if result.nose_valid_ratio is None else f"{result.nose_valid_ratio:.3f}",
                "" if result.neck_valid_ratio is None else f"{result.neck_valid_ratio:.3f}",
                "" if result.neck_angle_delta_deg is None else f"{result.neck_angle_delta_deg:.3f}",
                "" if result.neck_pitch_deg is None else f"{result.neck_pitch_deg:.3f}",
                "" if result.neck_vector_x_m is None else f"{result.neck_vector_x_m:.6f}",
                "" if result.neck_vector_y_m is None else f"{result.neck_vector_y_m:.6f}",
                "" if result.neck_vector_z_m is None else f"{result.neck_vector_z_m:.6f}",
            ]
        )
        self.rows_written += 1

    @staticmethod
    def _format_depth(measurement: Optional[DepthMeasurement]) -> str:
        if measurement is None or measurement.median_m is None:
            return ""
        return f"{measurement.median_m:.4f}"

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            print(f"Log saved to {self.path} ({self.rows_written} rows)", flush=True)
            self.file = None
            self.writer = None
