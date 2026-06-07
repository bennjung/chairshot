from __future__ import annotations

import sys
import threading
import time

import numpy as np

from .analyzer import PostureAnalyzer
from .config import MonitorConfig
from .depth import measure_depth, measure_keypoint_depth
from .depth_camera import D435DepthCamera, report_missing_realsense
from .models import DepthMeasurement
from .pose import LEFT_SHOULDER, NOSE, RIGHT_SHOULDER, extract_roi_boxes, select_person
from .shared import SharedState
from .turtle_neck_filter import KeypointDepth3D, build_neck_vector

REQUIRED_ROI_NAMES = {"head", "shoulder", "chest"}


class PoseWorker(threading.Thread):
    def __init__(
        self,
        config: MonitorConfig,
        analyzer: PostureAnalyzer,
        shared: SharedState,
        stop_event: threading.Event,
        depth_active: bool,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.analyzer = analyzer
        self.shared = shared
        self.stop_event = stop_event
        self.depth_active = depth_active
        self.last_packet_timestamp = 0.0
        self.last_valid_roi_boxes = []
        self.last_valid_features = None
        self.last_valid_keypoints = None
        self.last_valid_roi_at = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            with self.shared.lock:
                packet = self.shared.latest_pose_packet

            if packet is None or packet.timestamp <= self.last_packet_timestamp:
                time.sleep(0.01)
                continue

            self.last_packet_timestamp = packet.timestamp
            person = select_person(packet.keypoints, packet.scores)
            if person is None:
                held_boxes, held_features, held_keypoints = self._held_roi_state()
                with self.shared.lock:
                    self.shared.latest_roi_boxes = held_boxes
                    self.shared.latest_feature_frame = held_features
                    self.shared.latest_selected_keypoints = held_keypoints
                if not self.depth_active:
                    with self.analyzer.lock:
                        result = self.analyzer.analyze(None)
                    with self.shared.lock:
                        self.shared.latest_result = result
                continue

            with self.analyzer.lock:
                if self.analyzer.baseline is None:
                    self.analyzer.add_baseline_frame(person)
                features = self.analyzer.extract_features(person)
                result = None if self.depth_active else self.analyzer.analyze(person)

            roi_boxes = []
            if features is not None:
                roi_boxes = extract_roi_boxes(person, (self.config.ai_height, self.config.ai_width, 3), self.config)

            if self._roi_complete(roi_boxes):
                self.last_valid_roi_boxes = roi_boxes
                self.last_valid_features = features
                self.last_valid_keypoints = person
                self.last_valid_roi_at = time.time()
                published_boxes = roi_boxes
                published_features = features
                published_keypoints = person
            else:
                published_boxes, published_features, published_keypoints = self._held_roi_state(current_keypoints=person)

            with self.shared.lock:
                self.shared.latest_roi_boxes = published_boxes
                self.shared.latest_feature_frame = published_features
                self.shared.latest_selected_keypoints = published_keypoints
                if result is not None:
                    self.shared.latest_result = result

    @staticmethod
    def _roi_complete(roi_boxes) -> bool:
        return REQUIRED_ROI_NAMES.issubset({box.name for box in roi_boxes})

    def _held_roi_state(self, current_keypoints=None):
        if (
            self.depth_active
            and self.last_valid_roi_boxes
            and time.time() - self.last_valid_roi_at <= self.config.out_of_range_tolerance_seconds
        ):
            keypoints = current_keypoints if current_keypoints is not None else self.last_valid_keypoints
            return list(self.last_valid_roi_boxes), self.last_valid_features, keypoints
        return [], None, current_keypoints


class DepthWorker(threading.Thread):
    def __init__(
        self,
        config: MonitorConfig,
        analyzer: PostureAnalyzer,
        shared: SharedState,
        stop_event: threading.Event,
        coord_logging_enabled: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.analyzer = analyzer
        self.shared = shared
        self.stop_event = stop_event
        self.coord_logging_enabled = coord_logging_enabled
        self.camera = D435DepthCamera(config)

    def run(self) -> None:
        if not self.camera.available:
            report_missing_realsense()
            return

        try:
            self.camera.start()

            while not self.stop_event.is_set():
                with self.shared.lock:
                    roi_boxes = list(self.shared.latest_roi_boxes)
                    features = self.shared.latest_feature_frame
                    keypoints = self.shared.latest_selected_keypoints

                if not roi_boxes:
                    if self.coord_logging_enabled:
                        with self.shared.lock:
                            self.shared.latest_coord_depth_measurements = []
                    with self.analyzer.lock:
                        if self.analyzer.depth_baseline is not None:
                            result = self.analyzer.handle_out_of_range(time.time(), "person_missing_or_range_left")
                            with self.shared.lock:
                                self.shared.latest_result = result
                    time.sleep(0.05)
                    continue

                try:
                    depth_m = self.camera.wait_depth_m(timeout_ms=500)
                except Exception as exc:
                    print(f"Waiting for D435 depth frame failed: {exc}", file=sys.stderr, flush=True)
                    time.sleep(0.1)
                    continue

                if depth_m is None:
                    continue

                measurements = [measure_depth(depth_m, roi, self.config) for roi in roi_boxes]
                turtle_neck_result, coord_measurements = self._measure_turtle_neck_depth(depth_m, keypoints, measurements)
                with self.shared.lock:
                    self.shared.latest_depth_measurements = measurements
                    if self.coord_logging_enabled:
                        self.shared.latest_coord_depth_measurements = coord_measurements

                with self.analyzer.lock:
                    if self.analyzer.depth_baseline is None:
                        self.analyzer.add_depth_baseline_frame(measurements, turtle_neck_result)
                        continue
                    result = self.analyzer.analyze_depth(measurements, features, turtle_neck_result)

                with self.shared.lock:
                    self.shared.latest_result = result
        except Exception as exc:
            print(f"D435 depth worker stopped: {exc}", file=sys.stderr, flush=True)
        finally:
            self.camera.stop()

    def _measure_turtle_neck_depth(self, depth_m, keypoints, measurements):
        if keypoints is None or self.camera.depth_intrinsics is None:
            return None, []
        required = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)
        if any(float(keypoints[index][2]) < self.config.min_keypoint_confidence for index in required):
            return None, []

        nose = keypoints[NOSE]
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        neck_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
        neck_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
        neck_confidence = float(min(left_shoulder[2], right_shoulder[2]))

        nose_measurement = measure_keypoint_depth(
            depth_m,
            "nose",
            float(nose[0]),
            float(nose[1]),
            float(nose[2]),
            self.config,
        )
        neck_measurement = measure_keypoint_depth(depth_m, "neck", neck_x, neck_y, neck_confidence, self.config)
        left_measurement = measure_keypoint_depth(
            depth_m,
            "left_shoulder_kp",
            float(left_shoulder[0]),
            float(left_shoulder[1]),
            float(left_shoulder[2]),
            self.config,
        )
        right_measurement = measure_keypoint_depth(
            depth_m,
            "right_shoulder_kp",
            float(right_shoulder[0]),
            float(right_shoulder[1]),
            float(right_shoulder[2]),
            self.config,
        )
        shoulder_depth_values = [
            measurement.median_m
            for measurement in (left_measurement, right_measurement, neck_measurement)
            if measurement.median_m is not None
        ]
        if shoulder_depth_values:
            neck_measurement = DepthMeasurement(
                timestamp=neck_measurement.timestamp,
                roi_timestamp=neck_measurement.roi_timestamp,
                roi=neck_measurement.roi,
                cx=neck_measurement.cx,
                cy=neck_measurement.cy,
                sample_size=neck_measurement.sample_size,
                confidence=neck_measurement.confidence,
                valid_ratio=neck_measurement.valid_ratio,
                median_m=float(np.median(shoulder_depth_values)),
                center_m=neck_measurement.center_m,
            )

        measurements.extend([nose_measurement, neck_measurement])
        coord_measurements = [nose_measurement, neck_measurement, left_measurement, right_measurement]

        intrinsics = self.camera.depth_intrinsics
        neck_result = build_neck_vector(
            KeypointDepth3D(
                x_px=float(nose[0]),
                y_px=float(nose[1]),
                depth_x=nose_measurement.cx,
                depth_y=nose_measurement.cy,
                depth_m=nose_measurement.median_m,
                valid_ratio=nose_measurement.valid_ratio,
                confidence=float(nose[2]),
            ),
            KeypointDepth3D(
                x_px=neck_x,
                y_px=neck_y,
                depth_x=neck_measurement.cx,
                depth_y=neck_measurement.cy,
                depth_m=neck_measurement.median_m,
                valid_ratio=neck_measurement.valid_ratio,
                confidence=neck_confidence,
            ),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.ppx),
            cy=float(intrinsics.ppy),
            nose_valid_ratio_threshold=self.config.nose_depth_valid_ratio_threshold,
            neck_valid_ratio_threshold=self.config.neck_depth_valid_ratio_threshold,
        )
        return neck_result, coord_measurements
