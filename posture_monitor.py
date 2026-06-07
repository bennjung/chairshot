#!/usr/bin/env python3
"""Raspberry Pi AI Camera + D435 posture monitor orchestrator."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from posture.alerts import GpioAlert
from posture.analyzer import PostureAnalyzer
from posture.config import MonitorConfig
from posture.event_publisher import HttpPostureEventPublisher
from posture.imx_camera import Imx500Camera
from posture.logging import CsvLogger
from posture.models import PostureResult
from posture.shared import SharedState
from posture.state import canonical_posture_state
from posture.workers import DepthWorker, PoseWorker

COORD_LOGGING_ENABLED = False


class StablePostureEventGate:
    def __init__(self, stable_seconds: float) -> None:
        self.stable_seconds = stable_seconds
        self.current_state = ""
        self.state_started_at = 0.0

    def should_publish(self, result: PostureResult) -> bool:
        state = canonical_posture_state(result)
        timestamp = result.timestamp
        if state != self.current_state:
            self.current_state = state
            self.state_started_at = timestamp
            return False
        return timestamp - self.state_started_at >= self.stable_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi AI Camera + D435 posture monitor")
    parser.add_argument("--model", default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--config", type=Path, default=Path("posture_config.json"))
    parser.add_argument("--labels")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection-threshold", type=float, default=0.30)
    parser.add_argument("--log-csv", type=Path)
    parser.add_argument(
        "--coord-csv",
        "--coord.csv",
        dest="coord_csv",
        nargs="?",
        const=Path("coord.csv"),
        type=Path,
        help="Write raw keypoint/depth/angle evidence for accuracy analysis",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-overlay", action="store_true", help="Keep preview but skip skeleton/ROI/depth drawing")
    parser.add_argument("--disable-depth", action="store_true")
    parser.add_argument("--print-intrinsics", action="store_true")
    parser.add_argument("--event-url", help="HTTP endpoint for posture events, e.g. http://127.0.0.1:8800/api/posture-events/")
    parser.add_argument("--event-timeout", type=float, default=1.0)
    parser.add_argument("--event-queue-size", type=int, default=300)
    return parser.parse_args()


class PostureMonitorApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = MonitorConfig.from_file(args.config)
        self.analyzer = PostureAnalyzer(self.config)
        self.shared = SharedState()
        self.stop_event = threading.Event()
        self.alerts = GpioAlert(
            self.config.green_led_pin,
            self.config.yellow_led_pin,
            self.config.red_led_pin,
            self.config.buzzer_pin,
            self.config.led_switch_delay_seconds,
        )
        self.logger = CsvLogger(args.log_csv)
        self.event_publisher = (
            HttpPostureEventPublisher(args.event_url, args.event_timeout, args.event_queue_size)
            if args.event_url
            else None
        )
        self.event_gate = StablePostureEventGate(self.config.event_publish_stable_seconds)
        self.coord_logger = None
        if args.coord_csv is not None and COORD_LOGGING_ENABLED:
            from develops.accur import CoordinateAccuracyLogger

            self.coord_logger = CoordinateAccuracyLogger(args.coord_csv, self.config)
        elif args.coord_csv is not None:
            print("Coordinate accuracy logging is temporarily disabled.", flush=True)
        self.last_alarm_state = False
        self.last_out_of_range_state = False
        self.last_status_print = 0.0
        self.last_processed_result_at = 0.0
        self.imx_camera = Imx500Camera(
            model_path=args.model,
            labels_path=args.labels,
            fps=args.fps,
            detection_threshold=args.detection_threshold,
            no_preview=args.no_preview,
            no_overlay=args.no_overlay,
            config=self.config,
            shared=self.shared,
        )
        self.depth_active = self.config.depth_enabled and not args.disable_depth
        self.pose_worker = PoseWorker(self.config, self.analyzer, self.shared, self.stop_event, self.depth_active)
        self.depth_worker = (
            DepthWorker(self.config, self.analyzer, self.shared, self.stop_event, coord_logging_enabled=COORD_LOGGING_ENABLED)
            if self.depth_active
            else None
        )

    def process_posture_result(self, result: PostureResult) -> None:
        self.alerts.set_state(result.state, result.alarm)
        self.logger.write(result)
        if self.event_publisher is not None and self.event_gate.should_publish(result):
            self.event_publisher.publish(result)
        if self.coord_logger is not None:
            with self.shared.lock:
                keypoints = self.shared.latest_selected_keypoints
                roi_boxes = list(self.shared.latest_roi_boxes)
                coord_measurements = list(self.shared.latest_coord_depth_measurements)
            self.coord_logger.write(result, keypoints, roi_boxes, coord_measurements)

        if result.alarm and not self.last_alarm_state:
            print(f"ALARM: bad posture persisted for {result.elapsed_bad_seconds:.1f}s ({result.reason})", flush=True)
        elif not result.alarm and self.last_alarm_state:
            print("ALARM CLEARED: posture returned to normal or became untrackable", flush=True)
        self.last_alarm_state = result.alarm

        is_out_of_range = result.state == "out_of_range"
        if is_out_of_range and not self.last_out_of_range_state:
            print(f"OUT_OF_RANGE: pose/depth target left valid area for {self.config.out_of_range_tolerance_seconds:.1f}s", flush=True)
        elif not is_out_of_range and self.last_out_of_range_state:
            print("OUT_OF_RANGE CLEARED: valid pose detected again", flush=True)
        self.last_out_of_range_state = is_out_of_range

        now = time.time()
        if now - self.last_status_print >= 1.0:
            deltas = result.depth_deltas or {}
            depth_text = ""
            if deltas:
                depth_text = (
                    f" head_d={deltas.get('head', 0.0):.3f}m"
                    f" shoulder_d={deltas.get('shoulder', 0.0):.3f}m"
                )
            if result.neck_angle_delta_deg is not None:
                depth_text += f" neck_angle={result.neck_angle_delta_deg:.1f}deg"
            if result.nose_depth_m is not None:
                depth_text += f" nose_z={result.nose_depth_m:.3f}m"
            if result.nose_depth_delta_m is not None:
                depth_text += f" nose_d={result.nose_depth_delta_m:.3f}m"
            print(
                f"state={result.state} reason={result.reason} "
                f"bad_for={result.elapsed_bad_seconds:.1f}s alarm={result.alarm}{depth_text}",
                flush=True,
            )
            self.last_status_print = now

    def wait_for_baseline(self) -> None:
        target_text = "valid pose/depth frames" if self.depth_active else "valid pose frames"
        print(f"Hold a correct sitting posture until {self.config.baseline_seconds:.1f}s of {target_text} are recorded.")
        self.alerts.set_baseline()
        with self.shared.lock:
            self.shared.baseline_active = True
            self.shared.baseline_depth_active = self.depth_active
            self.shared.baseline_pose_seconds = 0.0
            self.shared.baseline_depth_seconds = 0.0
            self.shared.baseline_target_seconds = self.config.baseline_seconds

        baseline_valid_started = None
        baseline_deadline = time.time() + max(30.0, self.config.baseline_seconds * 6.0)
        last_baseline_print = 0.0

        while time.time() < baseline_deadline:
            with self.analyzer.lock:
                valid_seconds = 0.0
                if self.analyzer.baseline_frames:
                    if baseline_valid_started is None:
                        baseline_valid_started = self.analyzer.baseline_frames[0].timestamp
                    valid_seconds = self.analyzer.baseline_frames[-1].timestamp - baseline_valid_started
                pose_frame_count = len(self.analyzer.baseline_frames)
                depth_valid_seconds = self.analyzer.depth_baseline_valid_seconds()
                depth_frame_count = min((len(values) for values in self.analyzer.depth_baseline_frames.values()), default=0)

            with self.shared.lock:
                self.shared.baseline_pose_seconds = valid_seconds
                self.shared.baseline_depth_seconds = depth_valid_seconds
                self.shared.baseline_target_seconds = self.config.baseline_seconds

            now = time.time()
            if now - last_baseline_print >= 1.0:
                if self.depth_active:
                    print(
                        f"Baseline pose={valid_seconds:.1f}/{self.config.baseline_seconds:.1f}s "
                        f"depth={depth_valid_seconds:.1f}/{self.config.baseline_seconds:.1f}s "
                        f"depth_frames={depth_frame_count}",
                        flush=True,
                    )
                else:
                    print(
                        f"Baseline valid pose: {valid_seconds:.1f}/{self.config.baseline_seconds:.1f}s "
                        f"frames={pose_frame_count}",
                        flush=True,
                    )
                last_baseline_print = now

            if valid_seconds >= self.config.baseline_seconds and (
                not self.depth_active or depth_valid_seconds >= self.config.baseline_seconds
            ):
                break
            time.sleep(0.1)

        with self.analyzer.lock:
            pose_baseline_ok = self.analyzer.finalize_baseline()
            depth_baseline_ok = True if not self.depth_active else self.analyzer.finalize_depth_baseline()

        if not pose_baseline_ok:
            raise SystemExit("Could not record baseline. Check camera angle and keypoint confidence.")
        if not depth_baseline_ok:
            raise SystemExit("Could not record depth baseline. Check D435 stream and ROI/depth overlap.")
        if self.depth_active:
            print(f"Depth baseline recorded: {self.analyzer.depth_baseline}", flush=True)
        with self.shared.lock:
            self.shared.baseline_pose_seconds = self.config.baseline_seconds
            self.shared.baseline_depth_seconds = self.config.baseline_seconds
            self.shared.baseline_active = False

    def run(self) -> None:
        if self.args.print_intrinsics:
            self.imx_camera.print_intrinsics()
            return
        self.imx_camera.start()

        self.pose_worker.start()
        if self.depth_worker is not None:
            self.depth_worker.start()

        try:
            self.wait_for_baseline()
            if self.event_publisher is not None:
                self.event_publisher.start()
            print("Baseline recorded. Monitoring posture...")

            while True:
                with self.shared.lock:
                    result = self.shared.latest_result
                if result is not None and result.timestamp > self.last_processed_result_at:
                    self.process_posture_result(result)
                    self.last_processed_result_at = result.timestamp
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping posture monitor.")
        finally:
            self.stop_event.set()
            self.pose_worker.join(timeout=2.0)
            if self.depth_worker is not None:
                self.depth_worker.join(timeout=2.0)
            self.alerts.close()
            self.logger.close()
            if self.event_publisher is not None:
                self.event_publisher.close()
            if self.coord_logger is not None:
                self.coord_logger.close()
            self.imx_camera.stop()


def main() -> None:
    PostureMonitorApp(parse_args()).run()


if __name__ == "__main__":
    main()
