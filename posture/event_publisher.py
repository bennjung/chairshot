from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse

from .models import DepthMeasurement, PostureResult
from .state import canonical_posture_state


class HttpPostureEventPublisher:
    def __init__(
        self,
        event_url: str,
        timeout_seconds: float = 1.0,
        queue_size: int = 300,
    ) -> None:
        self.event_url = event_url
        self.timeout_seconds = timeout_seconds
        self.queue: queue.Queue[Dict[str, object]] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="posture-event-publisher", daemon=True)
        self.sent_count = 0
        self.drop_count = 0
        self.fail_count = 0
        self.session_id: Optional[int] = None
        self.last_error_printed_at = 0.0

    def start(self) -> None:
        self.session_id = self._start_session()
        self.thread.start()
        print(f"Posture events -> {self.event_url}", flush=True)
        if self.session_id is not None:
            print(f"Posture session started: {self.session_id}", flush=True)

    def publish(self, result: PostureResult) -> None:
        try:
            self.queue.put_nowait(posture_result_payload(result, self.session_id))
        except queue.Full:
            self.drop_count += 1

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self._end_session()
        print(
            f"Posture event publisher stopped "
            f"(sent={self.sent_count}, failed={self.fail_count}, dropped={self.drop_count})",
            flush=True,
        )

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                payload = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._post_json(self.event_url, payload)
                self.sent_count += 1
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.fail_count += 1
                self._print_error(str(exc))
            finally:
                self.queue.task_done()

    def _post_json(self, url: str, payload: Dict[str, object]) -> Dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status >= 400:
                raise urllib.error.URLError(f"HTTP {response.status}")
            response_body = response.read().decode("utf-8")
            if not response_body:
                return {}
            data = json.loads(response_body)
            return data if isinstance(data, dict) else {}

    def _start_session(self) -> Optional[int]:
        try:
            response = self._post_json(self._session_endpoint("start"), {"timestamp": time.time()})
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self._print_error(f"session start failed: {exc}")
            return None
        session_id = response.get("session_id")
        try:
            return int(session_id)
        except (TypeError, ValueError):
            return None

    def _end_session(self) -> None:
        if self.session_id is None:
            return
        try:
            self._post_json(
                self._session_endpoint("end"),
                {"timestamp": time.time(), "session_id": self.session_id},
            )
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self._print_error(f"session end failed: {exc}")

    def _session_endpoint(self, action: str) -> str:
        parsed = urlparse(self.event_url)
        path = parsed.path
        if path.endswith("/api/posture-events/"):
            path = path[: -len("/posture-events/")] + f"/sessions/{action}/"
        elif path.endswith("/api/posture-events"):
            path = path[: -len("/posture-events")] + f"/sessions/{action}/"
        else:
            path = "/api/sessions/" + action + "/"
        return urlunparse(parsed._replace(path=path, query="", fragment=""))

    def _print_error(self, message: str) -> None:
        now = time.time()
        if now - self.last_error_printed_at < 5.0:
            return
        print(f"Posture event POST failed: {message}", flush=True)
        self.last_error_printed_at = now


def posture_result_payload(result: PostureResult, session_id: Optional[int] = None) -> Dict[str, object]:
    features = result.features
    depth_by_roi = {
        measurement.roi: measurement
        for measurement in (result.depth_measurements or [])
    }
    deltas = result.depth_deltas or {}
    payload = {
        "session_id": session_id,
        "timestamp": result.timestamp,
        "posture_state": canonical_posture_state(result),
        "raw_state": result.state,
        "reason": result.reason,
        "alarm": result.alarm,
        "elapsed_bad_seconds": result.elapsed_bad_seconds,
        "head_x": None if features is None else features.head_x,
        "shoulder_x": None if features is None else features.shoulder_x,
        "shoulder_tilt_px": None if features is None else features.shoulder_tilt,
        "shoulder_angle_deg": None if features is None else features.shoulder_angle_deg,
        "min_confidence": None if features is None else features.min_confidence,
        "head_m": _median(depth_by_roi.get("head")),
        "shoulder_m": _median(depth_by_roi.get("shoulder")),
        "chest_m": _median(depth_by_roi.get("chest")),
        "head_delta_m": deltas.get("head"),
        "shoulder_delta_m": deltas.get("shoulder"),
        "chest_delta_m": deltas.get("chest"),
        "nose_m": _median(depth_by_roi.get("nose")),
        "neck_m": _median(depth_by_roi.get("neck")),
        "nose_delta_m": result.nose_depth_delta_m,
        "pose_shoulder_tilt_delta_px": result.pose_shoulder_tilt_delta_px,
        "pose_shoulder_angle_delta_deg": result.pose_shoulder_angle_delta_deg,
        "neck_angle_delta_deg": result.neck_angle_delta_deg,
        "neck_pitch_deg": result.neck_pitch_deg,
        "nose_valid_ratio": result.nose_valid_ratio,
        "neck_valid_ratio": result.neck_valid_ratio,
    }
    return payload


def _median(measurement: Optional[DepthMeasurement]) -> Optional[float]:
    if measurement is None:
        return None
    return measurement.median_m
