from __future__ import annotations

# Temporary in-memory dashboard used for local testing without Django/SQLite.

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .models import PostureResult


POSTURE_STATES = ("normal", "turtle_neck", "shoulder_tilt", "out_of_range")


def canonical_posture_state(result: PostureResult) -> str:
    if result.reason in ("turtle_neck", "shoulder_tilt"):
        return result.reason
    if result.state == "normal" or result.reason == "normal":
        return "normal"
    return "out_of_range"


@dataclass
class StateEvent:
    timestamp: float
    state: str
    previous_state: Optional[str]


class PostureDashboardStats:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.last_update_at: Optional[float] = None
        self.current_state = "out_of_range"
        self.counts: Dict[str, int] = {state: 0 for state in POSTURE_STATES}
        self.durations: Dict[str, float] = {state: 0.0 for state in POSTURE_STATES}
        self.events: List[StateEvent] = []
        self.lock = threading.Lock()

    def update(self, result: PostureResult) -> None:
        state = canonical_posture_state(result)
        timestamp = float(result.timestamp)

        with self.lock:
            if self.last_update_at is None:
                self.last_update_at = timestamp
                self.current_state = state
                self.counts[state] += 1
                self.events.append(StateEvent(timestamp, state, None))
                return

            elapsed = max(0.0, timestamp - self.last_update_at)
            self.durations[self.current_state] += elapsed
            self.last_update_at = timestamp

            if state != self.current_state:
                previous_state = self.current_state
                self.current_state = state
                self.counts[state] += 1
                self.events.append(StateEvent(timestamp, state, previous_state))
                if len(self.events) > 200:
                    self.events = self.events[-200:]

    def snapshot(self) -> Dict[str, object]:
        now = time.time()
        with self.lock:
            durations = dict(self.durations)
            if self.last_update_at is not None:
                durations[self.current_state] += max(0.0, now - self.last_update_at)
            total_seconds = sum(durations.values())
            percentages = {
                state: (durations[state] / total_seconds * 100.0) if total_seconds > 0 else 0.0
                for state in POSTURE_STATES
            }
            return {
                "started_at": self.started_at,
                "now": now,
                "current_state": self.current_state,
                "total_seconds": total_seconds,
                "durations": durations,
                "percentages": percentages,
                "counts": dict(self.counts),
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "state": event.state,
                        "previous_state": event.previous_state,
                    }
                    for event in self.events[-50:]
                ],
            }

    def reset(self) -> None:
        with self.lock:
            self.started_at = time.time()
            self.last_update_at = None
            self.current_state = "out_of_range"
            self.counts = {state: 0 for state in POSTURE_STATES}
            self.durations = {state: 0.0 for state in POSTURE_STATES}
            self.events = []


class DashboardServer:
    def __init__(self, host: str, port: int, stats: PostureDashboardStats) -> None:
        self.host = host
        self.port = port
        self.stats = stats
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        handler = self._build_handler()
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="dashboard-server", daemon=True)
        self.thread.start()
        print(f"Dashboard server: http://{self.host}:{self.port}", flush=True)

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _build_handler(self):
        stats = self.stats

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/dashboard"):
                    self._send_html(DASHBOARD_HTML)
                    return
                if path == "/api/status":
                    self._send_json(stats.snapshot())
                    return
                if path == "/api/events":
                    self._send_json({"events": stats.snapshot()["events"]})
                    return
                self.send_error(404, "Not found")

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path == "/api/reset":
                    stats.reset()
                    self._send_json({"ok": True})
                    return
                self.send_error(404, "Not found")

            def log_message(self, format: str, *args) -> None:
                return

            def _send_json(self, payload: Dict[str, object]) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_html(self, html: str) -> None:
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


DASHBOARD_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chairshot Posture Dashboard</title>
  <style>
    :root {
      --bg: #101418;
      --panel: #182028;
      --text: #e7eef5;
      --muted: #8a9aaa;
      --normal: #3ddc97;
      --turtle: #ffb020;
      --tilt: #ff6b6b;
      --range: #7c8da1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at 20% 10%, #22313f, var(--bg) 42%);
      color: var(--text);
    }
    main { width: min(1100px, 94vw); margin: 32px auto; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 16px; margin-bottom: 24px; }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 48px); letter-spacing: -0.04em; }
    button {
      border: 0; border-radius: 12px; padding: 11px 16px; color: var(--text);
      background: #2b3947; cursor: pointer; font-weight: 700;
    }
    .current {
      background: linear-gradient(135deg, #202a34, #111820);
      border: 1px solid #2f4050;
      border-radius: 24px;
      padding: 24px;
      margin-bottom: 18px;
    }
    .state { font-size: clamp(34px, 7vw, 76px); font-weight: 900; letter-spacing: -0.06em; }
    .meta { color: var(--muted); margin-top: 8px; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .card { background: rgba(24,32,40,0.86); border: 1px solid #2a3a48; border-radius: 18px; padding: 18px; }
    .label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .08em; }
    .value { font-size: 30px; font-weight: 850; margin-top: 8px; }
    .bar { height: 10px; background: #0d1217; border-radius: 999px; overflow: hidden; margin-top: 14px; }
    .fill { height: 100%; width: 0%; transition: width .25s ease; }
    .normal { color: var(--normal); } .normal-fill { background: var(--normal); }
    .turtle_neck { color: var(--turtle); } .turtle_neck-fill { background: var(--turtle); }
    .shoulder_tilt { color: var(--tilt); } .shoulder_tilt-fill { background: var(--tilt); }
    .out_of_range { color: var(--range); } .out_of_range-fill { background: var(--range); }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr 1fr; } header { align-items: start; flex-direction: column; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Posture Dashboard</h1>
      <div class="meta">normal · turtle_neck · shoulder_tilt · out_of_range</div>
    </div>
    <button onclick="resetStats()">Reset</button>
  </header>
  <section class="current">
    <div class="label">Current State</div>
    <div id="current" class="state">loading</div>
    <div id="total" class="meta">-</div>
  </section>
  <section class="grid" id="cards"></section>
</main>
<script>
const states = ["normal", "turtle_neck", "shoulder_tilt", "out_of_range"];
const labels = {
  normal: "Normal",
  turtle_neck: "Turtle Neck",
  shoulder_tilt: "Shoulder Tilt",
  out_of_range: "Out Of Range"
};
function formatSeconds(value) {
  const total = Math.max(0, Math.round(value || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
function card(state, payload) {
  const seconds = payload.durations[state] || 0;
  const percent = payload.percentages[state] || 0;
  const count = payload.counts[state] || 0;
  return `<article class="card">
    <div class="label">${labels[state]}</div>
    <div class="value ${state}">${formatSeconds(seconds)}</div>
    <div class="meta">${percent.toFixed(1)}% · ${count} entries</div>
    <div class="bar"><div class="fill ${state}-fill" style="width:${Math.min(100, percent)}%"></div></div>
  </article>`;
}
async function refresh() {
  const res = await fetch("/api/status");
  const data = await res.json();
  document.getElementById("current").textContent = labels[data.current_state] || data.current_state;
  document.getElementById("current").className = `state ${data.current_state}`;
  document.getElementById("total").textContent = `Total sitting observation: ${formatSeconds(data.total_seconds)}`;
  document.getElementById("cards").innerHTML = states.map((state) => card(state, data)).join("");
}
async function resetStats() {
  await fetch("/api/reset", { method: "POST" });
  await refresh();
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
