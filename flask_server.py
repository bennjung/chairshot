#!/usr/bin/env python3
"""Flask + SQLite posture dashboard server."""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, Response, g, jsonify, render_template_string, request


POSTURE_STATES = ("normal", "turtle_neck", "shoulder_tilt", "out_of_range")
SESSION_TABLE = "posture_logs_posturesession"
LOG_TABLE = "posture_logs_posturelog"


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {SESSION_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    total_seconds REAL NOT NULL DEFAULT 0.0,
    normal_seconds REAL NOT NULL DEFAULT 0.0,
    turtle_neck_seconds REAL NOT NULL DEFAULT 0.0,
    shoulder_tilt_seconds REAL NOT NULL DEFAULT 0.0,
    out_of_range_seconds REAL NOT NULL DEFAULT 0.0,
    event_count INTEGER NOT NULL DEFAULT 0,
    normal_count INTEGER NOT NULL DEFAULT 0,
    turtle_neck_count INTEGER NOT NULL DEFAULT 0,
    shoulder_tilt_count INTEGER NOT NULL DEFAULT 0,
    out_of_range_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    observed_at TEXT NOT NULL,
    session_id INTEGER,
    posture_state TEXT NOT NULL,
    raw_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    alarm INTEGER NOT NULL DEFAULT 0,
    elapsed_bad_seconds REAL NOT NULL DEFAULT 0.0,
    head_x REAL,
    shoulder_x REAL,
    shoulder_tilt_px REAL,
    shoulder_angle_deg REAL,
    min_confidence REAL,
    head_m REAL,
    shoulder_m REAL,
    chest_m REAL,
    head_delta_m REAL,
    shoulder_delta_m REAL,
    chest_delta_m REAL,
    nose_m REAL,
    neck_m REAL,
    nose_delta_m REAL,
    pose_shoulder_tilt_delta_px REAL,
    pose_shoulder_angle_delta_deg REAL,
    neck_angle_delta_deg REAL,
    neck_pitch_deg REAL,
    nose_valid_ratio REAL,
    neck_valid_ratio REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS posture_session_started_idx
ON {SESSION_TABLE} (started_at, id);

CREATE INDEX IF NOT EXISTS posture_session_active_idx
ON {SESSION_TABLE} (is_active, started_at, id);

CREATE INDEX IF NOT EXISTS posture_log_observed_idx
ON {LOG_TABLE} (observed_at, id);

CREATE INDEX IF NOT EXISTS posture_log_timestamp_idx
ON {LOG_TABLE} (timestamp);

CREATE INDEX IF NOT EXISTS posture_log_state_idx
ON {LOG_TABLE} (posture_state);

CREATE INDEX IF NOT EXISTS posture_log_session_idx
ON {LOG_TABLE} (session_id, observed_at, id);
"""


LOG_COLUMNS = {
    "session_id": "INTEGER",
    "nose_m": "REAL",
    "neck_m": "REAL",
    "nose_delta_m": "REAL",
    "pose_shoulder_tilt_delta_px": "REAL",
    "pose_shoulder_angle_delta_deg": "REAL",
    "neck_angle_delta_deg": "REAL",
    "neck_pitch_deg": "REAL",
    "nose_valid_ratio": "REAL",
    "neck_valid_ratio": "REAL",
}


DASHBOARD_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chairshot Posture Dashboard</title>
  <style>
    :root {
      --bg: #101418;
      --panel: #192027;
      --text: #ecf3f8;
      --muted: #96a8b7;
      --green: #2fd17c;
      --red: #ff5d5d;
      --yellow: #ffd65a;
      --blue: #45b7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 10%, rgba(69, 183, 255, 0.18), transparent 34rem),
        radial-gradient(circle at 85% 0%, rgba(255, 214, 90, 0.14), transparent 30rem),
        var(--bg);
    }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 16px; margin-bottom: 24px; }
    h1 { margin: 0; font-size: clamp(30px, 5vw, 54px); letter-spacing: -0.06em; }
    .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .panel { border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 18px; background: rgba(25,32,39,0.82); backdrop-filter: blur(10px); }
    .metric { font-size: 34px; font-weight: 800; margin-top: 8px; }
    .normal { color: var(--green); }
    .turtle_neck, .shoulder_tilt { color: var(--red); }
    .out_of_range { color: var(--yellow); }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.07); text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    @media (max-width: 820px) { .grid { grid-template-columns: 1fr 1fr; } header { display: block; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="muted">Flask + SQLite</div>
      <h1>Posture Dashboard</h1>
    </div>
    <div class="muted" id="updated">loading...</div>
  </header>

  <section class="grid">
    <div class="panel"><div class="muted">Current</div><div class="metric" id="current">-</div></div>
    <div class="panel"><div class="muted">Normal</div><div class="metric normal" id="normal">0</div></div>
    <div class="panel"><div class="muted">Bad</div><div class="metric turtle_neck" id="bad">0</div></div>
    <div class="panel"><div class="muted">Out of Range</div><div class="metric out_of_range" id="oor">0</div></div>
  </section>

  <section class="panel" style="margin-top: 14px;">
    <h2 style="margin: 0 0 8px;">Sessions</h2>
    <table>
      <thead><tr><th>ID</th><th>Started</th><th>Ended</th><th>Total</th><th>Normal</th><th>Turtle</th><th>Tilt</th><th>Out</th></tr></thead>
      <tbody id="sessions"></tbody>
    </table>
  </section>

  <section class="panel" style="margin-top: 14px;">
    <h2 style="margin: 0 0 8px;">Recent Logs</h2>
    <table>
      <thead><tr><th>Time</th><th>State</th><th>Reason</th><th>Alarm</th><th>Neck</th><th>Shoulder</th></tr></thead>
      <tbody id="logs"></tbody>
    </table>
  </section>
</main>
<script>
const fmtSeconds = (v) => `${Number(v || 0).toFixed(1)}s`;
const pct = (v) => `${Number(v || 0).toFixed(1)}%`;

async function refresh() {
  const [status, logs, sessions] = await Promise.all([
    fetch('/api/status/').then(r => r.json()),
    fetch('/api/logs/?limit=20').then(r => r.json()),
    fetch('/api/sessions/?limit=10').then(r => r.json()),
  ]);
  const current = status.current_state || '-';
  document.getElementById('current').textContent = current;
  document.getElementById('current').className = `metric ${current}`;
  document.getElementById('normal').textContent = pct(status.percentages.normal);
  document.getElementById('bad').textContent = pct((status.percentages.turtle_neck || 0) + (status.percentages.shoulder_tilt || 0));
  document.getElementById('oor').textContent = pct(status.percentages.out_of_range);
  document.getElementById('updated').textContent = new Date().toLocaleTimeString();
  document.getElementById('logs').innerHTML = logs.logs.map(row => `
    <tr>
      <td>${new Date(row.observed_at).toLocaleTimeString()}</td>
      <td class="${row.posture_state}">${row.posture_state}</td>
      <td>${row.reason}</td>
      <td>${row.alarm ? 'yes' : 'no'}</td>
      <td>${Number(row.neck_angle_delta_deg || 0).toFixed(1)}deg</td>
      <td>${Number(row.pose_shoulder_angle_delta_deg || 0).toFixed(1)}deg</td>
    </tr>`).join('');
  document.getElementById('sessions').innerHTML = sessions.sessions.map(row => `
    <tr>
      <td>${row.id}${row.is_active ? ' active' : ''}</td>
      <td>${new Date(row.started_at).toLocaleString()}</td>
      <td>${row.ended_at ? new Date(row.ended_at).toLocaleString() : '-'}</td>
      <td>${fmtSeconds(row.total_seconds)}</td>
      <td>${fmtSeconds(row.normal_seconds)}</td>
      <td>${fmtSeconds(row.turtle_neck_seconds)}</td>
      <td>${fmtSeconds(row.shoulder_tilt_seconds)}</td>
      <td>${fmtSeconds(row.out_of_range_seconds)}</td>
    </tr>`).join('');
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def create_app(db_path: Path | str = "db.sqlite3") -> Flask:
    app = Flask(__name__)
    app.config["DATABASE"] = str(db_path)
    init_database(app)

    @app.get("/")
    def dashboard() -> str:
        return render_template_string(DASHBOARD_HTML)

    @app.get("/api/status/")
    def status_api() -> Response:
        window_minutes = max(_int_query("minutes", 60), 1)
        since = time.time() - window_minutes * 60.0
        db = get_db()
        counts = {state: 0 for state in POSTURE_STATES}
        for row in db.execute(
            f"SELECT posture_state, COUNT(*) AS count FROM {LOG_TABLE} "
            "WHERE timestamp >= ? GROUP BY posture_state",
            (since,),
        ):
            if row["posture_state"] in counts:
                counts[row["posture_state"]] = int(row["count"])
        total = sum(counts.values())
        percentages = {
            state: (counts[state] / total * 100.0) if total else 0.0
            for state in POSTURE_STATES
        }
        latest = db.execute(
            f"SELECT * FROM {LOG_TABLE} ORDER BY observed_at DESC, id DESC LIMIT 1"
        ).fetchone()
        active_session = db.execute(
            f"SELECT * FROM {SESSION_TABLE} WHERE is_active = 1 "
            "ORDER BY started_at DESC, id DESC LIMIT 1"
        ).fetchone()
        return jsonify(
            {
                "window_minutes": window_minutes,
                "current_state": None if latest is None else latest["posture_state"],
                "latest": None if latest is None else serialize_log(latest),
                "active_session": None if active_session is None else serialize_session(active_session),
                "total_rows": total,
                "counts": counts,
                "percentages": percentages,
            }
        )

    @app.get("/api/logs/")
    def logs_api() -> Response:
        limit = min(max(_int_query("limit", 100), 1), 1000)
        session_id = _int_query("session_id", 0)
        db = get_db()
        if session_id > 0:
            rows = db.execute(
                f"SELECT * FROM {LOG_TABLE} WHERE session_id = ? "
                "ORDER BY observed_at DESC, id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = db.execute(
                f"SELECT * FROM {LOG_TABLE} ORDER BY observed_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return jsonify({"logs": [serialize_log(row) for row in rows]})

    @app.get("/api/sessions/")
    def sessions_api() -> Response:
        limit = min(max(_int_query("limit", 50), 1), 500)
        rows = get_db().execute(
            f"SELECT * FROM {SESSION_TABLE} ORDER BY started_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return jsonify({"sessions": [serialize_session(row) for row in rows]})

    @app.post("/api/sessions/start/")
    def session_start_api() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid_json"}), 400
        timestamp = _payload_timestamp(payload)
        now = _iso_from_timestamp(timestamp)
        db = get_db()
        with db:
            cursor = db.execute(
                f"INSERT INTO {SESSION_TABLE} "
                "(started_at, is_active, created_at, updated_at) VALUES (?, 1, ?, ?)",
                (now, now, now),
            )
        return jsonify({"ok": True, "session_id": cursor.lastrowid})

    @app.post("/api/sessions/end/")
    def session_end_api() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid_json"}), 400
        session_id = _int_payload(payload.get("session_id"))
        if session_id is None:
            return jsonify({"error": "missing_session_id"}), 400
        timestamp = _payload_timestamp(payload)
        ended_at = _iso_from_timestamp(timestamp)
        db = get_db()
        with db:
            session = db.execute(
                f"SELECT * FROM {SESSION_TABLE} WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return jsonify({"error": "session_not_found"}), 404
            latest = db.execute(
                f"SELECT * FROM {LOG_TABLE} WHERE session_id = ? "
                "ORDER BY observed_at DESC, id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if latest is not None:
                _add_state_duration(db, session_id, latest["posture_state"], timestamp - float(latest["timestamp"]))
            total_seconds = max(0.0, timestamp - _timestamp_from_iso(session["started_at"]))
            db.execute(
                f"UPDATE {SESSION_TABLE} SET ended_at = ?, is_active = 0, "
                "total_seconds = ?, updated_at = ? WHERE id = ?",
                (ended_at, total_seconds, ended_at, session_id),
            )
        session = get_db().execute(
            f"SELECT * FROM {SESSION_TABLE} WHERE id = ?",
            (session_id,),
        ).fetchone()
        return jsonify({"ok": True, "session": serialize_session(session)})

    @app.post("/api/posture-events/")
    def posture_event_api() -> Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "invalid_json"}), 400

        timestamp = _payload_timestamp(payload)
        posture_state = str(payload.get("posture_state") or _canonical_state(payload))
        if posture_state not in POSTURE_STATES:
            return jsonify({"error": "invalid_posture_state"}), 400

        observed_at = _iso_from_timestamp(timestamp)
        session_id = _int_payload(payload.get("session_id"))
        db = get_db()
        with db:
            session = None
            previous = None
            if session_id is not None:
                session = db.execute(
                    f"SELECT * FROM {SESSION_TABLE} WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if session is not None:
                    previous = db.execute(
                        f"SELECT * FROM {LOG_TABLE} WHERE session_id = ? "
                        "ORDER BY observed_at DESC, id DESC LIMIT 1",
                        (session_id,),
                    ).fetchone()

            cursor = db.execute(
                f"""
                INSERT INTO {LOG_TABLE} (
                    session_id, timestamp, observed_at, posture_state, raw_state, reason, alarm,
                    elapsed_bad_seconds, head_x, shoulder_x, shoulder_tilt_px,
                    shoulder_angle_deg, min_confidence, head_m, shoulder_m, chest_m,
                    head_delta_m, shoulder_delta_m, chest_delta_m, nose_m, neck_m,
                    nose_delta_m, pose_shoulder_tilt_delta_px,
                    pose_shoulder_angle_delta_deg, neck_angle_delta_deg, neck_pitch_deg,
                    nose_valid_ratio, neck_valid_ratio, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session["id"] if session is not None else None,
                    timestamp,
                    observed_at,
                    posture_state,
                    str(payload.get("raw_state") or ""),
                    str(payload.get("reason") or posture_state),
                    1 if bool(payload.get("alarm", False)) else 0,
                    _float_value(payload.get("elapsed_bad_seconds")) or 0.0,
                    _float_value(payload.get("head_x")),
                    _float_value(payload.get("shoulder_x")),
                    _float_value(payload.get("shoulder_tilt_px")),
                    _float_value(payload.get("shoulder_angle_deg")),
                    _float_value(payload.get("min_confidence")),
                    _float_value(payload.get("head_m")),
                    _float_value(payload.get("shoulder_m")),
                    _float_value(payload.get("chest_m")),
                    _float_value(payload.get("head_delta_m")),
                    _float_value(payload.get("shoulder_delta_m")),
                    _float_value(payload.get("chest_delta_m")),
                    _float_value(payload.get("nose_m")),
                    _float_value(payload.get("neck_m")),
                    _float_value(payload.get("nose_delta_m")),
                    _float_value(payload.get("pose_shoulder_tilt_delta_px")),
                    _float_value(payload.get("pose_shoulder_angle_delta_deg")),
                    _float_value(payload.get("neck_angle_delta_deg")),
                    _float_value(payload.get("neck_pitch_deg")),
                    _float_value(payload.get("nose_valid_ratio")),
                    _float_value(payload.get("neck_valid_ratio")),
                    observed_at,
                ),
            )

            if session is not None:
                if previous is not None:
                    _add_state_duration(db, session["id"], previous["posture_state"], timestamp - float(previous["timestamp"]))
                _increment_session_count(db, session["id"], posture_state)
                total_seconds = max(0.0, timestamp - _timestamp_from_iso(session["started_at"]))
                db.execute(
                    f"UPDATE {SESSION_TABLE} SET total_seconds = ?, updated_at = ? WHERE id = ?",
                    (total_seconds, observed_at, session["id"]),
                )

        return jsonify({"ok": True, "id": cursor.lastrowid, "session_id": None if session is None else session["id"]})

    @app.teardown_appcontext
    def close_db(_: Optional[BaseException]) -> None:
        db = g.pop("db", None)
        if db is not None:
            db.close()

    return app


def init_database(app: Flask) -> None:
    db_path = Path(app.config["DATABASE"])
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as db:
        db.executescript(SCHEMA)
        for name, definition in LOG_COLUMNS.items():
            _ensure_column(db, LOG_TABLE, name, definition)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        db = sqlite3.connect(current_app_db_path())
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        g.db = db
    return g.db


def current_app_db_path() -> str:
    from flask import current_app

    return str(current_app.config["DATABASE"])


def _ensure_column(db: sqlite3.Connection, table: str, name: str, definition: str) -> None:
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _payload_timestamp(payload: Dict[str, Any]) -> float:
    timestamp = _float_value(payload.get("timestamp"))
    return time.time() if timestamp is None else timestamp


def _canonical_state(payload: Dict[str, Any]) -> str:
    reason = str(payload.get("reason") or "")
    raw_state = str(payload.get("raw_state") or "")
    if reason in ("turtle_neck", "shoulder_tilt"):
        return reason
    if raw_state == "normal" or reason == "normal":
        return "normal"
    return "out_of_range"


def _add_state_duration(db: sqlite3.Connection, session_id: int, state: str, seconds: float) -> None:
    if state not in POSTURE_STATES or seconds <= 0.0:
        return
    field = f"{state}_seconds"
    db.execute(
        f"UPDATE {SESSION_TABLE} SET {field} = {field} + ? WHERE id = ?",
        (seconds, session_id),
    )


def _increment_session_count(db: sqlite3.Connection, session_id: int, state: str) -> None:
    if state not in POSTURE_STATES:
        return
    field = f"{state}_count"
    db.execute(
        f"UPDATE {SESSION_TABLE} SET event_count = event_count + 1, "
        f"{field} = {field} + 1 WHERE id = ?",
        (session_id,),
    )


def serialize_log(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "timestamp": row["timestamp"],
        "observed_at": row["observed_at"],
        "posture_state": row["posture_state"],
        "raw_state": row["raw_state"],
        "reason": row["reason"],
        "alarm": bool(row["alarm"]),
        "elapsed_bad_seconds": row["elapsed_bad_seconds"],
        "head_delta_m": row["head_delta_m"],
        "shoulder_delta_m": row["shoulder_delta_m"],
        "chest_delta_m": row["chest_delta_m"],
        "nose_delta_m": row["nose_delta_m"],
        "pose_shoulder_tilt_delta_px": row["pose_shoulder_tilt_delta_px"],
        "pose_shoulder_angle_delta_deg": row["pose_shoulder_angle_delta_deg"],
        "neck_angle_delta_deg": row["neck_angle_delta_deg"],
        "neck_pitch_deg": row["neck_pitch_deg"],
        "nose_valid_ratio": row["nose_valid_ratio"],
        "neck_valid_ratio": row["neck_valid_ratio"],
    }


def serialize_session(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "id": row["id"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "is_active": bool(row["is_active"]),
        "total_seconds": row["total_seconds"],
        "normal_seconds": row["normal_seconds"],
        "turtle_neck_seconds": row["turtle_neck_seconds"],
        "shoulder_tilt_seconds": row["shoulder_tilt_seconds"],
        "out_of_range_seconds": row["out_of_range_seconds"],
        "event_count": row["event_count"],
        "normal_count": row["normal_count"],
        "turtle_neck_count": row["turtle_neck_count"],
        "shoulder_tilt_count": row["shoulder_tilt_count"],
        "out_of_range_count": row["out_of_range_count"],
    }


def _int_query(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_payload(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _timestamp_from_iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flask posture dashboard server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8800, help="HTTP port.")
    parser.add_argument("--db", default="db.sqlite3", help="SQLite DB path.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.db))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
