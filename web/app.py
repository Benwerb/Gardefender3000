import sys
import os
import threading
import sqlite3
import time
import psutil
from functools import wraps
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_from_directory, Response,
)
from flask_socketio import SocketIO, emit

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import water_plants

# COCO class presets (matches class IDs used in defendGarden.py)
CLASS_PRESETS = {
    "animals":    {15, 16, 17, 18, 19, 20, 21, 22, 23, 24},  # bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe
    "person":     {0},
    "everything": {0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},  # animals + person
}
DEFAULT_CLASSES = CLASS_PRESETS["animals"].copy()

# Shared state between Flask and the camera thread
_state = {
    "running":        False,
    "mode":           "auto",
    "latest_frame":   None,
    "moving_pan":     0,
    "moving_tilt":    0,
    "manual_fire":    False,
    "target_classes": DEFAULT_CLASSES.copy(),
    "fps":            0.0,
    "pan_angle":      87,
    "tilt_angle":     135,
}
_frame_lock = threading.Lock()
_camera_thread = None


def _start_camera_thread():
    global _camera_thread
    if _camera_thread and _camera_thread.is_alive():
        return
    try:
        import defendGarden
        _state["running"] = True
        _camera_thread = threading.Thread(
            target=defendGarden.run_loop,
            args=(_state, _frame_lock),
            daemon=True,
        )
        _camera_thread.start()
        print("Camera thread started.")
    except Exception as e:
        print(f"Camera thread failed to start (not on Pi?): {e}")


def create_app():
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

    from web.config import SECRET_KEY, PASSWORD, DATABASE_PATH, DETECTIONS_PATH
    app.secret_key = SECRET_KEY
    app.config["PASSWORD"] = PASSWORD
    app.config["DATABASE_PATH"] = DATABASE_PATH
    app.config["DETECTIONS_PATH"] = DETECTIONS_PATH

    water_plants.init_db(Path(DATABASE_PATH))
    _init_schedule_db(Path(DATABASE_PATH))
    _start_camera_thread()

    # Thread-safe watering state
    _water_lock = threading.Lock()
    _water_state = {"is_watering": False}

    # Scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()
    _load_schedules(scheduler, Path(DATABASE_PATH), _water_lock, _water_state)

    # ── Auth ──────────────────────────────────────────────────────────────

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            if request.form.get("password") == app.config["PASSWORD"]:
                session["logged_in"] = True
                return redirect(url_for("index"))
            error = "Wrong password."
        return render_template("login.html", error=error)

    @app.route("/logout")
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ── Dashboard ─────────────────────────────────────────────────────────

    @app.route("/")
    @login_required
    def index():
        return render_template("index.html")

    @app.route("/water", methods=["POST"])
    @login_required
    def trigger_water():
        with _water_lock:
            if _water_state["is_watering"]:
                return jsonify({"status": "already_running"}), 409

        try:
            duration = int(request.form.get("duration", 10))
        except (ValueError, TypeError):
            duration = 10
        duration = max(5, min(duration, 300))

        def _do_water():
            with _water_lock:
                _water_state["is_watering"] = True
            try:
                water_plants.water(
                    duration=duration,
                    trigger_type="manual",
                    db_path=Path(app.config["DATABASE_PATH"]),
                )
            finally:
                with _water_lock:
                    _water_state["is_watering"] = False

        threading.Thread(target=_do_water, daemon=True).start()
        return jsonify({"status": "started", "duration": duration}), 202

    @app.route("/api/status")
    @login_required
    def api_status():
        last = _get_last_watering(Path(app.config["DATABASE_PATH"]))
        with _water_lock:
            watering = _water_state["is_watering"]
        return jsonify({"is_watering": watering, "last_watering": last})

    @app.route("/api/log")
    @login_required
    def api_log():
        rows = _get_recent_waterings(Path(app.config["DATABASE_PATH"]))
        return jsonify(rows)

    # ── Detection gallery ─────────────────────────────────────────────────

    @app.route("/api/detections")
    @login_required
    def api_detections():
        detections_dir = Path(app.config["DETECTIONS_PATH"])
        if not detections_dir.exists():
            return jsonify([])
        events = []
        for folder in sorted(detections_dir.iterdir(), reverse=True):
            if not folder.is_dir():
                continue
            frames = sorted(f.name for f in folder.iterdir() if f.suffix == ".jpg")
            if not frames:
                continue
            parts = folder.name.split("_", 1)
            events.append({
                "name":      folder.name,
                "label":     parts[0] if len(parts) == 2 else folder.name,
                "timestamp": parts[1].replace("_", " ") if len(parts) == 2 else "",
                "frames":    frames,
                "thumbnail": frames[0],
            })
        return jsonify(events)

    @app.route("/detections/<path:filename>")
    @login_required
    def serve_detection(filename):
        return send_from_directory(app.config["DETECTIONS_PATH"], filename)

    # ── Video stream ──────────────────────────────────────────────────────

    @app.route("/video_feed")
    @login_required
    def video_feed():
        def generate():
            while True:
                with _frame_lock:
                    frame = _state.get("latest_frame")
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                time.sleep(1 / 30)
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ── Stats API ─────────────────────────────────────────────────────────

    @app.route("/api/stats")
    @login_required
    def api_stats():
        temp = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = round(int(f.read().strip()) / 1000, 1)
        except OSError:
            pass
        return jsonify({
            "fps":            round(_state.get("fps", 0), 1),
            "cpu":            psutil.cpu_percent(),
            "temp_c":         temp,
            "pan_angle":      round(_state.get("pan_angle",  0)),
            "tilt_angle":     round(_state.get("tilt_angle", 0)),
            "mode":           _state.get("mode", "auto"),
            "target_classes": list(_state.get("target_classes", [])),
        })

    # ── Schedule API ──────────────────────────────────────────────────────

    @app.route("/api/schedules", methods=["GET"])
    @login_required
    def api_get_schedules():
        return jsonify(_get_schedules(Path(app.config["DATABASE_PATH"])))

    @app.route("/api/schedules", methods=["POST"])
    @login_required
    def api_add_schedule():
        data = request.get_json()
        hour        = int(data["hour"])
        minute      = int(data["minute"])
        duration    = int(data["duration_sec"])
        db_path     = Path(app.config["DATABASE_PATH"])
        schedule_id = _save_schedule(db_path, hour, minute, duration)
        _add_scheduler_job(scheduler, schedule_id, hour, minute, duration,
                           db_path, _water_lock, _water_state)
        return jsonify({"id": schedule_id}), 201

    @app.route("/api/schedules/<int:schedule_id>", methods=["DELETE"])
    @login_required
    def api_delete_schedule(schedule_id):
        _delete_schedule(Path(app.config["DATABASE_PATH"]), schedule_id)
        job_id = f"water_{schedule_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
        return jsonify({"status": "deleted"})

    # ── Manual control page ───────────────────────────────────────────────

    @app.route("/manual")
    @login_required
    def manual():
        return render_template("manual.html", mode=_state["mode"],
                               presets=list(CLASS_PRESETS.keys()))

    # ── SocketIO events ───────────────────────────────────────────────────

    @socketio.on("set_mode")
    def handle_set_mode(data):
        mode = data.get("mode")
        if mode in ("auto", "manual"):
            _state["mode"]        = mode
            _state["moving_pan"]  = 0
            _state["moving_tilt"] = 0
            _state["manual_fire"] = False
            emit("mode_changed", {"mode": mode}, broadcast=True)

    @socketio.on("move")
    def handle_move(data):
        if _state.get("mode") != "manual":
            return
        _state["moving_pan"]  = int(data.get("pan",  0))
        _state["moving_tilt"] = int(data.get("tilt", 0))

    @socketio.on("set_classes")
    def handle_set_classes(data):
        preset = data.get("preset")
        if preset in CLASS_PRESETS:
            _state["target_classes"] = CLASS_PRESETS[preset].copy()
            emit("classes_changed", {"preset": preset}, broadcast=True)

    @socketio.on("fire")
    def handle_fire():
        if _state.get("mode") == "manual":
            _state["manual_fire"] = True

    return app, socketio


# ── DB helpers ────────────────────────────────────────────────────────────

def _get_last_watering(db_path: Path):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM watering_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None


def _get_recent_waterings(db_path: Path, limit: int = 50):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM watering_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


# ── Schedule DB helpers ───────────────────────────────────────────────────

def _init_schedule_db(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS water_schedules (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                hour         INTEGER NOT NULL,
                minute       INTEGER NOT NULL,
                duration_sec INTEGER NOT NULL
            )
        """)
        conn.commit()


def _get_schedules(db_path: Path):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM water_schedules ORDER BY hour, minute"
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def _save_schedule(db_path: Path, hour: int, minute: int, duration_sec: int) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO water_schedules (hour, minute, duration_sec) VALUES (?, ?, ?)",
            (hour, minute, duration_sec),
        )
        conn.commit()
        return cur.lastrowid


def _delete_schedule(db_path: Path, schedule_id: int):
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM water_schedules WHERE id = ?", (schedule_id,))
        conn.commit()


def _add_scheduler_job(scheduler, schedule_id, hour, minute, duration_sec,
                       db_path, water_lock, water_state):
    def _job():
        with water_lock:
            if water_state["is_watering"]:
                return
            water_state["is_watering"] = True
        try:
            water_plants.water(duration=duration_sec, trigger_type="schedule",
                               db_path=db_path)
        finally:
            with water_lock:
                water_state["is_watering"] = False

    scheduler.add_job(
        _job,
        trigger="cron",
        hour=hour,
        minute=minute,
        id=f"water_{schedule_id}",
        replace_existing=True,
    )


def _load_schedules(scheduler, db_path, water_lock, water_state):
    for s in _get_schedules(db_path):
        _add_scheduler_job(scheduler, s["id"], s["hour"], s["minute"],
                           s["duration_sec"], db_path, water_lock, water_state)
