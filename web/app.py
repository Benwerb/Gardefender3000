import sys
import os
import threading
import sqlite3
import time
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, send_from_directory, Response,
)
from flask_socketio import SocketIO, emit

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import water_plants

# Shared state between Flask and the camera thread
_state = {
    "running":      False,
    "mode":         "auto",   # "auto" | "manual"
    "latest_frame": None,     # JPEG bytes
    "moving_pan":   0,        # -1 / 0 / 1
    "moving_tilt":  0,
    "manual_fire":  False,
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
    _start_camera_thread()

    # Thread-safe watering state
    _water_lock = threading.Lock()
    _water_state = {"is_watering": False}

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

    # ── Manual control page ───────────────────────────────────────────────

    @app.route("/manual")
    @login_required
    def manual():
        return render_template("manual.html", mode=_state["mode"])

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
