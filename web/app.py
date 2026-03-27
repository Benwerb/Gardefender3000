import sys
import threading
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify,
)

# Allow importing water_plants from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import water_plants


def create_app():
    app = Flask(__name__)

    from web.config import SECRET_KEY, PASSWORD, DATABASE_PATH
    app.secret_key = SECRET_KEY
    app.config["PASSWORD"] = PASSWORD
    app.config["DATABASE_PATH"] = DATABASE_PATH

    # Ensure DB + table exist on startup
    water_plants.init_db(Path(DATABASE_PATH))

    # Thread-safe watering state
    _lock = threading.Lock()
    _state = {"is_watering": False}

    # ── Auth helpers ──────────────────────────────────────────────────────

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    # ── Auth routes ───────────────────────────────────────────────────────

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

    # ── Main routes ───────────────────────────────────────────────────────

    @app.route("/")
    @login_required
    def index():
        return render_template("index.html")

    @app.route("/water", methods=["POST"])
    @login_required
    def trigger_water():
        with _lock:
            if _state["is_watering"]:
                return jsonify({"status": "already_running"}), 409

        try:
            duration = int(request.form.get("duration", 10))
        except (ValueError, TypeError):
            duration = 10
        duration = max(5, min(duration, 300))

        def _do_water():
            with _lock:
                _state["is_watering"] = True
            try:
                water_plants.water(
                    duration=duration,
                    trigger_type="manual",
                    db_path=Path(app.config["DATABASE_PATH"]),
                )
            finally:
                with _lock:
                    _state["is_watering"] = False

        t = threading.Thread(target=_do_water, daemon=True)
        t.start()
        return jsonify({"status": "started", "duration": duration}), 202

    @app.route("/api/status")
    @login_required
    def api_status():
        db_path = Path(app.config["DATABASE_PATH"])
        last = _get_last_watering(db_path)
        with _lock:
            watering = _state["is_watering"]
        return jsonify({"is_watering": watering, "last_watering": last})

    @app.route("/api/log")
    @login_required
    def api_log():
        db_path = Path(app.config["DATABASE_PATH"])
        rows = _get_recent_waterings(db_path)
        return jsonify(rows)

    return app


# ── DB helpers (read-only queries against water_plants schema) ────────────

def _get_last_watering(db_path: Path):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM watering_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            return dict(row)
    except sqlite3.OperationalError:
        pass
    return None


def _get_recent_waterings(db_path: Path, limit: int = 50):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM watering_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
