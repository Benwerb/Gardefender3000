#!/usr/bin/env python3
"""
water_plants.py
---------------
Controls a MOSFET-triggered 9V sprinkler valve via Raspberry Pi GPIO.

Wiring:
    GPIO PIN (BCM) --> MOSFET Gate (via 1k resistor)
    MOSFET Drain   --> Sprinkler valve negative lead
    Sprinkler valve positive lead --> 9V power supply positive
    9V power supply GND + MOSFET Source --> Common GND (shared with Pi GND)

Usage:
    Standalone:  python3 water_plants.py
    As module:   from water_plants import water, WateringError
"""

import sqlite3
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

# ── Try importing GPIO; fall back to a stub for dev/testing on non-Pi hardware ──
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except (ImportError, RuntimeError):
    ON_PI = False
    logging.warning("RPi.GPIO not available — running in simulation mode.")


# ─────────────────────────────────────────────
#  CONFIG  (edit these values to match your setup)
# ─────────────────────────────────────────────

VALVE_PIN       = 27          # BCM GPIO pin connected to MOSFET gate (pin 17 reserved for defense solenoid)
DEFAULT_DURATION = 10         # seconds the valve stays open by default
DB_PATH         = Path(__file__).parent / "plants.db"
LOG_PATH        = Path(__file__).parent / "watering.log"


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  EXCEPTIONS
# ─────────────────────────────────────────────

class WateringError(Exception):
    """Raised when a watering cycle fails."""


# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> None:
    """Create the watering_events table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watering_events (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration_sec   INTEGER  NOT NULL,
                trigger_type   TEXT     NOT NULL,  -- 'scheduled' | 'manual' | 'sensor'
                success        INTEGER  NOT NULL,   -- 1 = ok, 0 = failed
                notes          TEXT
            )
        """)
        conn.commit()


def log_event(
    duration_sec: int,
    trigger_type: str,
    success: bool,
    notes: str = "",
    db_path: Path = DB_PATH,
) -> int:
    """Insert a watering event and return its new row id."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO watering_events (duration_sec, trigger_type, success, notes)
            VALUES (?, ?, ?, ?)
            """,
            (duration_sec, trigger_type, int(success), notes),
        )
        conn.commit()
        return cursor.lastrowid


# ─────────────────────────────────────────────
#  GPIO HELPERS
# ─────────────────────────────────────────────

def _setup_gpio() -> None:
    if not ON_PI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(VALVE_PIN, GPIO.OUT, initial=GPIO.LOW)


def _valve_open() -> None:
    if ON_PI:
        GPIO.output(VALVE_PIN, GPIO.HIGH)
    log.info("Valve OPEN  (pin %d → HIGH)", VALVE_PIN)


def _valve_close() -> None:
    if ON_PI:
        GPIO.output(VALVE_PIN, GPIO.LOW)
    log.info("Valve CLOSE (pin %d → LOW)", VALVE_PIN)


def _cleanup_gpio() -> None:
    if ON_PI:
        GPIO.cleanup(VALVE_PIN)


# ─────────────────────────────────────────────
#  CORE WATERING FUNCTION
# ─────────────────────────────────────────────

def water(
    duration: int = DEFAULT_DURATION,
    trigger_type: str = "scheduled",
    db_path: Path = DB_PATH,
) -> dict:
    """
    Open the sprinkler valve for `duration` seconds, then close it.

    Args:
        duration:     How long (seconds) to keep the valve open.
        trigger_type: Who/what triggered the cycle ('scheduled'|'manual'|'sensor').
        db_path:      Path to the SQLite database file.

    Returns:
        dict with keys: success (bool), duration (int), timestamp (str), event_id (int)

    Raises:
        WateringError: if the valve cannot be controlled.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    init_db(db_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    success = False
    notes = ""

    log.info("── Watering cycle START ── trigger=%s  duration=%ds", trigger_type, duration)

    try:
        _setup_gpio()
        _valve_open()
        time.sleep(duration)
        success = True

    except Exception as exc:
        notes = str(exc)
        log.error("Watering failed: %s", exc)
        raise WateringError(f"Valve control failed: {exc}") from exc

    finally:
        # Always try to close the valve, even if something went wrong
        try:
            _valve_close()
        except Exception as exc:
            log.error("CRITICAL — could not close valve: %s", exc)
            notes += f" | close error: {exc}"
        finally:
            _cleanup_gpio()

        event_id = log_event(
            duration_sec=duration,
            trigger_type=trigger_type,
            success=success,
            notes=notes,
            db_path=db_path,
        )
        log.info(
            "── Watering cycle END   ── success=%s  event_id=%d",
            success, event_id
        )

    return {
        "success": success,
        "duration": duration,
        "timestamp": started_at,
        "event_id": event_id,
        "trigger_type": trigger_type,
    }


# ─────────────────────────────────────────────
#  ENTRY POINT  (called by cron or CLI)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trigger a plant watering cycle.")
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Valve open time in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--trigger",
        default="scheduled",
        choices=["scheduled", "manual", "sensor"],
        help="What triggered this watering (default: scheduled)",
    )
    args = parser.parse_args()

    try:
        result = water(duration=args.duration, trigger_type=args.trigger)
        sys.exit(0)
    except WateringError:
        sys.exit(1)