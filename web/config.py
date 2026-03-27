import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATABASE_PATH = os.environ.get(
    "GARDEFENDER_DB", str(BASE_DIR / "plants.db")
)
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
PASSWORD = os.environ.get("GARDEFENDER_PASSWORD", "change-me")
PORT = 5000
