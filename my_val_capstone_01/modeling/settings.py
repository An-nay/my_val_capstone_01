"""Dataset-derived settings (models/notify_settings.json), built by modeling/build_notify_settings.py."""

import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "models" / "notify_settings.json"
with open(SETTINGS_PATH) as _handle:
    SETTINGS = json.load(_handle)
