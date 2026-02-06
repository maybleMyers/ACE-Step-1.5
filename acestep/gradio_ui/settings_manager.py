"""
Settings Manager Module
Handles saving and loading user preferences to/from JSON file
"""
import json
import os
from typing import Dict, Any, Optional
from loguru import logger

# Project root and settings file path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "acestep_ui_settings.json")


def save_settings(settings: Dict[str, Any]) -> str:
    """
    Save settings dictionary to JSON file.

    Args:
        settings: Dictionary of settings to save

    Returns:
        Status message string
    """
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Settings saved to {SETTINGS_FILE}")
        return f"✅ Settings saved successfully to {SETTINGS_FILE}"
    except Exception as e:
        error_msg = f"❌ Failed to save settings: {str(e)}"
        logger.error(error_msg)
        return error_msg


def load_settings() -> Dict[str, Any]:
    """
    Load settings from JSON file.

    Returns:
        Dictionary of settings, or empty dict if file doesn't exist or is corrupted
    """
    if not os.path.exists(SETTINGS_FILE):
        logger.info("No settings file found, using defaults")
        return {}

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        logger.info(f"✅ Settings loaded from {SETTINGS_FILE}")
        return settings
    except Exception as e:
        logger.error(f"❌ Failed to load settings: {str(e)}, using defaults")
        return {}


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a single setting value.

    Args:
        key: Setting key to retrieve
        default: Default value if key not found

    Returns:
        Setting value or default
    """
    settings = load_settings()
    return settings.get(key, default)
