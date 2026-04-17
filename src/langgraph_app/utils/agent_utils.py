"""Shared utility functions for agents and the Supervisor.

Consolidates duplicated profile-building, TDEE estimation, and meal-time
detection logic that was previously copied across 5+ agent files.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional


def build_profile_context(user_profile: Optional[Dict[str, Any]]) -> str:
    """Build a human-readable profile context string for injection into system prompts.

    Returns an empty string when the profile is None or empty.
    """
    if not user_profile:
        return ""
    lines = [
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in user_profile.items()
        if v
    ]
    if not lines:
        return ""
    return "\n\nUser Profile & Health Information:\n" + "\n".join(lines)


def calculate_tdee(user_profile: Optional[Dict[str, Any]]) -> Optional[int]:
    """Estimate TDEE via Mifflin-St Jeor + PAL.

    Returns None if the profile is missing required fields (weight, height, age).
    """
    if not user_profile:
        return None
    try:
        weight = float(user_profile.get("weight_kg") or 0)
        height = float(user_profile.get("height_cm") or 0)
        age = float(user_profile.get("age") or 0)
    except (ValueError, TypeError):
        return None

    if not (weight and height and age):
        return None

    gender = (user_profile.get("gender") or "").lower()
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 78  # average

    goals = (user_profile.get("fitness_goals") or "").lower()
    pal = 1.5 if any(w in goals for w in ("high intensity", "active", "athlete")) else 1.2
    return round(bmr * pal)


def detect_meal_time(tz_name: Optional[str] = None) -> str:
    """Detect the current meal time period based on the user's timezone.

    Returns one of: "breakfast time", "lunch time", "dinner time", "not meal time".
    """
    _TZ_UTC8 = timezone(timedelta(hours=8))
    user_tz = _TZ_UTC8
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            user_tz = ZoneInfo(tz_name)
        except Exception:
            pass

    now_local = datetime.now(user_tz)
    hour = now_local.hour
    minute = now_local.minute
    t = hour * 60 + minute  # minutes since midnight

    if 7 * 60 <= t <= 9 * 60 + 30:
        return "breakfast time"
    elif 11 * 60 + 30 <= t <= 13 * 60 + 30:
        return "lunch time"
    elif 17 * 60 + 30 <= t <= 19 * 60 + 30:
        return "dinner time"
    else:
        return "not meal time"


def build_daily_cal_ref(user_profile: Optional[Dict[str, Any]]) -> str:
    """Build a daily calorie reference string."""
    tdee = calculate_tdee(user_profile)
    if tdee:
        return f"~{tdee} kcal (your estimated daily needs)"
    return "~2000 kcal (average adult estimate)"
