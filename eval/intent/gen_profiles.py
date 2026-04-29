"""Deterministic user profile generator for intent eval.

Usage (standalone):
    python eval/intent/gen_profiles.py --seed 42 --n 10

Usage (as module):
    from eval.intent.gen_profiles import make_profile
    profile = make_profile(seed=42, index=0)

Profile fields match src/server/models.py:UserProfile.
"""

import math
import random
import json
import argparse
from typing import Optional

_FITNESS_GOALS = [
    "weight loss",
    "muscle gain",
    "maintain weight",
    "improve endurance",
    "reduce sugar intake",
    "eat healthier",
]
_ALLERGIES = [None, None, None, "peanuts", "lactose", "gluten", "shellfish"]
_DIET_PREFS = [None, None, None, "vegetarian", "low-carb", "high-protein"]
_HEALTH_CONDS = [None, None, None, None, "diabetes type 2", "high blood pressure"]
_GENDERS = ["male", "female"]


def _norm(rng: random.Random, mu: float, sd: float, lo: float, hi: float) -> float:
    """Box-Muller normal sample clipped to [lo, hi]."""
    while True:
        u1, u2 = rng.random(), rng.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        v = mu + sd * z
        if lo <= v <= hi:
            return v


def make_profile(seed: int, index: int = 0) -> dict:
    """Generate a single user profile.

    Args:
        seed:  Base random seed (controls the cohort).
        index: Sample index within the cohort; combined with seed so each
               sample in a run gets a distinct but reproducible profile.
    """
    rng = random.Random(seed * 10_000 + index)

    gender = rng.choice(_GENDERS)
    age = int(_norm(rng, 30, 8, 18, 70))
    if gender == "male":
        height_cm = round(_norm(rng, 175, 8, 155, 200), 1)
    else:
        height_cm = round(_norm(rng, 163, 7, 148, 185), 1)
    bmi = _norm(rng, 22.5, 3.0, 16.0, 32.0)
    weight_kg = round(bmi * (height_cm / 100) ** 2, 1)

    profile: dict = {
        "age": age,
        "gender": gender,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "fitness_goals": rng.choice(_FITNESS_GOALS),
    }

    allergy: Optional[str] = rng.choice(_ALLERGIES)
    if allergy:
        profile["allergies"] = allergy

    diet_pref: Optional[str] = rng.choice(_DIET_PREFS)
    if diet_pref:
        profile["dietary_preferences"] = diet_pref

    health_cond: Optional[str] = rng.choice(_HEALTH_CONDS)
    if health_cond:
        profile["health_conditions"] = health_cond

    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview generated profiles.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    for i in range(args.n):
        p = make_profile(seed=args.seed, index=i)
        print(f"[{i:02d}]", json.dumps(p, ensure_ascii=False))
