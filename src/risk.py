from __future__ import annotations

import math
from typing import Any


RISK_COLORS = {
    "Niskie": "#35A36F",
    "Podwyższone": "#43B5A0",
    "Średnie": "#E9B949",
    "Wysokie": "#F08A5D",
    "Krytyczne": "#EF6A6A",
}


def _risk_label(score: float) -> str:
    if score < 20:
        return "Niskie"
    if score < 40:
        return "Podwyższone"
    if score < 60:
        return "Średnie"
    if score < 80:
        return "Wysokie"
    return "Krytyczne"


def calculate_risk(strength_class: int, breach: dict[str, Any]) -> dict[str, Any]:
    """Jawna agregacja: wynik naruszenia może tylko podnieść ryzyko."""
    structural = float(100 - 25 * max(0, min(4, strength_class)))
    breach_component: float | None
    if breach.get("status") != "checked":
        breach_component = None
        total = structural
        complete = False
    elif breach.get("found"):
        count = max(1, int(breach.get("count") or 1))
        breach_component = min(100.0, 60.0 + 10.0 * math.log10(count))
        total = max(structural, breach_component)
        complete = True
    else:
        breach_component = 0.0
        total = structural
        complete = True

    label = _risk_label(total)
    return {
        "score": round(total, 1),
        "label": label,
        "color": RISK_COLORS[label],
        "structural_component": round(structural, 1),
        "breach_component": None if breach_component is None else round(breach_component, 1),
        "complete": complete,
    }
