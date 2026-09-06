from __future__ import annotations

from typing import Any

from .breach_check import check_pwned_password
from .classifier import predict_strength
from .explanations import build_recommendations, shap_explanation
from .risk import calculate_risk


def analyze_password(model: Any, password: str, check_breach: bool = True) -> dict[str, Any]:
    strength = predict_strength(model, password)
    if check_breach:
        breach = check_pwned_password(password).to_dict()
    else:
        breach = {
            "status": "skipped",
            "found": None,
            "count": None,
            "message": "Weryfikacja HIBP została pominięta.",
        }

    risk = calculate_risk(strength["class_id"], breach)
    try:
        explanation = shap_explanation(model, strength["feature_frame"], strength["class_id"])
        explanation_error = None
    except Exception:
        explanation = []
        explanation_error = "Nie udało się przygotować lokalnego wyjaśnienia SHAP."

    result = {
        "strength": {key: value for key, value in strength.items() if key != "feature_frame"},
        "breach": breach,
        "risk": risk,
        "explanation": explanation,
        "explanation_error": explanation_error,
        "recommendations": build_recommendations(strength["features"], breach.get("found")),
    }
    return result

