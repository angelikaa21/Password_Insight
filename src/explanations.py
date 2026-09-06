from __future__ import annotations

from typing import Any

import numpy as np

from .features import FEATURE_LABELS, FEATURE_NAMES


def shap_explanation(model: Any, frame: Any, class_id: int) -> list[dict[str, float | str]]:
    """Oblicza lokalne wkłady SHAP dla przewidywanej klasy."""
    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(frame)
    values = np.asarray(explanation.values)

    if values.ndim == 3:
        class_positions = {int(value): index for index, value in enumerate(model.classes_)}
        contributions = values[0, :, class_positions[class_id]]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError("Nieobsługiwany format wartości SHAP.")

    items = [
        {
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "value": float(frame.iloc[0][feature]),
            "contribution": float(contribution),
        }
        for feature, contribution in zip(FEATURE_NAMES, contributions)
    ]
    return sorted(items, key=lambda item: abs(float(item["contribution"])), reverse=True)


def build_recommendations(features: dict[str, float], breached: bool | None) -> list[str]:
    recommendations: list[str] = []
    if breached:
        recommendations.append("Nie używaj tego hasła – wystąpiło w znanych naruszeniach.")
    if features["length"] < 15:
        recommendations.append("Zwiększ długość hasła do co najmniej 15 znaków.")
    if features["common_pattern_count"] > 0:
        recommendations.append("Usuń popularne słowo, rok lub przewidywalny fragment.")
    if features["keyboard_sequence_length"] >= 3 or features["sequence_length"] >= 3:
        recommendations.append("Zastąp prostą sekwencję mniej przewidywalnym fragmentem.")
    if features["max_char_run"] >= 3 or features["repeated_fragment_count"] > 0:
        recommendations.append("Ogranicz powtórzenia znaków i krótkich fragmentów.")
    if features["unique_ratio"] < 0.65 and features["length"] > 0:
        recommendations.append("Zwiększ różnorodność znaków zamiast powtarzać te same znaki.")
    if not recommendations:
        recommendations.append("Nie wykryto prostego sposobu poprawy struktury hasła.")
    return recommendations[:4]

