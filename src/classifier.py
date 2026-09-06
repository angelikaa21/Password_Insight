from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import FEATURE_NAMES, extract_features


CLASS_LABELS = {
    0: "Bardzo słabe",
    1: "Słabe",
    2: "Średnie",
    3: "Silne",
    4: "Bardzo silne",
}


def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu klasyfikacyjnego: {path}.")
    return joblib.load(path)


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def predict_strength(model: Any, password: str) -> dict[str, Any]:
    features = extract_features(password)
    frame = pd.DataFrame([features], columns=FEATURE_NAMES)
    predicted = int(model.predict(frame)[0])

    probabilities = {index: 0.0 for index in CLASS_LABELS}
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(frame)[0]
        probabilities.update(
            {int(label): float(probability) for label, probability in zip(model.classes_, values)}
        )

    return {
        "class_id": predicted,
        "class_label": CLASS_LABELS[predicted],
        "class_probability": probabilities[predicted],
        "probabilities": probabilities,
        "features": features,
        "feature_frame": frame,
    }
