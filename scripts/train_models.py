from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib

# Backend przeznaczony do zapisu wykresów bez interfejsu graficznego.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_NAMES, features_frame  # noqa: E402


SEEDS = (20260902, 20260903, 20260904)
RF_CANDIDATES = (
    {"n_estimators": 160, "max_depth": 20, "min_samples_leaf": 2},
    {"n_estimators": 220, "max_depth": None, "min_samples_leaf": 2},
)
MODEL_JOBS = 2


def read_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if not {"password", "label"}.issubset(data.columns):
        raise ValueError("Zbiór musi zawierać kolumny password i label.")
    data = data.dropna(subset=["password", "label"])
    data["password"] = data["password"].astype(str)
    data["label"] = data["label"].astype(int)
    data = data[data["label"].between(0, 4)]
    conflicts = data.groupby("password")["label"].nunique()
    data = data[~data["password"].isin(conflicts[conflicts > 1].index)]
    return data.drop_duplicates(subset="password").reset_index(drop=True)


def split_data(data: pd.DataFrame, seed: int):
    indices = np.arange(len(data))
    train_idx, temporary_idx = train_test_split(
        indices, test_size=0.30, stratify=data["label"], random_state=seed
    )
    validation_idx, test_idx = train_test_split(
        temporary_idx,
        test_size=0.50,
        stratify=data.iloc[temporary_idx]["label"],
        random_state=seed,
    )
    return train_idx, validation_idx, test_idx


def build_models(seed: int, rf_parameters: dict[str, int | None]):
    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            **rf_parameters, n_jobs=MODEL_JOBS, class_weight="balanced", random_state=seed
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=220,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=MODEL_JOBS,
            tree_method="hist",
            random_state=seed,
        )
    except ImportError:
        print("Pominięto XGBoost: biblioteka nie jest zainstalowana.")
    return models


def tune_random_forest(x_train, y_train, x_validation, y_validation, seed: int):
    rows = []
    best_parameters = None
    best_score = -1.0
    for parameters in RF_CANDIDATES:
        model = RandomForestClassifier(
            **parameters, n_jobs=MODEL_JOBS, class_weight="balanced", random_state=seed
        )
        started = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        score = f1_score(y_validation, model.predict(x_validation), average="macro")
        rows.append({**parameters, "validation_f1_macro": score, "training_seconds": elapsed})
        if score > best_score:
            best_score, best_parameters = score, parameters
    return best_parameters, rows


def model_size_mb(model) -> float:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.joblib"
        joblib.dump(model, path)
        return path.stat().st_size / (1024 * 1024)


def single_latency_ms(model, x_test: pd.DataFrame, repeats: int = 75) -> tuple[float, float]:
    sample = x_test.iloc[[0]]
    model.predict(sample)
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict(sample)
        measurements.append((time.perf_counter() - started) * 1000)
    return float(np.median(measurements)), float(np.percentile(measurements, 95))


def save_confusion(y_true, prediction, name: str, seed: int, output_dir: Path) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, prediction, labels=[0, 1, 2, 3, 4], cmap="RdPu", colorbar=False
    )
    display.ax_.set_title(f"{name} — ziarno {seed}")
    display.ax_.set_xlabel("Klasa przewidziana")
    display.ax_.set_ylabel("Klasa rzeczywista")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / f"confusion_{name}_{seed}.png", dpi=180)
    plt.close(display.figure_)


def evaluate_model(name, model, x_train, y_train, x_test, y_test, seed: int, output_dir: Path):
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(x_test)
    batch_seconds = time.perf_counter() - started
    median_ms, p95_ms = single_latency_ms(model, x_test)
    report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
    (output_dir / f"classification_{name}_{seed}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_confusion(y_test, prediction, name, seed, output_dir)
    return {
        "seed": seed,
        "model": name,
        "accuracy": accuracy_score(y_test, prediction),
        "precision_macro": precision_score(y_test, prediction, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, prediction, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, prediction, average="weighted", zero_division=0),
        "mae_class": mean_absolute_error(y_test, prediction),
        "training_seconds": training_seconds,
        "batch_prediction_seconds": batch_seconds,
        "single_prediction_median_ms": median_ms,
        "single_prediction_p95_ms": p95_ms,
        "model_size_mb": model_size_mb(model),
    }


def evaluate_zxcvbn(passwords: pd.Series, y_test: pd.Series, seed: int, output_dir: Path):
    from zxcvbn import zxcvbn

    predictions = []
    measurements = []
    for password in passwords:
        started = time.perf_counter()
        predictions.append(int(zxcvbn(password)["score"]))
        measurements.append((time.perf_counter() - started) * 1000)
    save_confusion(y_test, predictions, "zxcvbn", seed, output_dir)
    return {
        "seed": seed,
        "model": "zxcvbn",
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "mae_class": mean_absolute_error(y_test, predictions),
        "training_seconds": 0.0,
        "batch_prediction_seconds": sum(measurements) / 1000,
        "single_prediction_median_ms": float(np.median(measurements)),
        "single_prediction_p95_ms": float(np.percentile(measurements, 95)),
        "model_size_mb": 0.0,
    }


def hash_password(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def size_experiment(x_train, y_train, x_validation, y_validation, parameters, seed: int):
    rows = []
    for fraction in (0.10, 0.25, 0.50, 1.00):
        if fraction < 1:
            subset, _ = train_test_split(
                np.arange(len(x_train)), train_size=fraction, stratify=y_train, random_state=seed
            )
            current_x, current_y = x_train.iloc[subset], y_train.iloc[subset]
        else:
            current_x, current_y = x_train, y_train
        model = RandomForestClassifier(
            **parameters, n_jobs=MODEL_JOBS, class_weight="balanced", random_state=seed
        )
        started = time.perf_counter()
        model.fit(current_x, current_y)
        elapsed = time.perf_counter() - started
        prediction = model.predict(x_validation)
        rows.append(
            {
                "seed": seed,
                "training_rows": len(current_x),
                "fraction_of_training_set": fraction,
                "validation_f1_macro": f1_score(y_validation, prediction, average="macro"),
                "training_seconds": elapsed,
            }
        )
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksperyment porównawczy i trening modelu badawczego.")
    parser.add_argument(
        "dataset", nargs="?", type=Path, default=ROOT / "data" / "processed" / "pwlds_sample.csv"
    )
    args = parser.parse_args()

    data = read_dataset(args.dataset)
    print(f"Ekstrakcja {len(data):,} wektorów cech...".replace(",", " "))
    features = features_frame(data["password"])
    labels = data["label"]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    all_results = []
    tuning_rows = []
    size_rows = []
    deployment_payload = None

    for seed in SEEDS:
        print(f"Eksperyment dla ziarna {seed}")
        train_idx, validation_idx, test_idx = split_data(data, seed)
        x_train, y_train = features.iloc[train_idx], labels.iloc[train_idx]
        x_validation, y_validation = features.iloc[validation_idx], labels.iloc[validation_idx]
        x_test, y_test = features.iloc[test_idx], labels.iloc[test_idx]

        rf_parameters, current_tuning = tune_random_forest(
            x_train, y_train, x_validation, y_validation, seed
        )
        tuning_rows.extend({"seed": seed, **row} for row in current_tuning)
        # Eksperyment wpływu liczebności wykorzystuje bazowe ziarno losowe.
        if seed == SEEDS[0]:
            size_rows.extend(
                size_experiment(x_train, y_train, x_validation, y_validation, rf_parameters, seed)
            )
        for name, model in build_models(seed, rf_parameters).items():
            print(f"  trening: {name}")
            all_results.append(
                evaluate_model(name, model, x_train, y_train, x_test, y_test, seed, results_dir)
            )
        print("  ocena: zxcvbn")
        all_results.append(
            evaluate_zxcvbn(data.iloc[test_idx]["password"], y_test, seed, results_dir)
        )

        manifest = pd.DataFrame({"record_sha256": data["password"].map(hash_password), "split": "train"})
        manifest.loc[validation_idx, "split"] = "validation"
        manifest.loc[test_idx, "split"] = "test"
        manifest.to_csv(results_dir / f"split_manifest_{seed}.csv", index=False)

        if seed == SEEDS[0]:
            deployment_payload = (rf_parameters, np.concatenate([train_idx, validation_idx]))

    results = pd.DataFrame(all_results)
    results.to_csv(results_dir / "model_comparison_all_seeds.csv", index=False)
    summary = (
        results.groupby("model")
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            accuracy_mean=("accuracy", "mean"),
            mae_class_mean=("mae_class", "mean"),
            latency_median_ms=("single_prediction_median_ms", "mean"),
            latency_p95_ms=("single_prediction_p95_ms", "mean"),
        )
        .reset_index()
        .sort_values("f1_macro_mean", ascending=False)
    )
    summary.to_csv(results_dir / "model_comparison_summary.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(results_dir / "random_forest_tuning.csv", index=False)
    pd.DataFrame(size_rows).to_csv(results_dir / "data_size_experiment.csv", index=False)

    parameters, deployment_indices = deployment_payload
    deployment_model = RandomForestClassifier(
        **parameters, n_jobs=MODEL_JOBS, class_weight="balanced", random_state=SEEDS[0]
    )
    deployment_model.fit(features.iloc[deployment_indices], labels.iloc[deployment_indices])
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(deployment_model, models_dir / "password_strength_model.joblib")
    metadata = {
        "model_type": "RandomForestClassifier",
        "purpose": "RESEARCH_MODEL",
        "dataset": "PWLDS",
        "dataset_file_sha256": file_sha256(args.dataset),
        "sample_count_after_cleaning": len(data),
        "training_rows_for_deployment": len(deployment_indices),
        "test_rows_reserved": int(round(len(data) * 0.15)),
        "seeds": list(SEEDS),
        "feature_names": FEATURE_NAMES,
        "parameters": parameters,
        "comparison_best_model": str(summary.iloc[0]["model"]),
        "deployed_model": "random_forest",
        "deployment_reason": "Główny model badany i model obsługiwany przez TreeSHAP",
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nPodsumowanie:")
    print(summary.to_string(index=False))
    print("\nModel badawczy zapisano w katalogu models.")


if __name__ == "__main__":
    main()
