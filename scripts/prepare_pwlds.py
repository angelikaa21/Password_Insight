from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://huggingface.co/datasets/InfinitodeLTD/PWLDS/resolve/main/{filename}?download=true"
CLASS_FILES = {
    0: "pwlds_very_weak.csv",
    1: "pwlds_weak.csv",
    2: "pwlds_average.csv",
    3: "pwlds_strong.csv",
    4: "pwlds_very_strong.csv",
}


def download_file(filename: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 10_000:
        print(f"Pomijam pobieranie istniejącego pliku: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Pobieranie {filename}...")
    with requests.get(BASE_URL.format(filename=filename), stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary.replace(destination)


def normalize_chunk(chunk: pd.DataFrame, label: int) -> pd.DataFrame:
    normalized = {str(column).strip().lower(): column for column in chunk.columns}
    password_column = normalized.get("password")
    if password_column is None:
        raise ValueError("W pliku PWLDS nie znaleziono kolumny Password.")
    result = pd.DataFrame({"password": chunk[password_column]})
    result = result.dropna(subset=["password"])
    result["password"] = result["password"].astype(str)
    result = result[result["password"].str.len().between(1, 256)]
    result["label"] = label
    return result


def priority_sample(path: Path, label: int, target: int, seed: int) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed + label)
    reservoir = pd.DataFrame(columns=["password", "label", "_priority"])
    observed = 0
    for chunk in pd.read_csv(path, chunksize=250_000, dtype=str, keep_default_na=False):
        clean = normalize_chunk(chunk, label)
        observed += len(clean)
        clean["_priority"] = rng.random(len(clean))
        candidate = pd.concat([reservoir, clean], ignore_index=True)
        reservoir = candidate.nsmallest(min(target, len(candidate)), "_priority")
        print(f"  klasa {label}: przejrzano {observed:,} rekordów".replace(",", " "), end="\r")
    print()
    return reservoir.drop(columns="_priority"), observed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pobranie i przygotowanie stratyfikowanej próbki PWLDS.")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "pwlds_sample.csv")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 5_000:
        raise ValueError("Próbka powinna zawierać co najmniej 5000 rekordów.")

    raw_dir = ROOT / "data" / "raw"
    target_per_class = args.sample_size // 5
    samples: list[pd.DataFrame] = []
    observed_counts: dict[str, int] = {}
    for label, filename in CLASS_FILES.items():
        path = raw_dir / filename
        if not args.skip_download:
            download_file(filename, path)
        if not path.exists():
            raise FileNotFoundError(f"Brak pliku {path}. Usuń --skip-download albo pobierz dane ręcznie.")
        sample, observed = priority_sample(path, label, target_per_class, args.seed)
        samples.append(sample)
        observed_counts[str(label)] = observed

    data = pd.concat(samples, ignore_index=True)
    conflicts = data.groupby("password")["label"].nunique()
    conflicting_passwords = set(conflicts[conflicts > 1].index)
    if conflicting_passwords:
        data = data[~data["password"].isin(conflicting_passwords)]
    data = data.drop_duplicates(subset=["password", "label"])
    data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    summary = {
        "dataset": "Password Weakness and Level Dataset (PWLDS)",
        "source": "https://huggingface.co/datasets/InfinitodeLTD/PWLDS",
        "license": "CC BY 4.0",
        "synthetic": True,
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "sample_size_after_cleaning": len(data),
        "class_counts": {str(key): int(value) for key, value in data["label"].value_counts().sort_index().items()},
        "observed_source_rows": observed_counts,
        "removed_conflicting_passwords": len(conflicting_passwords),
        "sample_sha256": sha256(args.output),
    }
    summary_path = args.output.with_name("dataset_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano próbkę: {args.output}")
    print(json.dumps(summary["class_counts"], indent=2))


if __name__ == "__main__":
    main()

