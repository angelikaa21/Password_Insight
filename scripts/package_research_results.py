from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "WYNIKI_BADAWCZE.zip"


def main() -> None:
    metadata_path = ROOT / "models" / "model_metadata.json"
    required_results = [
        ROOT / "results" / "model_comparison_all_seeds.csv",
        ROOT / "results" / "model_comparison_summary.csv",
        ROOT / "results" / "random_forest_tuning.csv",
        ROOT / "results" / "data_size_experiment.csv",
        *(ROOT / "results" / f"split_manifest_{seed}.csv" for seed in (20260902, 20260903, 20260904)),
    ]
    if not metadata_path.is_file():
        raise RuntimeError("Brak metadanych modelu. Trening nie zostal zakonczony.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("purpose") != "RESEARCH_MODEL":
        raise RuntimeError("Model badawczy nie zostal zapisany. Archiwum nie zostanie utworzone.")
    missing = [path.name for path in required_results if not path.is_file()]
    if missing:
        raise RuntimeError(f"Brakuje wynikow eksperymentu: {', '.join(missing)}")

    candidates = [ROOT / "models" / "password_strength_model.joblib", ROOT / "models" / "model_metadata.json"]
    candidates.extend(sorted((ROOT / "results").glob("*")))
    candidates.append(ROOT / "data" / "processed" / "dataset_summary.json")
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for path in candidates:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    print(f"Utworzono: {OUTPUT}")


if __name__ == "__main__":
    main()
