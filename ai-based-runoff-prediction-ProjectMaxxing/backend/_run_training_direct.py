from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_training_backend import auto_train_best_model


def _load_dataset(project_root: Path) -> pd.DataFrame:
    dataset_dir = project_root / "datasets"
    candidates = sorted(dataset_dir.glob("*.xlsx")) + sorted(dataset_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No dataset found in {dataset_dir}")

    dataset_path = candidates[0]
    if dataset_path.suffix.lower() == ".xlsx":
        excel_file = pd.ExcelFile(dataset_path)
        sheet_name = "Sheet1" if "Sheet1" in excel_file.sheet_names else excel_file.sheet_names[0]
        df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(dataset_path)

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.sort_values("DATE").reset_index(drop=True)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    backend_dir = Path(__file__).resolve().parent

    df = _load_dataset(project_root)
    target_col = "Discharge" if "Discharge" in df.columns else "Discharge (CUMEC)"
    feature_cols = [
        col
        for col in df.columns
        if col not in {"DATE", "Target_t_plus_3", "Discharge", "Discharge (CUMEC)"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    result = auto_train_best_model(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_end_year=2000,
        test_start_year=2006,
        search_type="randomized",
        n_iter=50,
        n_jobs=-1,
        registry_path=(backend_dir / "model_registry.json").as_posix(),
        models_dir=(backend_dir / "models").as_posix(),
    )

    print("Training completed.")
    print("Best model:", result["best_model"])
    print("Artifact:", result["artifact_path"])


if __name__ == "__main__":
    main()
