from pathlib import Path

import pandas as pd

from .config import CANONICAL_FILES, DATASET_COLUMNS, FIGURES_DIR, TABLES_DIR


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def read_timeseries_csv(path: Path, required_columns: list[str], dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    validate_columns(df, required_columns, dataset_name)
    df = df.set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_asset_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_timeseries_csv(CANONICAL_FILES["asset_train"], DATASET_COLUMNS["asset_prices"], "asset_train"),
        read_timeseries_csv(CANONICAL_FILES["asset_test"], DATASET_COLUMNS["asset_prices"], "asset_test"),
    )


def load_global_macro_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_timeseries_csv(CANONICAL_FILES["macro_global_train"], DATASET_COLUMNS["macro_global"], "macro_global_train"),
        read_timeseries_csv(CANONICAL_FILES["macro_global_test"], DATASET_COLUMNS["macro_global"], "macro_global_test"),
    )


def load_india_macro_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = read_timeseries_csv(
        CANONICAL_FILES["macro_india_train"],
        DATASET_COLUMNS["macro_india_train"],
        "macro_india_train",
    )
    test_df = read_timeseries_csv(
        CANONICAL_FILES["macro_india_test"],
        DATASET_COLUMNS["macro_india_test_required"],
        "macro_india_test",
    )
    train_columns = [col for col in DATASET_COLUMNS["macro_india_train"] if col != "Date"]
    return train_df[train_columns], test_df.reindex(columns=train_columns)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
