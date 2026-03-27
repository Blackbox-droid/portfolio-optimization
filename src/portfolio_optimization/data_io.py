from pathlib import Path

import pandas as pd

from .config import CANONICAL_FILES, FIGURES_DIR, TABLES_DIR


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def read_timeseries_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_asset_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_timeseries_csv(CANONICAL_FILES["asset_train"]),
        read_timeseries_csv(CANONICAL_FILES["asset_test"]),
    )


def load_global_macro_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_timeseries_csv(CANONICAL_FILES["macro_global_train"]),
        read_timeseries_csv(CANONICAL_FILES["macro_global_test"]),
    )


def load_india_macro_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_timeseries_csv(CANONICAL_FILES["macro_india_train"]),
        read_timeseries_csv(CANONICAL_FILES["macro_india_test"]),
    )


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

