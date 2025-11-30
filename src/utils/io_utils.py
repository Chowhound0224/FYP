"""I/O utilities for loading data and saving artifacts."""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR


def _json_default(obj: Any) -> Any:
    """Convert non-serializable objects to JSON-compatible types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    
    # ⛑ Instead of raising an error, fallback to string
    return str(obj)


def ensure_artifacts_dir() -> None:
    """Create artifacts directory if it doesn't exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Safely save JSON, converting unsupported objects to strings.

    Args:
        data: Dictionary to save
        path: Output file path
    """

    def safe_default(obj):
        try:
            return _json_default(obj)
        except TypeError:
            return str(obj)  # fallback for models, pipelines, etc.

    path.write_text(json.dumps(data, indent=2, default=safe_default), encoding="utf-8")


def file_md5(path: Path) -> str:
    """
    Calculate MD5 hash of file.

    Args:
        path: Path to file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_dataset(path: Path, required_cols: tuple, dataset_name: str) -> pd.DataFrame:
    """
    Load dataset and validate required columns exist.

    Args:
        path: Path to CSV file
        required_cols: Tuple of required column names
        dataset_name: Name for error messages

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset not found. Expected file at {path}")

    df = pd.read_csv(path)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{dataset_name} dataset missing required columns: {missing_cols}")

    return df


def profile_dataframe(df: pd.DataFrame, label_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate profiling statistics for a dataframe.

    Args:
        df: DataFrame to profile
        label_col: Optional label column for distribution

    Returns:
        Dictionary with profiling stats
    """
    profile = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_counts": df.isnull().sum().to_dict(),
    }

    if label_col and label_col in df.columns:
        label_counts = df[label_col].value_counts(dropna=False)
        profile["label_distribution"] = label_counts.to_dict()
        profile["label_distribution_normalized"] = (
            (label_counts / label_counts.sum()).round(4).to_dict()
        )

    return profile



