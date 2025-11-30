"""Metrics and visualization utilities."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Use non-interactive backend for safe PNG saving
plt.switch_backend("Agg")


def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
    figsize: tuple = (20, 16),
    normalize: bool = False
) -> None:
    """
    Save confusion matrix heatmap (raw or normalized).

    Args:
        confusion_matrix: 2D numpy array of confusion matrix values
        labels: Class names
        output_path: Path to save PNG
        figsize: Figure size
        normalize: Whether to plot normalized CM (percentages)
    """

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=labels
    )

    # Normalized CM uses floats; raw uses counts
    cmap = "Blues"
    colorbar = True

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap=cmap,
        colorbar=colorbar,
        values_format=".2f" if normalize else "d"
    )

    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    plt.title(title, fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {output_path}")


def save_confusion_matrices_both(
    confusion_matrix: np.ndarray,
    labels: List[str],
    output_dir: Path
) -> None:
    """
    Convenience function to save BOTH raw and normalized confusion matrices.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        confusion_matrix=confusion_matrix,
        labels=labels,
        output_path=output_dir / "confusion_matrix_raw.png",
        normalize=False
    )

    save_confusion_matrix(
        confusion_matrix=confusion_matrix,
        labels=labels,
        output_path=output_dir / "confusion_matrix_normalized.png",
        normalize=True
    )




