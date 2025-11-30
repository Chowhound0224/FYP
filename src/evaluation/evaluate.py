"""Model evaluation and metrics calculation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from typing import Any, Dict


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model_type: str
) -> Dict[str, Any]:
    """
    Evaluate a trained model using multiple metrics.

    Args:
        model: Trained classifier
        X_test: Test feature matrix
        y_test: Encoded test labels
        label_encoder: Encoder for decoding labels
        model_type: Model name (e.g., "xgboost", "logistic", "random_forest")

    Returns:
        Dictionary containing evaluation results
    """

    # ----------------------------------------------------------------------
    # 1. Predict
    # ----------------------------------------------------------------------
    y_pred = model.predict(X_test)

    # ----------------------------------------------------------------------
    # 2. Compute metrics
    # ----------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    # Use correct label order
    class_indices = np.arange(len(label_encoder.classes_))

    conf_matrix = confusion_matrix(
        y_test, y_pred, labels=class_indices
    )

    # ----------------------------------------------------------------------
    # 3. Print summary
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"MODEL PERFORMANCE SUMMARY — {model_type.upper()}")
    print("=" * 80)
    print(f"[OK] Accuracy:        {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"[OK] F1 (Weighted):   {f1_weighted:.4f}")
    print(f"[OK] F1 (Macro):      {f1_macro:.4f}")
    print(f"[OK] Model Type:      {model_type}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    # ----------------------------------------------------------------------
    # 4. Return comprehensive dictionary
    # ----------------------------------------------------------------------
    return {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "model_type": model_type,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_matrix_labels": label_encoder.classes_.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist()
    }



