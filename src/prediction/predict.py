"""
Prediction Pipeline for Resume Classification

This module loads the trained model, TF-IDF vectorizer, scaler, and
label encoder, builds the combined SBERT + TF-IDF + Custom features,
and predicts the resume category.

Used by: app.py, batch_predict.py, API endpoints.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from src.cleaning.text_cleaning import clean_text
from src.features.build_features import build_combined_features
from src.config import (
    BASE_DIR,
)

# =============================================================================
# Artifact paths (latest models saved by train_improved.py)
# =============================================================================

MODEL_PATH = BASE_DIR / "improved_classifier.pkl"
TFIDF_PATH = BASE_DIR / "improved_tfidf.pkl"
SCALER_PATH = BASE_DIR / "improved_scaler.pkl"
ENCODER_PATH = BASE_DIR / "improved_label_encoder.pkl"


# =============================================================================
# Artifact Loader
# =============================================================================
def load_artifacts() -> Tuple[Any, Any, Any, Any]:
    """
    Load the latest saved classifier, TF-IDF, scaler, and label encoder.

    Returns:
        (model, tfidf, scaler, label_encoder)
    """

    missing = []

    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not TFIDF_PATH.exists():
        missing.append(str(TFIDF_PATH))
    if not SCALER_PATH.exists():
        missing.append(str(SCALER_PATH))
    if not ENCODER_PATH.exists():
        missing.append(str(ENCODER_PATH))

    if missing:
        raise FileNotFoundError(
            "❌ Missing artifacts. Train the improved model first using train_improved.py.\n"
            "Missing files:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    return model, tfidf, scaler, label_encoder


# =============================================================================
# Prediction Function
# =============================================================================
def predict_resume_category(resume_text: str) -> Dict[str, Any]:
    """
    Predict the job category for a single resume.

    Args:
        resume_text: Raw resume text

    Returns:
        Dictionary containing:
            - predicted_category
            - confidence_score
            - probabilities (dict per class)
            - cleaned_text
    """

    # ----------------------------------------------------------------------
    # 1. Load classifier + transformers
    # ----------------------------------------------------------------------
    model, tfidf, scaler, label_encoder = load_artifacts()

    # ----------------------------------------------------------------------
    # 2. Clean text using your universal cleaner
    # ----------------------------------------------------------------------
    cleaned_text = clean_text(resume_text)

    # ----------------------------------------------------------------------
    # 3. Build combined feature vector
    #    (SBERT + TFIDF + Custom Features)
    # ----------------------------------------------------------------------
    features, _, _ = build_combined_features(
    raw_texts=np.array([resume_text]),
    tfidf=tfidf,
    scaler=scaler,
    fit=False
    )


    # ----------------------------------------------------------------------
    # 4. Predict
    # ----------------------------------------------------------------------
    y_pred_numeric = model.predict(features)[0]

    # If model does not support predict_proba (rare case)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
    else:
        probabilities = np.zeros(len(label_encoder.classes_))
        probabilities[y_pred_numeric] = 1.0
        confidence = 1.0

    predicted_category = label_encoder.inverse_transform([y_pred_numeric])[0]

    # ----------------------------------------------------------------------
    # 5. Return structured response
    # ----------------------------------------------------------------------
    return {
        "predicted_category": predicted_category,
        "confidence_score": confidence,
        "probabilities": {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, probabilities)
        },
        "cleaned_text": cleaned_text
    }


# =============================================================================
# Optional: CLI Test
# =============================================================================
if __name__ == "__main__":
    sample = input("Paste resume text below:\n\n")
    result = predict_resume_category(sample)

    print("\nPrediction Result")
    print("=" * 70)
    print(f"Predicted Category : {result['predicted_category']}")
    print(f"Confidence Score   : {result['confidence_score']:.4f}")
    print("\nTop Probabilities:")
    for label, prob in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {label:30s} — {prob:.4f}")



