"""
Train improved resume classifier using SBERT + TF-IDF + Custom Features
with Optuna optimization.
"""

import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RAW_RESUME_PATH,
    REQUIRED_RESUME_COLS,
    ARTIFACTS_DIR,
    BASE_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils import ensure_artifacts_dir, save_json, load_dataset
from src.cleaning.text_cleaning import clean_text
from src.features.build_features import build_combined_features
from src.models.lightgbm_trainer import train_with_optuna
from src.evaluation.evaluate import evaluate_model
from src.utils.metrics import save_confusion_matrices_both


def main():
    """Main training pipeline."""
    ensure_artifacts_dir()

    print("=" * 80)
    print("IMPROVED RESUME CLASSIFIER TRAINING")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------------------
    print("\nLoading dataset...")
    resume_df = load_dataset(RAW_RESUME_PATH, REQUIRED_RESUME_COLS, "Resume")
    print(f"[OK] Loaded {len(resume_df)} resumes")

    # ----------------------------------------------------------------------
    # 2. Remove empty entries / duplicates
    # ----------------------------------------------------------------------
    print("\nCleaning raw text...")
    resume_df["cleaned_text"] = resume_df["Resume_str"].fillna("").apply(clean_text)
    resume_df = resume_df[resume_df["cleaned_text"].str.strip() != ""]
    resume_df = resume_df.drop_duplicates(subset=["cleaned_text"])
    print(f"[OK] Cleaned rows remaining: {len(resume_df)}")

    # ----------------------------------------------------------------------
    # 2b. Filter out low-sample categories
    # ----------------------------------------------------------------------
    EXCLUDED_CATEGORIES = [
        'BPO',                  # 22 samples
        'AUTOMOBILE',           # 36 samples
        'AGRICULTURE',          # 63 samples
        'DIGITAL-MEDIA',        # 96 samples
        'APPAREL',              # 97 samples
        'ARTS',                 # 103 samples (user requested)
        'BANKING'               # 115 samples (user requested)
    ]

    initial_count = len(resume_df)
    resume_df = resume_df[~resume_df["Category"].isin(EXCLUDED_CATEGORIES)]
    dropped_count = initial_count - len(resume_df)

    print(f"[OK] Excluded {len(EXCLUDED_CATEGORIES)} categories: {', '.join(EXCLUDED_CATEGORIES)}")
    print(f"[OK] Dropped {dropped_count} resumes, kept {len(resume_df)} resumes")
    print(f"[OK] Remaining categories: {len(resume_df['Category'].unique())}")

    # Extract text and labels
    X_text = resume_df["Resume_str"].values     # raw for SBERT + custom
    y_raw = resume_df["Category"].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"[OK] Classes found: {len(label_encoder.classes_)} categories")

    # ----------------------------------------------------------------------
    # 3. Build features (SBERT + TF-IDF + Custom)
    # ----------------------------------------------------------------------
    print("\nBuilding feature vectors...")
    X_combined, tfidf, scaler = build_combined_features(
        raw_texts=X_text,
        fit=True
    )

    # ----------------------------------------------------------------------
    # 4. Train-test split
    # ----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[OK] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ----------------------------------------------------------------------
    # 5. Hyperparameter optimization + model training
    # ----------------------------------------------------------------------
    best_model, best_params, best_score = train_with_optuna(X_train, y_train)

    # ----------------------------------------------------------------------
    # 6. Evaluation
    # ----------------------------------------------------------------------
    evaluation = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        model_type="xgboost"  # Fixed: XGBoost-only trainer
    )

    # ----------------------------------------------------------------------
    # 7. Save all artifacts
    # ----------------------------------------------------------------------
    print("\nSaving artifacts...")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save timestamped
    model_path = ARTIFACTS_DIR / f"improved_model_{timestamp}.pkl"
    tfidf_path = ARTIFACTS_DIR / f"improved_tfidf_{timestamp}.pkl"
    scaler_path = ARTIFACTS_DIR / f"improved_scaler_{timestamp}.pkl"
    encoder_path = ARTIFACTS_DIR / f"improved_label_encoder_{timestamp}.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    # Save latest (for app.py)
    joblib.dump(best_model, BASE_DIR / "improved_classifier.pkl")
    joblib.dump(tfidf, BASE_DIR / "improved_tfidf.pkl")
    joblib.dump(scaler, BASE_DIR / "improved_scaler.pkl")
    joblib.dump(label_encoder, BASE_DIR / "improved_label_encoder.pkl")

    print(f"[OK] Saved model: {model_path}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "best_params": best_params,
        "best_cv_score": float(best_score),
        "test_accuracy": evaluation["accuracy"],
        "test_f1_weighted": evaluation["f1_weighted"],
        "test_f1_macro": evaluation["f1_macro"],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_combined.shape[1],
        "classes": label_encoder.classes_.tolist(),
    }
    save_json(metadata, ARTIFACTS_DIR / f"metadata_{timestamp}.json")

    # Save confusion matrices (raw + normalized)
    save_confusion_matrices_both(
        confusion_matrix=np.array(evaluation["confusion_matrix"]),
        labels=evaluation["confusion_matrix_labels"],
        output_dir=ARTIFACTS_DIR / f"confusion_matrices_{timestamp}"
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE — MODEL READY FOR app.py !")
    print("=" * 80)
    print(f"Final Accuracy: {evaluation['accuracy']*100:.2f}%")
    print(f"Final F1-Weighted: {evaluation['f1_weighted']*100:.2f}%")
    print(f"Final F1-Macro: {evaluation['f1_macro']*100:.2f}%")


if __name__ == "__main__":
    main()




