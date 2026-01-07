"""
Quick test script to verify model loading and feature extraction.
Run: python test_model_loading.py
"""

import joblib
import numpy as np
from src.features.build_features import build_combined_features
from src.config import TOTAL_FEATURES

print("=" * 60)
print("MODEL LOADING TEST")
print("=" * 60)

# Test 1: Load all model artifacts
print("\n[1/4] Loading model artifacts...")
try:
    classifier = joblib.load("improved_classifier.pkl")
    tfidf = joblib.load("improved_tfidf.pkl")
    scaler = joblib.load("improved_scaler.pkl")
    label_encoder = joblib.load("improved_label_encoder.pkl")
    print(f"[OK] All artifacts loaded successfully")
    print(f"     - Model type: {type(classifier).__name__}")
    print(f"     - Expected features: {classifier.n_features_in_}")
    print(f"     - Categories: {len(label_encoder.classes_)}")
except Exception as e:
    print(f"[ERROR] Failed to load artifacts: {e}")
    exit(1)

# Test 2: Verify feature dimensions
print("\n[2/4] Verifying feature dimensions...")
if classifier.n_features_in_ == TOTAL_FEATURES:
    print(f"[OK] Model expects {TOTAL_FEATURES} features (matches config)")
else:
    print(f"[ERROR] Feature mismatch!")
    print(f"     - Model expects: {classifier.n_features_in_}")
    print(f"     - Config defines: {TOTAL_FEATURES}")
    exit(1)

# Test 3: Test feature extraction
print("\n[3/4] Testing feature extraction...")
test_resume = """
Senior Software Engineer with 5+ years of experience in Python, Java, and cloud technologies.
Bachelor's degree in Computer Science. Proficient in AWS, Docker, and Kubernetes.
Strong leadership and communication skills. AWS Certified Solutions Architect.
"""

try:
    X_test, _, _ = build_combined_features(
        raw_texts=np.array([test_resume]),
        tfidf=tfidf,
        scaler=scaler,
        fit=False
    )
    print(f"[OK] Feature extraction successful")
    print(f"     - Extracted features: {X_test.shape[1]}")
    print(f"     - Feature vector shape: {X_test.shape}")
except Exception as e:
    print(f"[ERROR] Feature extraction failed: {e}")
    exit(1)

# Test 4: Test prediction
print("\n[4/4] Testing prediction...")
try:
    prediction = classifier.predict(X_test)
    category = label_encoder.inverse_transform(prediction)[0]
    print(f"[OK] Prediction successful")
    print(f"     - Predicted category: {category}")
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nYour model is working correctly and ready to use.")
print("You can now run: streamlit run app.py")
