"""Build combined feature matrix from SBERT + TF-IDF + custom features."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import SBERT_MODEL_NAME, TFIDF_CONFIG
from src.cleaning.text_cleaning import clean_text
from .custom_features import extract_custom_features

# Global SBERT model cache
_sbert_model: Optional[SentenceTransformer] = None


def load_sbert_model(model_name: str = SBERT_MODEL_NAME) -> SentenceTransformer:
    """Load a SentenceTransformer model only once."""
    global _sbert_model
    if _sbert_model is None:
        print(f"Loading SBERT model: {model_name}...")
        _sbert_model = SentenceTransformer(model_name)
        print("[OK] SBERT model loaded.")
    return _sbert_model


def build_combined_features(
    raw_texts: np.ndarray,
    tfidf: Optional[TfidfVectorizer] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, TfidfVectorizer, StandardScaler]:
    """
    Build combined features using:
    - SBERT (semantic)
    - TF-IDF (keyword-based)
    - Custom handcrafted features

    Args:
        raw_texts: np.ndarray of raw resume/job text
        tfidf: optional pre-fitted TF-IDF vectorizer
        scaler: optional pre-fitted StandardScaler
        fit: whether to fit (training) or transform (inference)

    Returns:
        X_combined: numpy feature matrix
        tfidf: trained or passed-in vectorizer
        scaler: trained or passed-in scaler
    """

    # ----------------------------------------------------------------------
    # 1. Clean text for TF-IDF using your NEW single universal cleaner
    # ----------------------------------------------------------------------
    print("\nCleaning text for TF-IDF...")
    cleaned_texts = np.array([clean_text(t) for t in raw_texts])

    # ----------------------------------------------------------------------
    # 2. SBERT embeddings
    # ----------------------------------------------------------------------
    print("Encoding text with SBERT...")
    sbert = load_sbert_model()
    sbert_embeddings = sbert.encode(list(raw_texts), show_progress_bar=True)
    print(f"[OK] SBERT shape = {sbert_embeddings.shape}")

    # ----------------------------------------------------------------------
    # 3. TF-IDF features
    # ----------------------------------------------------------------------
    print("Building TF-IDF features...")

    if fit:
        tfidf = TfidfVectorizer(**TFIDF_CONFIG)
        tfidf_features = tfidf.fit_transform(cleaned_texts).toarray()
    else:
        if tfidf is None:
            raise ValueError("tfidf must be provided when fit=False")
        tfidf_features = tfidf.transform(cleaned_texts).toarray()

    print(f"[OK] TF-IDF shape = {tfidf_features.shape}")

    # ----------------------------------------------------------------------
    # 4. Custom handcrafted features
    # ----------------------------------------------------------------------
    print("Extracting custom features...")
    custom_list = [extract_custom_features(t) for t in raw_texts]
    custom_features = pd.DataFrame(custom_list).values
    print(f"[OK] Custom feature shape = {custom_features.shape}")

    # ----------------------------------------------------------------------
    # 5. Standardize handcrafted features
    # ----------------------------------------------------------------------
    print("Scaling custom features...")

    if fit:
        scaler = StandardScaler()
        custom_scaled = scaler.fit_transform(custom_features)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        custom_scaled = scaler.transform(custom_features)

    # ----------------------------------------------------------------------
    # 6. Combine all features
    # ----------------------------------------------------------------------
    X_combined = np.hstack([sbert_embeddings, tfidf_features, custom_scaled])
    print(f"[OK] Combined feature matrix = {X_combined.shape}")

    return X_combined, tfidf, scaler



