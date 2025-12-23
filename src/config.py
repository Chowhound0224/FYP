"""Configuration and paths for the resume screening system."""

import re
from dataclasses import dataclass
from pathlib import Path

import nltk
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Data paths
RAW_RESUME_PATH = BASE_DIR / "Resume.csv"
CLEANED_DATA_PATH = BASE_DIR / "cleaned_resume_data.csv"
CLEANED_JOBS_PATH = ARTIFACTS_DIR / "cleaned_job_data.csv"
TRAINING_DATA_PATH = BASE_DIR / "training_data.csv"

# Model paths (legacy)
LATEST_MODEL_PATH = BASE_DIR / "resume_classifier_model.pkl"
LATEST_VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

# Improved model paths
IMPROVED_CLASSIFIER_PATH = BASE_DIR / "improved_classifier.pkl"
IMPROVED_TFIDF_PATH = BASE_DIR / "improved_tfidf.pkl"
IMPROVED_SCALER_PATH = BASE_DIR / "improved_scaler.pkl"
IMPROVED_LABEL_ENCODER_PATH = BASE_DIR / "improved_label_encoder.pkl"
IMPROVED_METADATA_PATH = ARTIFACTS_DIR / "improved_model_metadata.json"
IMPROVED_EVAL_REPORT_PATH = ARTIFACTS_DIR / "improved_evaluation.json"
IMPROVED_CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "improved_confusion_matrix.png"


# Artifact paths
PROFILE_PATH = ARTIFACTS_DIR / "data_profile.json"
CLEAN_CONFIG_PATH = ARTIFACTS_DIR / "clean_config.json"
EVAL_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
CV_RESULTS_PATH = ARTIFACTS_DIR / "cv_results.csv"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
EXPERIMENT_LOG_PATH = ARTIFACTS_DIR / "experiments_log.jsonl"

# Required columns
REQUIRED_RESUME_COLS = ("Category", "Resume_str")
REQUIRED_JOB_COLS = ("company_name", "job_description", "position_title")

# ============================================================================
# TEXT CLEANING
# ============================================================================

# Regex patterns
URL_PATTERN = re.compile(r"http\S+|www\S+")
EMAIL_PATTERN = re.compile(r"\S*@\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Stop words
STOP_WORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {"resume", "job", "applicant", "skills", "experience", "position", "responsible"}
STOP_WORDS.update(CUSTOM_STOPWORDS)


@dataclass
class CleanConfig:
    """Configuration for text cleaning pipeline."""
    lower: bool = True
    strip_urls: bool = True
    strip_emails: bool = True
    remove_non_alpha: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    mask_numbers: bool = True


DEFAULT_CLEAN_CONFIG = CleanConfig()

# ============================================================================
# MODEL TRAINING
# ============================================================================

# SBERT model name
SBERT_MODEL_NAME = "all-mpnet-base-v2"  # Upgraded from all-MiniLM-L6-v2 (384 dims)


# Feature dimensions
SBERT_DIM = 768  # all-mpnet-base-v2 embedding dimension
TFIDF_MAX_FEATURES = 5000  # Reduced from 25000 for faster training and less overfitting
CUSTOM_FEATURES_COUNT = 7
TOTAL_FEATURES = SBERT_DIM + TFIDF_MAX_FEATURES + CUSTOM_FEATURES_COUNT  # 5775

# TF-IDF configuration
TFIDF_CONFIG = {
    "max_features": TFIDF_MAX_FEATURES,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.8,
    "sublinear_tf": True
}

# Optuna configuration
OPTUNA_N_TRIALS = 10
OPTUNA_CV_FOLDS = 5
OPTUNA_N_JOBS = 1  # Sequential processing for large feature sets

# Random state
RANDOM_STATE = 42
TEST_SIZE = 0.20
