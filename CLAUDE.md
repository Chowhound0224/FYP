# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Powered Resume Screening System - A Final Year Project (FYP) that uses machine learning to match resumes with job descriptions and rank candidates.

**Tech Stack**: Python, Streamlit, scikit-learn, XGBoost, SBERT (Sentence Transformers), Optuna

## Modular Structure

The codebase has been refactored into a clean, modular structure:

```
src/
├─ config.py              # All configuration, paths, constants
├─ cleaning/              # Text preprocessing
├─ features/              # Feature extraction (SBERT + TF-IDF + custom)
├─ models/                # Training logic with Optuna
├─ evaluation/            # Model evaluation and metrics
└─ utils/                 # I/O and helper functions
```

This structure separates concerns and makes the code easier to modify and test.

## Running the Application

### Train the improved model:
```bash
python train_improved.py
```

Runs complete pipeline:
1. Loads Resume.csv and cleans data
2. Extracts 3 feature types (SBERT + TF-IDF + custom = 5391 features)
3. Optuna optimization (10 trials, ~1-2 minutes)
4. Saves models to `improved_*.pkl` files
5. Generates confusion matrix and metadata

### Start the Streamlit web app:
```bash
streamlit run app.py
```

Two modes:
1. **Job Seeker Mode**: Upload resume → Get predicted category + top 3 matching jobs
2. **HR Mode**: Enter job details + upload candidates → Get AI-powered rankings

## Architecture

### Feature Pipeline (src/features/)

The core innovation is **three-feature fusion**:

1. **SBERT Embeddings** (384 dims)
   - Built in `build_features.py::build_combined_features()`
   - Uses `all-MiniLM-L6-v2` model cached via singleton pattern
   - Generated from **raw text** (not cleaned) for better semantic understanding

2. **TF-IDF Features** (5000 dims)
   - Config in `src/config.py::TFIDF_CONFIG`
   - Generated from **cleaned text** (lemmatized, no stopwords)
   - `ngram_range=(1,2)` for bigrams

3. **Custom Features** (7 dims)
   - Extracted in `features/custom_features.py::extract_custom_features()`
   - Returns dict with: max_years_exp, total_years_mentioned, education_level, tech_skill_count, soft_skill_count, word_count, cert_count
   - Scaled with StandardScaler before combining

**Critical**: Features must be extracted in this exact order for shape to match (5391 total).

### Training (src/models/)

**`lightgbm_trainer.py::train_with_optuna()`**
- Tests 3 model types: XGBoost, LogisticRegression, RandomForest
- Uses TPE sampler with configurable trials (default: 10)
- **Sequential CV** (`n_jobs=1`) avoids parallel overhead with large features
- Returns: (best_model, best_params, best_score)

### Evaluation (src/evaluation/)

**`evaluate.py::evaluate_model()`**
- Calculates accuracy, F1-weighted, confusion matrix
- Uses label_encoder to display category names in classification report
- Returns dict with all metrics for saving

### Configuration (src/config.py)

Central location for all settings:
- **Paths**: All file paths as Path objects
- **Feature Config**: `SBERT_DIM`, `TFIDF_MAX_FEATURES`, `CUSTOM_FEATURES_COUNT`
- **Training Config**: `OPTUNA_N_TRIALS`, `OPTUNA_CV_FOLDS`, `RANDOM_STATE`
- **Cleaning Config**: `CleanConfig` dataclass with preprocessing options

### Data Flow

```
train_improved.py:
  1. load_dataset() → DataFrame
  2. apply_cleaning() → cleaned text
  3. build_combined_features() → X_combined (5391 features)
  4. train_with_optuna() → best_model
  5. evaluate_model() → metrics
  6. save models + artifacts

app.py:
  1. load_models() → (classifier, tfidf, scaler, label_encoder, sbert)
  2. predict_category_improved() → category name
  3. rank_uploaded_resumes_hybrid() → ranked candidates
```

## Important Implementation Details

### Label Encoding
- **Training**: LabelEncoder converts string categories → integers (0-23)
- **Prediction**: Must use `label_encoder.inverse_transform()` to get category names
- **XGBoost requirement**: Only accepts numeric labels, not strings

### Windows Console Compatibility
- **Never use emoji characters in print()** - causes `UnicodeEncodeError`
- Console encoding is cp1252, doesn't support Unicode emojis
- Use ASCII: `[OK]`, `[ERROR]` instead of ✅, ❌

### Feature Synchronization
- `extract_custom_features()` returns **exactly 7 features**
- Order matters for hstack: SBERT (384) + TF-IDF (5000) + custom (7) = 5391
- If features don't match, model will raise "Feature shape mismatch"

### Optuna Performance
- 10 trials (not 50) for speed (~1-2 minutes total)
- Each trial: ~6-10 seconds with `n_jobs=1`
- Using `n_jobs=-1` causes 50+ seconds per trial due to parallel overhead

### SBERT Model Loading
- First load: ~10 seconds (downloads model)
- Cached globally in `features/build_features.py::_sbert_model`
- Singleton pattern prevents re-downloading

## Common Issues

**"ModuleNotFoundError: No module named 'src'"**
- Run commands from project root: `C:/Users/Admin/Documents/FYP`
- Python needs to see `src/` as a package

**"Feature shape mismatch, expected: 5391, got 5000"**
- Missing custom features or scaler in prediction
- Check `build_combined_features()` is called with all 3 feature types

**"Invalid classes inferred from unique values of y"**
- Forgot to encode labels before XGBoost training
- Use `LabelEncoder().fit_transform()` before passing to model

**Slow Optuna (50+ sec/trial)**
- Change `OPTUNA_N_JOBS` from -1 to 1 in config.py
- Already optimized in current code

## File Locations

### Training Outputs
- **Latest models** (used by app.py): `improved_classifier.pkl`, `improved_tfidf.pkl`, `improved_scaler.pkl`, `improved_label_encoder.pkl`
- **Timestamped versions**: `artifacts/improved_model_YYYYMMDD_HHMMSS.pkl`
- **Metadata**: `artifacts/improved_model_metadata_*.json`
- **Confusion Matrix**: `artifacts/confusion_matrix_*.png`

### Data Files
- `Resume.csv` - Raw dataset (2484 samples, 24 categories)
- `training_data.csv` - Job descriptions for matching (used by app.py)
- `cleaned_resume_data.csv` - Preprocessed resumes (generated during training)

## Modifying the Code

### Adding New Features
1. Edit `src/features/custom_features.py::extract_custom_features()`
2. Add new feature to returned dict
3. Update `CUSTOM_FEATURES_COUNT` in `src/config.py`
4. Update `TOTAL_FEATURES` calculation
5. Re-train model: `python train_improved.py`

### Changing Model Types
1. Edit `src/models/lightgbm_trainer.py::objective()`
2. Add new model type to `trial.suggest_categorical()`
3. Add hyperparameter suggestions for new model
4. Re-train: `python train_improved.py`

### Adjusting Training Speed
- `src/config.py::OPTUNA_N_TRIALS` - reduce from 10 to 5 for faster training
- `src/config.py::TFIDF_MAX_FEATURES` - reduce from 5000 to 1000 for smaller features
- Trade-off: speed vs accuracy

### Legacy Code
- `AI_model.py` - Still contains `rank_uploaded_resumes_hybrid()` function
- This function is used by app.py for HR ranking mode
- Will be refactored to `src/` structure in future
