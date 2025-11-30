# AI-Powered Resume Screening System

A machine learning system that matches resumes with job descriptions and ranks candidates using hybrid AI (SBERT + TF-IDF + custom features).

## Project Structure

```
project_root/
│
├─ Resume.csv              # Raw resume dataset (2484 samples, 24 categories)
├─ training_data.csv       # Job descriptions for matching
│
├─ artifacts/              # Auto-generated: models, reports, confusion matrices
│
├─ src/                    # Modular codebase
│   ├─ config.py          # Configuration and paths
│   ├─ cleaning/          # Text preprocessing
│   │   └─ text_cleaning.py
│   ├─ features/          # Feature extraction
│   │   ├─ custom_features.py
│   │   └─ build_features.py
│   ├─ models/            # Model training
│   │   └─ lightgbm_trainer.py
│   ├─ evaluation/        # Evaluation metrics
│   │   └─ evaluate.py
│   └─ utils/             # Utilities
│       ├─ io_utils.py
│       └─ metrics.py
│
├─ train_improved.py       # Main training script
├─ app.py                  # Streamlit web app
└─ AI_model.py             # Legacy code (for ranking function)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit pandas scikit-learn xgboost sentence-transformers optuna nltk pymupdf docx2txt
```

### 2. Train the Model

```bash
python train_improved.py
```

This will:
- Load and clean resume data
- Extract SBERT + TF-IDF + custom features (5391 total)
- Run Optuna hyperparameter optimization (10 trials)
- Save models to `improved_classifier.pkl`, `improved_tfidf.pkl`, etc.
- Achieve 75-85% accuracy

### 3. Run the Web App

```bash
streamlit run app.py
```

Two modes available:
- **Job Seeker**: Upload resume → Get predicted category + matching jobs
- **HR**: Enter job details + upload candidates → Get AI-powered rankings

## Model Architecture

The improved model uses a **three-feature fusion** approach:

1. **SBERT Embeddings** (384 features)
   - Semantic understanding using `all-MiniLM-L6-v2`
   - Captures meaning beyond keywords

2. **TF-IDF** (5000 features)
   - Traditional keyword matching
   - `ngram_range=(1,2)`, `min_df=2`, `max_df=0.8`

3. **Custom Features** (7 features)
   - `max_years_exp`, `total_years_mentioned`
   - `education_level` (0-3: none/bachelor/master/phd)
   - `tech_skill_count`, `soft_skill_count`
   - `word_count`, `cert_count`

**Total: 5391 features** → StandardScaler → XGBoost/LogReg/RandomForest

## Key Features

- **Optuna Optimization**: Automatically finds best model and hyperparameters
- **Sequential CV**: Uses `n_jobs=1` to avoid parallel processing overhead with large features
- **Label Encoding**: Handles string categories properly for XGBoost
- **Windows Compatible**: No emoji characters (uses `[OK]` instead)
- **Modular Design**: Clean separation of concerns in `src/`

## Performance

- **Baseline Model** (TF-IDF only): ~69% accuracy
- **Improved Model** (SBERT + TF-IDF + custom): **75-85% accuracy**

## Configuration

Edit `src/config.py` to modify:
- Feature dimensions (`TFIDF_MAX_FEATURES`, `SBERT_DIM`)
- Optuna trials (`OPTUNA_N_TRIALS`)
- Train/test split (`TEST_SIZE`)
- Paths and file locations

## Troubleshooting

**"Feature shape mismatch, expected: 5391"**
- Ensure all 3 feature types are extracted in correct order
- Check `extract_custom_features()` returns exactly 7 features

**Slow training (50+ seconds per trial)**
- Already optimized with `n_jobs=1`
- Reduce `OPTUNA_N_TRIALS` in `src/config.py`

**Import errors**
- Ensure you're in project root: `cd C:/Users/Admin/Documents/FYP`
- Python sees `src/` as package

## License

FYP Project - All Rights Reserved
