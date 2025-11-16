import json
import hashlib
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('chi', SelectKBest(chi2, k=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

plt.switch_backend("Agg")

# Download required NLTK resources once at import time
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Define key paths
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RAW_RESUME_PATH = BASE_DIR / "Resume.csv"
CLEANED_DATA_PATH = BASE_DIR / "cleaned_resume_data.csv"
CLEANED_JOBS_PATH = ARTIFACTS_DIR / "cleaned_job_data.csv"
TRAINING_DATA_PATH = BASE_DIR / "training_data.csv"
LATEST_MODEL_PATH = BASE_DIR / "resume_classifier_model.pkl"
LATEST_VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
PROFILE_PATH = ARTIFACTS_DIR / "data_profile.json"
CLEAN_CONFIG_PATH = ARTIFACTS_DIR / "clean_config.json"
EVAL_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
CV_RESULTS_PATH = ARTIFACTS_DIR / "cv_results.csv"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
EXPERIMENT_LOG_PATH = ARTIFACTS_DIR / "experiments_log.jsonl"

REQUIRED_RESUME_COLS = ("Category", "Resume_str")
REQUIRED_JOB_COLS = ("company_name", "job_description", "position_title")

URL_PATTERN = re.compile(r"http\S+|www\S+")
EMAIL_PATTERN = re.compile(r"\S*@\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

STOP_WORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {"resume", "job", "applicant", "skills", "experience", "position", "responsible"}
STOP_WORDS.update(CUSTOM_STOPWORDS)
LEMMATIZER = WordNetLemmatizer()


@dataclass
class CleanConfig:
    lower: bool = True
    strip_urls: bool = True
    strip_emails: bool = True
    remove_non_alpha: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    mask_numbers: bool = True


DEFAULT_CLEAN_CONFIG = CleanConfig()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Safely save JSON, converting unsupported objects to strings."""
    def safe_default(obj):
        try:
            return _json_default(obj)
        except TypeError:
            return str(obj)  # fallback for models, pipelines, etc.
    path.write_text(json.dumps(data, indent=2, default=safe_default), encoding="utf-8")




def file_md5(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_dataset(path: Path, required_cols: tuple, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset not found. Expected file at {path}")
    df = pd.read_csv(path)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{dataset_name} dataset missing required columns: {missing_cols}")
    return df


def profile_dataframe(df: pd.DataFrame, label_col: Optional[str] = None) -> Dict[str, Any]:
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
            label_counts / max(label_counts.sum(), 1)
        ).round(4).to_dict()
    return profile


def clean_text(text: Any, config: CleanConfig = DEFAULT_CLEAN_CONFIG) -> str:
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    if config.lower:
        text = text.lower()

    # Remove URLs & emails
    if config.strip_urls:
        text = URL_PATTERN.sub(" ", text)
    if config.strip_emails:
        text = EMAIL_PATTERN.sub(" ", text)

    # Standardize common career terms
    text = re.sub(r"c\+\+", "cpp", text)
    text = re.sub(r"\b(e-mail|email)\b", "email", text)

    # Remove generic resume words (VERY IMPORTANT)
    NOISE_PATTERNS = [
        r"\bsummary\b", r"\bobjective\b", r"\beducation\b", r"\bcontact\b",
        r"\bwork experience\b", r"\bexperience\b", r"\bskills\b",
        r"\bcompany name\b", r"\bphone number\b", r"\baddress\b",
        r"\bcity\b", r"\bstate\b",
        r"\bjr\b|\bsr\b|\bjunior\b|\bsenior\b",
    ]
    for patt in NOISE_PATTERNS:
        text = re.sub(patt, " ", text)

    # Mask numbers
    if config.mask_numbers:
        text = NUMBER_PATTERN.sub(" number ", text)

    # Remove all non-letters
    if config.remove_non_alpha:
        text = NON_ALPHA_PATTERN.sub(" ", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    if config.remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # Lemmatize
    if config.lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    # TRUNCATE overly long resumes (helps A LOT)
    MAX_TOKENS = 600
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    cleaned = " ".join(tokens)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned




def apply_cleaning(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    config: CleanConfig,
) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df[target_column] = cleaned_df[text_column].fillna('').apply(
        lambda value: clean_text(value, config)
    )
    return cleaned_df

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def build_param_grid(class_weight: Optional[str]):
    logistic = LogisticRegression(max_iter=800, solver='lbfgs', class_weight=class_weight)
    sgd = SGDClassifier(loss='modified_huber', max_iter=2000, class_weight=class_weight)
    svc = LinearSVC(class_weight=class_weight, max_iter=2000)
    nb = MultinomialNB()

    return [
        {'tfidf__ngram_range': [(1,2), (1,3)], 'tfidf__max_features': [5000, 10000],
         'clf': [logistic], 'clf__C': [0.5, 1.0, 2.0, 5.0]},
        {'tfidf__ngram_range': [(1,2)], 'clf': [sgd], 'clf__alpha': [1e-4, 1e-3, 1e-2]},
        {'tfidf__ngram_range': [(1,2)], 'clf': [svc], 'clf__C': [0.5, 1.0, 2.0]},
        {'tfidf__ngram_range': [(1,1)], 'clf': [nb], 'tfidf__max_features': [5000, 7000]}
    ]


def prepare_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()

    # Ensure no missing text
    df['cleaned_text'] = df['cleaned_text'].fillna('')

    # Remove empty rows
    df = df[df['cleaned_text'].str.strip() != ""]

    # Remove duplicate resumes (important!)
    df = df.drop_duplicates(subset=['cleaned_text'])

    X = df['cleaned_text']
    y = df['Category']

    # Class weight decision (imbalanced dataset)
    label_counts = y.value_counts(dropna=False)
    imbalance_ratio = label_counts.min() / max(label_counts.sum(), 1)
    class_weight: Optional[str] = 'balanced' if imbalance_ratio < 0.05 else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_weight": class_weight,
        "label_counts": label_counts.to_dict(),
    }


def train_and_evaluate(training_bundle: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = Pipeline(
    [
        ('tfidf', TfidfVectorizer(sublinear_tf=True, smooth_idf=True)),
        ('clf', LogisticRegression(max_iter=200)),  # Placeholder; overridden by grid
    ]
    )

    from sklearn.ensemble import VotingClassifier
    from sklearn.naive_bayes import MultinomialNB

    # Step 3.2: Ensemble model (VotingClassifier)
    log_reg = LogisticRegression(max_iter=800, solver='lbfgs', random_state=42)
    sgd = SGDClassifier(loss='modified_huber', max_iter=2000, random_state=42)
    nb = MultinomialNB()

    ensemble = VotingClassifier(
        estimators=[
            ('lr', log_reg),
            ('sgd', sgd),
            ('nb', nb)
        ],  
        voting='soft'
    )


    from sklearn.model_selection import RandomizedSearchCV

    grid = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=build_param_grid(training_bundle['class_weight']),
        n_iter=20,
        scoring='f1_weighted',
        cv=5,
        n_jobs=-1,
        random_state=42
    )


    grid.fit(training_bundle['X_train'], training_bundle['y_train'])
    best_model = grid.best_estimator_
    y_pred = best_model.predict(training_bundle['X_test'])
    evaluation = {
        'accuracy': accuracy_score(training_bundle['y_test'], y_pred),
        'f1_weighted': f1_score(training_bundle['y_test'], y_pred, average='weighted'),
        'classification_report': classification_report(
            training_bundle['y_test'], y_pred, output_dict=True
        ),
        'classification_report_text': classification_report(
            training_bundle['y_test'], y_pred
        ),
        'confusion_matrix': confusion_matrix(
            training_bundle['y_test'], y_pred, labels=best_model.classes_
        ).tolist(),
        'confusion_matrix_labels': best_model.classes_.tolist(),
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
    }

    # --- Step 4.1: Cross-validation summary ---
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_summary = cv_results[['mean_test_score', 'std_test_score', 'params']].sort_values(
        'mean_test_score', ascending=False
    )
    evaluation['cv_summary'] = cv_summary.head(5).to_dict(orient='records')

    evaluation['label_counts'] = training_bundle['label_counts']
    return {
        'model': best_model,
        'evaluation': evaluation,
        'y_test': training_bundle['y_test'],
        'y_pred': y_pred,
        'grid': grid,
    }

import seaborn as sns

def save_confusion_matrix(cm: List[List[int]], labels: List[str], path: Path) -> None:
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def persist_artifacts(
    pipeline_model: Pipeline,
    dataset_hash: str,
    clean_config: CleanConfig,
    timestamp: str,
) -> Dict[str, str]:
    vectorizer: TfidfVectorizer = pipeline_model.named_steps['tfidf']
    classifier = pipeline_model.named_steps['clf']
    pipeline_path = ARTIFACTS_DIR / f"resume_pipeline_{timestamp}.pkl"
    model_path = ARTIFACTS_DIR / f"resume_classifier_{timestamp}.pkl"
    vectorizer_path = ARTIFACTS_DIR / f"tfidf_vectorizer_{timestamp}.pkl"

    joblib.dump(pipeline_model, pipeline_path)
    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Update latest pointers used by the Streamlit app
    joblib.dump(classifier, LATEST_MODEL_PATH)
    joblib.dump(vectorizer, LATEST_VECTORIZER_PATH)

    metadata = {
        'saved_at': timestamp,
        'dataset_hash': dataset_hash,
        'clean_config': asdict(clean_config),
        'latest_model_path': str(LATEST_MODEL_PATH),
        'latest_vectorizer_path': str(LATEST_VECTORIZER_PATH),
        'versioned_model_path': str(model_path),
        'versioned_vectorizer_path': str(vectorizer_path),
        'pipeline_path': str(pipeline_path),
    }
    save_json(metadata, ARTIFACTS_DIR / f"artifact_metadata_{timestamp}.json")
    return metadata


def log_experiment(entry: Dict[str, Any]) -> None:
    entry_with_time = {
        "logged_at": datetime.utcnow().isoformat(),
        **entry,
    }

    def safe_default(obj):
        try:
            return _json_default(obj)
        except TypeError:
            return str(obj)

    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry_with_time, default=safe_default) + "\n")



def match_resume_to_jobs(
    resume_text: str,
    jobs_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    clean_config: CleanConfig = DEFAULT_CLEAN_CONFIG,
    classifier: Optional[Any] = None,
    top_k: int = 5,
    blend_weight: float = 0.2,
) -> List[Dict[str, Any]]:
    cleaned_resume = clean_text(resume_text, clean_config)
    working_jobs = jobs_df.copy()
    if 'cleaned_jd' not in working_jobs.columns:
        working_jobs['cleaned_jd'] = working_jobs['job_description'].fillna('').apply(
            lambda text: clean_text(text, clean_config)
        )
    resume_vector = vectorizer.transform([cleaned_resume])
    job_vectors = vectorizer.transform(working_jobs['cleaned_jd'])
    similarities = cosine_similarity(resume_vector, job_vectors)[0]

    classifier_scores = None
    if classifier is not None and hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(resume_vector)
        classifier_scores = probs.max(axis=1)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        score = similarities[idx]
        if classifier_scores is not None:
            score = (1 - blend_weight) * score + blend_weight * classifier_scores
        results.append(
            {
                'company_name': working_jobs.iloc[idx]['company_name'],
                'position_title': working_jobs.iloc[idx]['position_title'],
                'matching_score': float(score),
            }
        )
    return results


def match_job_to_all_resumes(
    job_description: str,
    resumes_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    clean_config: CleanConfig = DEFAULT_CLEAN_CONFIG,
    keywords: Optional[List[str]] = None,
    top_k: int = 10,
    blend_weight: float = 0.2,
) -> List[Dict[str, Any]]:

    # Clean job description
    cleaned_jd = clean_text(job_description, clean_config)

    # Clean resumes if needed
    working_resumes = resumes_df.copy()
    if 'cleaned_text' not in working_resumes.columns:
        working_resumes['cleaned_text'] = working_resumes['Resume_str'].fillna('').apply(
            lambda t: clean_text(t, clean_config)
        )

    # Vectorize
    jd_vec = vectorizer.transform([cleaned_jd])
    resume_vecs = vectorizer.transform(working_resumes['cleaned_text'])

    # Semantic similarity
    similarities = cosine_similarity(jd_vec, resume_vecs)[0]

    results = []
    for idx, score in enumerate(similarities):

        # Keyword matching
        matched_keywords = []
        if keywords:
            for kw in keywords:
                if kw.lower() in working_resumes.iloc[idx]['cleaned_text'].lower():
                    matched_keywords.append(kw)

        # Keyword score boost
        keyword_score = len(matched_keywords) / max(len(keywords), 1) if keywords else 0
        final_score = (1 - blend_weight) * score + blend_weight * keyword_score

        results.append({
            "resume_index": idx,
            "matching_score": float(final_score),
            "similarity_score": float(score),
            "keyword_score": float(keyword_score),
            "matched_keywords": matched_keywords,
            "raw_resume_text": resumes_df.iloc[idx]["Resume_str"],
            "cleaned_resume_text": working_resumes.iloc[idx]["cleaned_text"]
        })

    # Sort highest score first
    results = sorted(results, key=lambda x: x["matching_score"], reverse=True)

    return results[:top_k]


def rank_uploaded_resumes(
    job_title: str,
    job_description: str,
    resume_texts: List[str],
    resume_filenames: List[str],
    vectorizer: TfidfVectorizer,
    clean_config: CleanConfig = DEFAULT_CLEAN_CONFIG,
    classifier: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Rank uploaded candidate resumes against a job description.

    Args:
        job_title: The job position title
        job_description: The job description text
        resume_texts: List of resume text content extracted from uploaded files
        resume_filenames: List of resume filenames for identification
        vectorizer: Fitted TF-IDF vectorizer
        clean_config: Configuration for text cleaning
        classifier: Optional trained classifier for category prediction

    Returns:
        List of ranked candidates with scores, sorted highest to lowest
    """
    # Combine job title and description
    job_text = f"{job_title} {job_description}"
    cleaned_job = clean_text(job_text, clean_config)

    # Vectorize job description
    job_vector = vectorizer.transform([cleaned_job])

    # Clean and vectorize all resumes
    cleaned_resumes = [clean_text(resume, clean_config) for resume in resume_texts]
    resume_vectors = vectorizer.transform(cleaned_resumes)

    # Calculate similarity scores
    similarities = cosine_similarity(job_vector, resume_vectors)[0]

    # Build results
    results: List[Dict[str, Any]] = []
    for idx, (filename, resume_text, similarity) in enumerate(zip(resume_filenames, resume_texts, similarities)):
        result = {
            'rank': 0,  # Will be assigned after sorting
            'filename': filename,
            'matching_score': float(similarity),
            'matching_percentage': float(similarity * 100),
        }

        # Add category prediction if classifier available
        if classifier is not None:
            resume_vector = vectorizer.transform([cleaned_resumes[idx]])
            predicted_category = classifier.predict(resume_vector)[0]
            result['predicted_category'] = predicted_category

            if hasattr(classifier, 'predict_proba'):
                probs = classifier.predict_proba(resume_vector)[0]
                result['category_confidence'] = float(max(probs) * 100)

        results.append(result)

    # Sort by matching score (highest first)
    results.sort(key=lambda x: x['matching_score'], reverse=True)

    # Assign ranks
    for rank, result in enumerate(results, start=1):
        result['rank'] = rank

    return results



def main() -> None:
    ensure_artifacts_dir()
    clean_config = DEFAULT_CLEAN_CONFIG

    resume_df = load_dataset(RAW_RESUME_PATH, REQUIRED_RESUME_COLS, 'Resume')
    job_df = load_dataset(TRAINING_DATA_PATH, REQUIRED_JOB_COLS, 'Job description')

    profile = {
        'generated_at': datetime.utcnow().isoformat(),
        'resume': profile_dataframe(resume_df, 'Category'),
        'jobs': profile_dataframe(job_df),
    }
    save_json(profile, PROFILE_PATH)
    print("✅ Data profiling saved to", PROFILE_PATH)

    cleaned_resumes = apply_cleaning(resume_df, 'Resume_str', 'cleaned_text', clean_config)
    cleaned_resumes.to_csv(CLEANED_DATA_PATH, index=False)
    save_json(asdict(clean_config), CLEAN_CONFIG_PATH)
    print("✅ Cleaned resumes saved to", CLEANED_DATA_PATH)

    cleaned_jobs = apply_cleaning(job_df, 'job_description', 'cleaned_jd', clean_config)
    cleaned_jobs.to_csv(CLEANED_JOBS_PATH, index=False)
    print("✅ Cleaned job descriptions saved to", CLEANED_JOBS_PATH)

    dataset_hash = file_md5(CLEANED_DATA_PATH)

    # Train the model
    training_bundle = prepare_training_data(cleaned_resumes)
    training_output = train_and_evaluate(training_bundle)

    evaluation = training_output['evaluation']
    save_json(evaluation, EVAL_REPORT_PATH)
    pd.DataFrame(training_output['grid'].cv_results_).to_csv(CV_RESULTS_PATH, index=False)
    save_confusion_matrix(
        evaluation['confusion_matrix'],
        evaluation['confusion_matrix_labels'],
        CONFUSION_MATRIX_PATH,
    )
    print("✅ Evaluation artifacts saved to", ARTIFACTS_DIR)

    # --- Step 4.2: Display model performance summary ---
    print("\n📈 Model Performance Summary:")
    print(f"   • Accuracy:       {evaluation['accuracy']:.4f}")
    print(f"   • F1 (Weighted):  {evaluation['f1_weighted']:.4f}")
    print(f"   • Best CV Score:  {evaluation['best_score']:.4f}")
    print(f"   • Best Parameters: {evaluation['best_params']}")


    # --- Step 4.3: Error analysis (fixed version) ---
    y_test = training_output['y_test']
    y_pred = training_output['y_pred']
    X_test_reset = training_bundle['X_test'].reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    error_df = pd.DataFrame({
        'resume_text': X_test_reset,
        'true_label': y_test_reset,
        'predicted_label': y_pred
    })
    misclassified = error_df[error_df['true_label'] != error_df['predicted_label']]

    print(f"🔍 Found {len(misclassified)} misclassified samples out of {len(y_test_reset)}")
    if not misclassified.empty:
        print("\nExamples of misclassifications:")
        print(misclassified.head(5).to_string(index=False))

        # Save full misclassified rows to file
        misclassified.to_csv("misclassified_samples.csv", index=False)
        print("📁 Saved full misclassified samples to misclassified_samples.csv")


    # --- Step 4.4: Visualize top cross-validation results ---
    cv_df = pd.DataFrame(training_output['grid'].cv_results_)
    top_models = cv_df.nlargest(5, 'mean_test_score')[['param_clf', 'mean_test_score']]
    plt.figure(figsize=(8,5))
    plt.barh(top_models['param_clf'].astype(str), top_models['mean_test_score'])
    plt.xlabel('Mean F1-Weighted Score')
    plt.title('Top 5 Model Performances')
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "top_model_scores.png")
    plt.close()
    print("📊 Saved model comparison chart to artifacts folder.")

    # --- Step 4.5: Logging and sample matching ---
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    metadata = persist_artifacts(training_output['model'], dataset_hash, clean_config, timestamp)

    log_experiment(
        {
            'dataset_hash': dataset_hash,
            'best_params': evaluation['best_params'],
            'accuracy': evaluation['accuracy'],
            'f1_weighted': evaluation['f1_weighted'],
            'artifact_metadata': metadata,
        }
    )

    sample_resume = cleaned_resumes['cleaned_text'].iloc[0]
    vectorizer: TfidfVectorizer = training_output['model'].named_steps['tfidf']
    classifier = training_output['model'].named_steps['clf']
    top_matches = match_resume_to_jobs(
        sample_resume,
        cleaned_jobs,
        vectorizer=vectorizer,
        clean_config=clean_config,
        classifier=classifier,
        top_k=3,
    )

    print("\n✅ Sample resume to job matches:")
    for match in top_matches:
        print(f"{match['company_name']} - {match['position_title']} (score: {match['matching_score']:.2f})")

    print("✅ Evaluation artifacts saved to", ARTIFACTS_DIR)

    print("\n🔎 Running new job-to-resume matching...")

    sample_job_description = """
    We are hiring a Software Engineer with strong skills in Python, machine learning,
    data analysis, problem solving, and experience with cloud environments.
    """

    sample_keywords = ["python", "machine learning", "cloud", "data analysis"]

    results = match_job_to_all_resumes(
        job_description=sample_job_description,
        resumes_df=cleaned_resumes,
        vectorizer=vectorizer,
        clean_config=clean_config,
        keywords=sample_keywords,
        top_k=5,
    )

    for r in results:
        print(f"Score: {r['matching_score']:.2f} | Keywords: {r['matched_keywords']}")





# ✅ Always keep this at the very bottom — nothing below it
if __name__ == '__main__':
    main()








