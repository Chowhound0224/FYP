import pandas as pd
import nltk
import re
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define key paths
BASE_DIR = Path(__file__).resolve().parent
RAW_RESUME_PATH = BASE_DIR / "Resume.csv"
CLEANED_DATA_PATH = BASE_DIR / "cleaned_resume_data.csv"
TRAINING_DATA_PATH = BASE_DIR / "training_data.csv"
MODEL_PATH = BASE_DIR / "resume_classifier_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"

if not RAW_RESUME_PATH.exists():
    raise FileNotFoundError(
        f"Resume dataset not found. Expected file at {RAW_RESUME_PATH}"
    )

# Load Resume dataset
df = pd.read_csv(RAW_RESUME_PATH)
print(df.head())
print(df.columns)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers/punctuation
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply cleaning to Resume_str column
df['cleaned_text'] = df['Resume_str'].apply(clean_text)

# Show sample output
print(df[['Category', 'cleaned_text']].head())

# Save cleaned data
df.to_csv(CLEANED_DATA_PATH, index=False)
print("✅ Cleaned resume data saved successfully!")



# ===========================================
# STEP 3: FEATURE EXTRACTION & MODEL TRAINING
# ===========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
df['cleaned_text'] = df['cleaned_text'].fillna('')  # Handle NaN values
X = tfidf.fit_transform(df['cleaned_text'])   # Features
y = df['Category']                            # Labels

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the SVM classifier
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation metrics
print("✅ Model training complete!\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Persist artifacts for downstream apps
joblib.dump(svm_model, MODEL_PATH)
joblib.dump(tfidf, VECTORIZER_PATH)
print(f"✅ Saved SVM model to {MODEL_PATH}")
print(f"✅ Saved TF-IDF vectorizer to {VECTORIZER_PATH}")



# ======================================================
# STEP 5: MATCHING SCORE BETWEEN RESUME & JOB DESCRIPTION
# ======================================================
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load trained model + vectorizer (if saved)
if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
    svm_model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    print("✅ Loaded persisted model and vectorizer.")
else:
    print("WARNING: Saved artifacts not found; using in-memory model and vectorizer.")

# Load job description dataset
if not TRAINING_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Job description dataset not found. Expected file at {TRAINING_DATA_PATH}"
    )

jd_df = pd.read_csv(TRAINING_DATA_PATH)

# Clean the job descriptions
jd_df['cleaned_jd'] = jd_df['job_description'].apply(clean_text)

# Transform job descriptions to TF-IDF vectors
jd_tfidf = tfidf.transform(jd_df['cleaned_jd'])

# Example: take one sample resume from your dataset
sample_resume = df['cleaned_text'].iloc[0]

# Transform resume into vector
resume_vector = tfidf.transform([sample_resume])

# Calculate cosine similarity
similarities = cosine_similarity(resume_vector, jd_tfidf)[0]

# Find best match
best_match_idx = np.argmax(similarities)
best_score = similarities[best_match_idx] * 100
best_company = jd_df.iloc[best_match_idx]['company_name']
best_position = jd_df.iloc[best_match_idx]['position_title']

print("\n✅ Resume–Job Matching Result:")
print(f"Best Match → {best_company} ({best_position})")
print(f"Matching Score: {best_score:.2f}%")

