import streamlit as st
import pandas as pd
import re
import fitz  # PyMuPDF for PDFs
import docx2txt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np

# ==========================
# Load trained ML components
# ==========================
svm_model = joblib.load("resume_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Load job description dataset
jd_df = pd.read_csv("training_data.csv")

# ==========================
# Helper functions
# ==========================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("🤖 AI-Powered Resume Screening System")
st.write("Upload your resume (PDF/DOCX) and find your best job match instantly.")

uploaded_file = st.file_uploader("📂 Upload your resume", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format!")
        st.stop()

    # Clean text
    cleaned_resume = clean_text(resume_text)

    # Predict category
    resume_vector = tfidf.transform([cleaned_resume])
    predicted_category = svm_model.predict(resume_vector)[0]

    st.success(f"🧠 Predicted Job Category: **{predicted_category}**")

    # Clean job descriptions
    jd_df['cleaned_jd'] = jd_df['job_description'].apply(clean_text)

    # Transform job descriptions
    jd_tfidf = tfidf.transform(jd_df['cleaned_jd'])

    # Compute cosine similarity
    similarities = cosine_similarity(resume_vector, jd_tfidf)[0]

    # Get top 3 matches
    top_indices = np.argsort(similarities)[-3:][::-1]

    st.subheader("🎯 Top Matching Job Positions:")
    for idx in top_indices:
        st.write(f"**Company:** {jd_df.iloc[idx]['company_name']}")
        st.write(f"**Position:** {jd_df.iloc[idx]['position_title']}")
        st.write(f"**Matching Score:** {similarities[idx] * 100:.2f}%")
        st.write("---")

    st.success("✅ Matching complete! Scroll up to see your results.")
