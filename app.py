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
from AI_model import rank_uploaded_resumes, CleanConfig, DEFAULT_CLEAN_CONFIG

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

# Create two modes: Job Seeker vs HR
mode = st.sidebar.radio(
    "Select Mode:",
    ["Job Seeker - Find Matching Jobs", "HR - Rank Candidates"]
)

# ==========================
# MODE 1: Job Seeker Mode
# ==========================
if mode == "Job Seeker - Find Matching Jobs":
    st.write("Upload your resume (PDF/DOCX) and find your best job match instantly.")

    uploaded_files = st.file_uploader("📂 Upload your resumes", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"📄 **Processing:** {uploaded_file.name}")
            resume_text = ""  # initialize variable to avoid NameError

            # Detect file type by extension
            file_name = uploaded_file.name.lower()
            if file_name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif file_name.endswith(".docx"):
                resume_text = extract_text_from_docx(uploaded_file)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue  # skip this file

            # Clean text
            cleaned_resume = clean_text(resume_text)

            # Predict category
            resume_vector = tfidf.transform([cleaned_resume])
            predicted_category = svm_model.predict(resume_vector)[0]
            st.success(f"🧠 Predicted Job Category for {uploaded_file.name}: **{predicted_category}**")

            # Compute job matches
            jd_df['cleaned_jd'] = jd_df['job_description'].apply(clean_text)
            jd_tfidf = tfidf.transform(jd_df['cleaned_jd'])
            similarities = cosine_similarity(resume_vector, jd_tfidf)[0]
            top_indices = np.argsort(similarities)[-3:][::-1]

            st.subheader(f"🎯 Top Matching Job Positions for {uploaded_file.name}:")
            for idx in top_indices:
                st.write(f"**Company:** {jd_df.iloc[idx]['company_name']}")
                st.write(f"**Position:** {jd_df.iloc[idx]['position_title']}")
                st.write(f"**Matching Score:** {similarities[idx] * 100:.2f}%")
                st.write("---")

        st.success("✅ All resumes processed successfully! Scroll up to see your results.")

# ==========================
# MODE 2: HR Ranking Mode
# ==========================
else:
    st.write("Enter job details and upload candidate resumes to get AI-powered rankings.")

    # Job input section
    st.subheader("📋 Job Details")
    job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
    job_description = st.text_area(
        "Job Description",
        placeholder="Enter the job description, requirements, and qualifications...",
        height=200
    )

    # Resume upload section
    st.subheader("📂 Upload Candidate Resumes")
    candidate_files = st.file_uploader(
        "Upload resumes (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="hr_uploader"
    )

    # Process and rank button
    if st.button("🚀 Rank Candidates", type="primary"):
        if not job_title or not job_description:
            st.error("⚠️ Please enter both job title and description!")
        elif not candidate_files:
            st.error("⚠️ Please upload at least one candidate resume!")
        else:
            with st.spinner("🔄 Processing and ranking candidates..."):
                # Extract text from all uploaded resumes
                resume_texts = []
                resume_filenames = []

                for uploaded_file in candidate_files:
                    file_name = uploaded_file.name.lower()
                    resume_text = ""

                    try:
                        if file_name.endswith(".pdf"):
                            resume_text = extract_text_from_pdf(uploaded_file)
                        elif file_name.endswith(".docx"):
                            resume_text = extract_text_from_docx(uploaded_file)

                        if resume_text.strip():
                            resume_texts.append(resume_text)
                            resume_filenames.append(uploaded_file.name)
                        else:
                            st.warning(f"⚠️ {uploaded_file.name} appears to be empty or unreadable")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

                if not resume_texts:
                    st.error("❌ No valid resumes could be processed!")
                else:
                    # Rank candidates using the AI model
                    ranked_results = rank_uploaded_resumes(
                        job_title=job_title,
                        job_description=job_description,
                        resume_texts=resume_texts,
                        resume_filenames=resume_filenames,
                        vectorizer=tfidf,
                        clean_config=DEFAULT_CLEAN_CONFIG,
                        classifier=svm_model
                    )

                    # Display results
                    st.success(f"✅ Successfully ranked {len(ranked_results)} candidates!")
                    st.subheader("🏆 Candidate Rankings")

                    # Display each ranked candidate
                    for result in ranked_results:
                        rank = result['rank']
                        filename = result['filename']
                        score = result['matching_percentage']
                        category = result.get('predicted_category', 'N/A')

                        # Color code based on score
                        if score >= 70:
                            color = "🟢"
                            badge_color = "green"
                        elif score >= 50:
                            color = "🟡"
                            badge_color = "orange"
                        else:
                            color = "🔴"
                            badge_color = "red"

                        with st.container():
                            col1, col2, col3, col4 = st.columns([1, 4, 2, 2])

                            with col1:
                                st.markdown(f"### #{rank}")

                            with col2:
                                st.markdown(f"**{filename}**")

                            with col3:
                                st.markdown(f"{color} **{score:.1f}%** match")

                            with col4:
                                st.markdown(f"Category: *{category}*")

                            st.divider()

                    # Download results as CSV
                    results_df = pd.DataFrame(ranked_results)
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Rankings as CSV",
                        data=csv,
                        file_name="candidate_rankings.csv",
                        mime="text/csv"
                    )
