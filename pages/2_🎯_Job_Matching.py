"""
Page 2: Job Matching & Candidate Ranking
Select job category and describe requirements to rank candidates
"""

import streamlit as st
import tempfile
import docx2txt
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np

from src.prediction.predict import predict_resume_category

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Job Matching",
    page_icon="🎯",
    layout="wide"
)

# ============================================================================
# JOB CATEGORIES (24 categories from dataset)
# ============================================================================
JOB_CATEGORIES = [
    "ACCOUNTANT",
    "ADVOCATE",
    "AGRICULTURE",
    "APPAREL",
    "ARTS",
    "AUTOMOBILE",
    "AVIATION",
    "BANKING",
    "BPO",
    "BUSINESS-DEVELOPMENT",
    "CHEF",
    "CONSTRUCTION",
    "CONSULTANT",
    "DESIGNER",
    "DIGITAL-MEDIA",
    "ENGINEERING",
    "FINANCE",
    "FITNESS",
    "HEALTHCARE",
    "HR",
    "INFORMATION-TECHNOLOGY",
    "PUBLIC-RELATIONS",
    "SALES",
    "TEACHER"
]

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .job-form {
        background: #f8f9fa;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
    }

    .rank-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .rank-gold {
        border-left-color: #FFD700 !important;
        background: linear-gradient(135deg, #fff9e6 0%, #ffffff 100%);
    }

    .rank-silver {
        border-left-color: #C0C0C0 !important;
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
    }

    .rank-bronze {
        border-left-color: #CD7F32 !important;
        background: linear-gradient(135deg, #fff0e6 0%, #ffffff 100%);
    }

    .rank-normal {
        border-left-color: #667eea !important;
    }

    .match-score-high {
        color: #28a745;
        font-size: 32px;
        font-weight: 800;
    }

    .match-score-medium {
        color: #ffc107;
        font-size: 32px;
        font-weight: 800;
    }

    .match-score-low {
        color: #dc3545;
        font-size: 32px;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT."""
    suffix = uploaded_file.name.lower()

    try:
        if suffix.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif suffix.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            return docx2txt.process(tmp_path)

        elif suffix.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")

        else:
            return ""
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""


def calculate_match_score(predicted_category: str, target_category: str, confidence: float) -> float:
    """
    Calculate matching score between predicted and target category.

    Returns:
        Score between 0-100
    """
    # Exact match
    if predicted_category == target_category:
        return confidence * 100

    # Partial match (if related categories exist, can add logic here)
    # For now, return low score for mismatch
    return confidence * 30


def get_score_class(score: float) -> str:
    """Return CSS class based on match score."""
    if score >= 70:
        return "match-score-high"
    elif score >= 50:
        return "match-score-medium"
    else:
        return "match-score-low"


def get_rank_class(rank: int) -> str:
    """Return CSS class based on rank."""
    if rank == 1:
        return "rank-gold"
    elif rank == 2:
        return "rank-silver"
    elif rank == 3:
        return "rank-bronze"
    else:
        return "rank-normal"


# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<h1 style="color: #667eea;">🎯 Job Matching & Candidate Ranking</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="job-form">
    <h2 style="margin-top: 0; color: #333;">Step 1: Define Job Requirements</h2>
    <p style="color: #666;">Select the target job category and describe requirements</p>
</div>
""", unsafe_allow_html=True)

# Job requirements form
with st.form("job_form"):
    col1, col2 = st.columns([1, 2])

    with col1:
        target_category = st.selectbox(
            "🎯 Target Job Category",
            options=JOB_CATEGORIES,
            help="Select the category you're hiring for"
        )

    with col2:
        job_title = st.text_input(
            "💼 Job Title",
            placeholder="e.g., Senior Software Engineer, Marketing Manager",
            help="Enter the specific job title"
        )

    job_description = st.text_area(
        "📝 Job Description & Requirements",
        height=200,
        placeholder="""Describe the role, key responsibilities, and requirements. For example:

- 5+ years of experience in software development
- Proficiency in Python, Java, or C++
- Experience with cloud platforms (AWS, Azure, GCP)
- Strong problem-solving and communication skills
- Bachelor's degree in Computer Science or related field
        """,
        help="Be as detailed as possible for better matching"
    )

    submitted = st.form_submit_button("💾 Save Job Requirements", type="primary", use_container_width=True)

    if submitted:
        if not job_title or not job_description:
            st.error("⚠️ Please fill in both job title and description")
        else:
            st.session_state['job_requirements'] = {
                'category': target_category,
                'title': job_title,
                'description': job_description
            }
            st.success("✅ Job requirements saved!")

# Show saved requirements
if 'job_requirements' in st.session_state:
    req = st.session_state['job_requirements']
    st.info(f"**Current Job:** {req['title']} | **Category:** {req['category']}")

st.markdown("---")

# Candidate upload section
st.markdown("### 📤 Step 2: Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Upload all candidate resumes you want to rank"
)

if uploaded_files and 'job_requirements' in st.session_state:
    st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")

    if st.button("🚀 Rank Candidates", type="primary", use_container_width=True):
        req = st.session_state['job_requirements']
        candidates = []

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process each resume
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

            resume_text = extract_text_from_file(uploaded_file)

            if resume_text.strip():
                try:
                    result = predict_resume_category(resume_text)

                    # Calculate match score
                    match_score = calculate_match_score(
                        result['predicted_category'],
                        req['category'],
                        result['confidence_score']
                    )

                    candidates.append({
                        "filename": uploaded_file.name,
                        "predicted_category": result['predicted_category'],
                        "confidence": result['confidence_score'],
                        "match_score": match_score,
                        "result": result
                    })
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            else:
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        progress_bar.empty()
        status_text.empty()

        # Sort by match score
        candidates = sorted(candidates, key=lambda x: x['match_score'], reverse=True)

        # Display rankings
        if candidates:
            st.markdown("---")
            st.markdown("## 🏆 Candidate Rankings")

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Candidates", len(candidates))

            with col2:
                perfect_matches = sum(1 for c in candidates if c['predicted_category'] == req['category'])
                st.metric("Perfect Matches", perfect_matches)

            with col3:
                avg_match = sum(c['match_score'] for c in candidates) / len(candidates)
                st.metric("Avg Match Score", f"{avg_match:.1f}%")

            with col4:
                high_matches = sum(1 for c in candidates if c['match_score'] >= 70)
                st.metric("Strong Candidates", high_matches)

            st.markdown("---")

            # Display ranked candidates
            for rank, candidate in enumerate(candidates, 1):
                rank_class = get_rank_class(rank)
                score_class = get_score_class(candidate['match_score'])

                # Medal for top 3
                medal = ""
                if rank == 1:
                    medal = "🥇"
                elif rank == 2:
                    medal = "🥈"
                elif rank == 3:
                    medal = "🥉"

                with st.container():
                    st.markdown(f"""
                    <div class="rank-card {rank_class}">
                        <h3>{medal} #{rank} — {candidate['filename']}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**Match Score**")
                        st.markdown(f'<p class="{score_class}">{candidate["match_score"]:.1f}%</p>', unsafe_allow_html=True)

                    with col2:
                        st.markdown("**Predicted Category**")
                        category_match = "✅" if candidate['predicted_category'] == req['category'] else "❌"
                        st.markdown(f"{category_match} {candidate['predicted_category']}")

                    with col3:
                        st.markdown("**Confidence**")
                        st.markdown(f"{candidate['confidence']*100:.1f}%")

                    # Expandable details
                    with st.expander(f"📊 View Full Analysis"):
                        st.markdown("**Top Predicted Categories:**")
                        top_probs = sorted(
                            candidate['result']['probabilities'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]

                        for cat, prob in top_probs:
                            st.progress(prob, text=f"{cat}: {prob*100:.1f}%")

            # Export rankings
            st.markdown("---")
            st.markdown("### 📥 Export Rankings")

            export_data = []
            for rank, candidate in enumerate(candidates, 1):
                export_data.append({
                    "Rank": rank,
                    "Filename": candidate['filename'],
                    "Match Score": f"{candidate['match_score']:.1f}%",
                    "Predicted Category": candidate['predicted_category'],
                    "Confidence": f"{candidate['confidence']*100:.1f}%",
                    "Category Match": "Yes" if candidate['predicted_category'] == req['category'] else "No"
                })

            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Rankings (CSV)",
                data=csv,
                file_name=f"candidate_rankings_{req['category']}.csv",
                mime="text/csv",
                use_container_width=True
            )

elif uploaded_files and 'job_requirements' not in st.session_state:
    st.warning("⚠️ Please define job requirements first (Step 1)")

else:
    st.info("👆 Upload candidate resumes to start ranking")

# Back to home
st.markdown("---")
if st.button("🏠 Back to Home", use_container_width=True):
    st.switch_page("app.py")
