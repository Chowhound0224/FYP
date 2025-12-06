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
import io
import zipfile

from typing import List, Tuple
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
    "AVIATION",
    "BUSINESS-DEVELOPMENT",
    "CHEF",
    "CONSTRUCTION",
    "CONSULTANT",
    "DESIGNER",
    "ENGINEERING",
    "FINANCE",
    "FITNESS",
    "HEALTHCARE",
    "HR",
    "INFORMATION-TECHNOLOGY",
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
                page_text = page.extract_text() or ""
                text += page_text + "\n"


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


def extract_resumes_from_zip(uploaded_zip) -> List[Tuple[str, str]]:

    """
    Extract multiple resumes from a ZIP file.

    Returns a list of (filename, extracted_text)
    Only processes .pdf, .docx, .txt inside the zip.
    """
    resumes = []

    try:
        with zipfile.ZipFile(uploaded_zip) as zf:
            for name in zf.namelist():
                # Skip folders
                if name.endswith("/"):
                    continue

                lower = name.lower()
                if not lower.endswith((".pdf", ".docx", ".txt")):
                    # Skip non-resume files
                    continue

                with zf.open(name) as f:
                    data = f.read()

                # Create a file-like object that looks like an uploaded file
                fake_file = io.BytesIO(data)
                fake_file.name = name  # So extract_text_from_file can read suffix
                fake_file.seek(0)  

                text = extract_text_from_file(fake_file)
                if text.strip():
                    resumes.append((name, text))

    except Exception as e:
        st.error(f"Error reading ZIP file: {str(e)}")

    return resumes



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
    col1, col2 = st.columns([1, 1])

    # ---- TARGET CATEGORY ----
    with col1:
        category_option = st.selectbox(
            "🎯 Target Job Category",
            ["-- Select Category --"] + JOB_CATEGORIES + ["Other (type manually)"],
            help="Choose a category or select 'Other' to enter your own."
        )

        # If user selects "Other", show text input
        if category_option == "Other (type manually)":
            custom_category = st.text_input(
                "🔤 Enter Custom Category",
                placeholder="e.g., DATA SCIENTIST, MARKETING"
            ).strip().upper()
            final_category = custom_category
        elif category_option != "-- Select Category --":
            final_category = category_option.upper()
        else:
            final_category = ""

    # ---- JOB TITLE ----
    with col2:
        job_title = st.text_input(
            "💼 Job Title",
            placeholder="e.g., Senior Software Engineer",
            help="Enter the job title for this role."
        ).strip()

    # ---- JOB DESCRIPTION ----
    job_description = st.text_area(
        "📝 Job Description & Requirements",
        height=180,
        placeholder=(
            "- Required skills\n"
            "- Responsibilities\n"
            "- Years of experience\n"
            "- Tools/technologies\n"
            "- Qualifications"
        )
    )

    # ---- SUBMIT BUTTON ----
    submitted = st.form_submit_button(
        "💾 Save Job Requirements",
        type="primary",
        use_container_width=True,
        key="save_job_requirements"
    )

    # ---- FORM VALIDATION ----
    if submitted:
        if not final_category:
            st.error("⚠️ Please select or enter a job category.")
        elif not job_title:
            st.error("⚠️ Please enter a job title.")
        elif not job_description.strip():
            st.error("⚠️ Please enter a job description.")
        else:
            st.session_state['job_requirements'] = {
                'category': final_category,
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
    "Upload candidate resumes (PDF, DOCX, TXT) or a ZIP containing many resumes",
    type=["pdf", "docx", "txt", "zip"],
    accept_multiple_files=True,
    help="You can drag & drop multiple files, including a ZIP with many resumes."
)

MAX_RESUMES = 500  # hard limit on how many resumes we process in total

all_resume_items = []  # list of dicts: {"filename": ..., "text": ...}

if uploaded_files:
    for uf in uploaded_files:
        name = uf.name.lower()

        # If it's a ZIP file, expand it
        if name.endswith(".zip"):
            extracted = extract_resumes_from_zip(uf)
            for fname, text in extracted:
                all_resume_items.append({"filename": fname, "text": text})
        else:
            # Normal single resume file
            text = extract_text_from_file(uf)
            if text.strip():
                all_resume_items.append({"filename": uf.name, "text": text})

    total_resumes = len(all_resume_items)

    if total_resumes == 0:
        st.warning("⚠️ No readable resumes found in the uploaded files.")
    else:
        # Enforce max limit
        if total_resumes > MAX_RESUMES:
            st.warning(
                f"⚠️ Found {total_resumes} resumes in total. "
                f"Only the first {MAX_RESUMES} will be processed for ranking."
            )
            all_resume_items = all_resume_items[:MAX_RESUMES]
            total_resumes = MAX_RESUMES

        st.success(f"✅ {total_resumes} resume(s) ready for analysis (limit: {MAX_RESUMES}).")


# ============================================================================
# STEP 3: RUN MATCHING & DISPLAY RESULTS
# ============================================================================

# We'll keep candidates in session_state so they persist after rerun
if "candidates" not in st.session_state:
    st.session_state["candidates"] = []

if all_resume_items and 'job_requirements' in st.session_state:
    if st.button("🚀 Rank Candidates", type="primary", use_container_width=True):
        req = st.session_state['job_requirements']
        candidates = []

        total = len(all_resume_items)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, item in enumerate(all_resume_items):
            filename = item["filename"]
            resume_text = item["text"]

            status_text.text(f"Analyzing {idx + 1}/{total}: {filename}")

            if resume_text.strip():
                try:
                    result = predict_resume_category(resume_text)

                    # ✅ Correct call: predicted, target, confidence
                    match_score = calculate_match_score(
                        predicted_category=result['predicted_category'].upper(),
                        target_category=req['category'].upper(),
                        confidence=result['confidence_score']
                    )

                    candidates.append({
                        "filename": filename,
                        "predicted_category": result['predicted_category'],
                        "confidence": result['confidence_score'],
                        "match_score": match_score,
                        "result": result,
                    })
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")
            else:
                st.warning(f"⚠️ Could not extract text from {filename}")

            progress_bar.progress((idx + 1) / total)

        progress_bar.empty()
        status_text.empty()

        # Sort candidates by score and save to session_state
        candidates = sorted(candidates, key=lambda x: x['match_score'], reverse=True)
        st.session_state["candidates"] = candidates


# ============================================================================
# DISPLAY RESULTS (IF ANY)
# ============================================================================

candidates = st.session_state.get("candidates", [])

if candidates and 'job_requirements' in st.session_state:
    req = st.session_state['job_requirements']

    st.markdown("---")
    st.markdown("## 🏆 Candidate Rankings")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Candidates", len(candidates))

    with col2:
        perfect_matches = sum(
            1 for c in candidates
            if c['predicted_category'].upper() == req['category'].upper()
        )
        st.metric("Perfect Matches", perfect_matches)

    with col3:
        avg_match = sum(c['match_score'] for c in candidates) / len(candidates)
        st.metric("Avg Match Score", f"{avg_match:.1f}%")

    with col4:
        high_matches = sum(1 for c in candidates if c['match_score'] >= 70)
        st.metric("Strong Candidates", high_matches)

    st.markdown("---")

    # Summary table
    summary_data = []
    for rank, candidate in enumerate(candidates, 1):
        summary_data.append({
            "Rank": rank,
            "Filename": candidate['filename'],
            "Match Score": f"{candidate['match_score']:.1f}%",
            "Predicted Category": candidate['predicted_category'],
            "Category Match": "Yes" if candidate['predicted_category'].upper() == req['category'].upper() else "No",
        })

    summary_df = pd.DataFrame(summary_data)

    def color_rows(row):
        score = float(row['Match Score'].replace('%', ''))
        if row['Category Match'] == 'Yes' or score >= 70:
            return ['background-color: #BDECB4'] * len(row)  # pastel green
        elif score >= 50:
            return ['background-color: #FFFACD'] * len(row)  # pale yellow
        else:
            return [''] * len(row)

    st.dataframe(
        summary_df.style.apply(color_rows, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Detailed Candidate Cards")

    # Detailed cards
    for rank, candidate in enumerate(candidates, 1):
        rank_class = get_rank_class(rank)
        score_class = get_score_class(candidate['match_score'])

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

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Match Score**")
                st.markdown(
                    f'<p class="{score_class}">{candidate["match_score"]:.1f}%</p>',
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown("**Predicted Category**")
                category_match = (
                    "✅" if candidate['predicted_category'].upper() == req['category'].upper() else "❌"
                )
                st.markdown(f"{category_match} {candidate['predicted_category']}")

            with c3:
                st.markdown("**Confidence**")
                st.markdown(f"{candidate['confidence']*100:.1f}%")

            with st.expander("📊 View Full Analysis"):
                st.markdown("**Top Predicted Categories:**")
                top_probs = sorted(
                    candidate['result']['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True,
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
            "Category Match": "Yes" if candidate['predicted_category'].upper() == req['category'].upper() else "No",
        })

    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Rankings (CSV)",
        data=csv,
        file_name=f"candidate_rankings_{req['category']}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("👆 Upload resumes and save job requirements to view rankings.")



# Back to home
st.markdown("---")
if st.button("🏠 Back to Home", use_container_width=True):
    st.switch_page("app.py")
