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

from sklearn.metrics.pairwise import cosine_similarity
from src.prediction.predict import predict_resume_category
from src.features.build_features import load_sbert_model

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Job Matching",
    page_icon="🎯",
    layout="wide"
)

# ============================================================================
# JOB CATEGORIES (17 categories - matches trained model)
# ============================================================================
JOB_CATEGORIES = [
    "ACCOUNTANT",
    "ADVOCATE",
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
    "PUBLIC-RELATIONS",    # Added - was missing!
    "SALES",
    "TEACHER"
]

# ============================================================================
# CUSTOM CSS WITH DARK/LIGHT MODE SUPPORT
# ============================================================================
st.markdown("""
<style>
    /* ========== DARK MODE (DEFAULT) ========== */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #A2D2FF;
    }

    .rank-card {
        background: #1a1a1a;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        border: 1px solid #333333;
    }

    .rank-gold {
        border-left-color: #FFD700 !important;
        background: linear-gradient(135deg, #2a2200 0%, #1a1a1a 100%);
    }

    .rank-silver {
        border-left-color: #C0C0C0 !important;
        background: linear-gradient(135deg, #252525 0%, #1a1a1a 100%);
    }

    .rank-bronze {
        border-left-color: #CD7F32 !important;
        background: linear-gradient(135deg, #2a1a00 0%, #1a1a1a 100%);
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

    /* ========== LIGHT MODE ========== */
    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }

        h1, h2, h3 {
            color: #4A5FC1 !important;
        }

        .rank-card {
            background: #FFFFFF !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            border: 1px solid #E0E0E0 !important;
        }

        .rank-gold {
            background: linear-gradient(135deg, #FFF9E6 0%, #FFFFFF 100%) !important;
        }

        .rank-silver {
            background: linear-gradient(135deg, #F5F5F5 0%, #FFFFFF 100%) !important;
        }

        .rank-bronze {
            background: linear-gradient(135deg, #FFF5E6 0%, #FFFFFF 100%) !important;
        }

        /* Ensure all text is visible in light mode */
        .stApp p, .stApp span, .stApp div {
            color: #000000;
        }

        /* Override Streamlit's default styles for light mode */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] div {
            color: #000000 !important;
        }

        /* Info/Warning/Success boxes in light mode */
        .stAlert {
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
        }

        /* Form elements in light mode */
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox select {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #CCCCCC !important;
        }

        /* Buttons in light mode */
        .stButton>button {
            background-color: #4A5FC1 !important;
            color: #FFFFFF !important;
        }

        .stButton>button:hover {
            background-color: #3A4FA1 !important;
        }

        /* Metrics in light mode */
        [data-testid="stMetricValue"] {
            color: #000000 !important;
        }

        /* Tables in light mode */
        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
        }

        /* Expander in light mode */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
        }

        /* Progress bars in light mode */
        .stProgress > div > div {
            background-color: #4A5FC1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT."""
    suffix = uploaded_file.name.lower()
    filename = uploaded_file.name

    try:
        if suffix.endswith(".pdf"):
            try:
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

                # Check if file is empty
                if not text.strip():
                    st.warning(f"⚠️ **{filename}** appears to be empty or contains no readable text.")
                    return ""

                return text
            except Exception as pdf_error:
                st.error(f"❌ **{filename}** - Corrupted or invalid PDF file. Error: {str(pdf_error)}")
                return ""

        elif suffix.endswith(".docx"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                text = docx2txt.process(tmp_path)

                # Check if file is empty
                if not text or not text.strip():
                    st.warning(f"⚠️ **{filename}** appears to be empty or contains no readable text.")
                    return ""

                return text
            except Exception as docx_error:
                st.error(f"❌ **{filename}** - Corrupted or invalid DOCX file. Error: {str(docx_error)}")
                return ""

        elif suffix.endswith(".txt"):
            try:
                text = uploaded_file.read().decode("utf-8")

                # Check if file is empty
                if not text.strip():
                    st.warning(f"⚠️ **{filename}** appears to be empty.")
                    return ""

                return text
            except UnicodeDecodeError:
                st.error(f"❌ **{filename}** - Unable to decode text file. File may be corrupted or in an unsupported encoding.")
                return ""
            except Exception as txt_error:
                st.error(f"❌ **{filename}** - Error reading text file. Error: {str(txt_error)}")
                return ""

        else:
            st.error(f"❌ **{filename}** - Unsupported file format.")
            return ""

    except Exception as e:
        st.error(f"❌ **{filename}** - Unexpected error occurred. Error: {str(e)}")
        return ""





def calculate_match_score(
    predicted_category: str,
    target_category: str,
    confidence: float,
    keyword_ratio: float = 0.0,
    sbert_similarity: float = 0.0
) -> dict:
    """
    Calculate hybrid matching score with breakdown.

    Scoring weights:
    - Category match: 30%
    - Keyword matches: 30%
    - SBERT similarity: 30%
    - Model confidence: 10%

    Returns:
        Dict with total score and breakdown
    """
    # Component 1: Category Match (30%)
    if predicted_category == target_category:
        category_score = 30.0  # Perfect match
    else:
        category_score = 0.0  # Mismatch

    # Component 2: Keyword Match (30%)
    keyword_score = keyword_ratio * 30.0

    # Component 3: SBERT Similarity (30%)
    similarity_score = sbert_similarity * 30.0

    # Component 4: Model Confidence (10%)
    confidence_score = confidence * 10.0

    # Total score
    total_score = category_score + keyword_score + similarity_score + confidence_score

    return {
        'total': total_score,
        'category': category_score,
        'keywords': keyword_score,
        'similarity': similarity_score,
        'confidence': confidence_score
    }


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


def find_matched_keywords(resume_text: str, keywords: list) -> list:
    """
    Find which keywords from the job requirements appear in the resume text.

    Args:
        resume_text: The candidate's resume text
        keywords: List of lowercase keywords to search for

    Returns:
        List of matched keywords (lowercase)
    """
    if not keywords:
        return []

    resume_lower = resume_text.lower()
    matched = []

    for keyword in keywords:
        # Check if keyword appears as a whole word or phrase
        if keyword in resume_lower:
            matched.append(keyword)

    return matched


# Cache SBERT model at module level
@st.cache_resource
def get_sbert_model():
    """Load and cache SBERT model."""
    return load_sbert_model()


def calculate_sbert_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using SBERT embeddings.

    Args:
        text1: First text (e.g., job description)
        text2: Second text (e.g., resume)

    Returns:
        Similarity score between 0 and 1
    """
    try:
        model = get_sbert_model()

        # Generate embeddings
        embedding1 = model.encode([text1])
        embedding2 = model.encode([text2])

        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]

        return float(similarity)
    except Exception as e:
        st.warning(f"Error calculating similarity: {str(e)}")
        return 0.0


# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<h1 style="color: #667eea;">🎯 Job Matching & Candidate Ranking</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="job-form">
    <h2 style="margin-top: 0; color: #FFFFFF;">Step 1: Define Job Requirements</h2>
    <p style="color: #FFFFFF;">Enter the target job category and describe requirements</p>
</div>
""", unsafe_allow_html=True)

# Job requirements form
with st.form("job_form"):
    # ---- TARGET CATEGORY ----
    st.markdown("### 🎯 Target Job Category")

    col1, col2 = st.columns([1, 2])

    with col1:
        category_choice = st.selectbox(
            "Choose from predefined or enter custom",
            ["-- Select from list or enter custom below --"] + JOB_CATEGORIES,
            label_visibility="collapsed"
        )

    with col2:
        custom_input = st.text_input(
            "Or enter custom category",
            placeholder="e.g., DATA-SCIENTIST, MARKETING, CYBERSECURITY",
            help="Select from dropdown OR type your own category here",
            label_visibility="collapsed"
        ).strip().upper()

    # Determine final category with validation
    if custom_input:
        # User typed something custom - use that
        final_category = custom_input
    elif category_choice != "-- Select from list or enter custom below --":
        # User selected from dropdown
        final_category = category_choice.upper()
    else:
        # Nothing selected or entered
        final_category = ""

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

    # ---- KEYWORDS ----
    st.markdown("### 🔑 Required Keywords (Optional)")
    keywords_input = st.text_input(
        "Enter comma-separated keywords to match",
        placeholder="e.g., python, machine learning, AWS, SQL, leadership",
        help="Keywords will be matched against candidate resumes to show relevance"
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
            st.error("⚠️ Please select a category from the dropdown or enter a custom category.")
        elif len(final_category) < 2:
            st.error("⚠️ Category name must be at least 2 characters long.")
        elif not all(c.isalpha() or c in ['-', '_', ' '] for c in final_category):
            st.error("⚠️ Category can only contain letters, hyphens, underscores, and spaces (no numbers).")
        elif not job_description.strip():
            st.error("⚠️ Please enter a job description.")
        else:
            # Parse keywords
            keywords = []
            if keywords_input.strip():
                keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]

            st.session_state['job_requirements'] = {
                'category': final_category,
                'description': job_description,
                'keywords': keywords
            }

            keyword_msg = f" with {len(keywords)} keyword(s)" if keywords else ""
            st.success(f"✅ Job requirements saved! Target category: **{final_category}**{keyword_msg}")

# Show saved requirements
if 'job_requirements' in st.session_state:
    req = st.session_state['job_requirements']
    keywords = req.get('keywords', [])
    keyword_display = f", **Keywords:** {', '.join(keywords)}" if keywords else ""
    st.info(f"**Target Category:** {req['category']}{keyword_display}")

st.markdown("---")

# Candidate upload section
st.markdown("### 📤 Step 2: Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="You can drag & drop multiple files at once. Supported formats: PDF, DOCX, TXT"
)

MAX_RESUMES = 500  # hard limit on how many resumes we process in total

all_resume_items = []  # list of dicts: {"filename": ..., "text": ...}
skipped_files_upload = []  # Track corrupted/empty files during upload

if uploaded_files:
    for uf in uploaded_files:
        # Process individual resume files
        text = extract_text_from_file(uf)
        if text.strip():
            all_resume_items.append({"filename": uf.name, "text": text})
        else:
            # File is corrupted or empty
            skipped_files_upload.append(uf.name)

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

        # Show summary with skipped files info
        if skipped_files_upload:
            st.warning(f"⚠️ **{len(skipped_files_upload)} file(s) skipped** due to corruption or empty content: {', '.join(skipped_files_upload)}")

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
        skipped_files_ranking = []  # Track files that failed during ranking

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

                    # Find matched keywords
                    keywords = req.get('keywords', [])
                    matched_keywords = find_matched_keywords(resume_text, keywords)
                    keyword_ratio = len(matched_keywords) / len(keywords) if keywords else 0.0

                    # Calculate SBERT similarity
                    job_description = req['description']
                    sbert_similarity = calculate_sbert_similarity(job_description, resume_text)

                    # Calculate hybrid match score
                    score_breakdown = calculate_match_score(
                        predicted_category=result['predicted_category'].upper(),
                        target_category=req['category'].upper(),
                        confidence=result['confidence_score'],
                        keyword_ratio=keyword_ratio,
                        sbert_similarity=sbert_similarity
                    )

                    candidates.append({
                        "filename": filename,
                        "predicted_category": result['predicted_category'],
                        "confidence": result['confidence_score'],
                        "match_score": score_breakdown['total'],
                        "score_breakdown": score_breakdown,
                        "result": result,
                        "matched_keywords": matched_keywords,
                        "total_keywords": len(keywords),
                        "sbert_similarity": sbert_similarity,
                    })
                except Exception as e:
                    st.error(f"❌ Error processing {filename}: {str(e)}")
                    skipped_files_ranking.append(filename)
            else:
                skipped_files_ranking.append(filename)

            progress_bar.progress((idx + 1) / total)

        progress_bar.empty()
        status_text.empty()

        # Show summary of processing
        if skipped_files_ranking:
            st.warning(f"⚠️ **{len(skipped_files_ranking)} file(s) skipped** during analysis due to errors: {', '.join(skipped_files_ranking)}")

        if candidates:
            st.success(f"✅ **{len(candidates)} candidate(s) successfully ranked** out of {total} resumes.")

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
            return ['background-color: #0A9548'] * len(row)  # vibrant green
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
                # Show score breakdown
                if 'score_breakdown' in candidate:
                    breakdown = candidate['score_breakdown']
                    st.markdown("**📈 Score Breakdown (Hybrid Scoring):**")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Category Match (30%)", f"{breakdown['category']:.1f}")
                        st.metric("Keywords (30%)", f"{breakdown['keywords']:.1f}")

                    with col2:
                        st.metric("Content Similarity (30%)", f"{breakdown['similarity']:.1f}")
                        st.metric("Model Confidence (10%)", f"{breakdown['confidence']:.1f}")

                    st.markdown(f"**Total Score: {breakdown['total']:.1f}/100**")
                    st.markdown("---")

                # Show matched keywords if any were specified
                if candidate.get('total_keywords', 0) > 0:
                    matched = candidate.get('matched_keywords', [])
                    total = candidate['total_keywords']
                    match_count = len(matched)

                    st.markdown("**🔑 Keyword Matches:**")
                    if matched:
                        keyword_tags = " ".join([f"`{kw}`" for kw in matched])
                        st.markdown(f"✅ **{match_count}/{total}** keywords found: {keyword_tags}")
                    else:
                        st.markdown(f"❌ **0/{total}** keywords found")

                    st.markdown("---")

                # Show SBERT similarity
                if 'sbert_similarity' in candidate:
                    sim_percent = candidate['sbert_similarity'] * 100
                    st.markdown(f"**🔍 Content Similarity:** {sim_percent:.1f}% (resume vs. job description)")
                    st.markdown("---")

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
        # Prepare keyword info for export
        keyword_info = ""
        if candidate.get('total_keywords', 0) > 0:
            matched = candidate.get('matched_keywords', [])
            keyword_info = f"{len(matched)}/{candidate['total_keywords']} ({', '.join(matched) if matched else 'none'})"

        export_data.append({
            "Rank": rank,
            "Filename": candidate['filename'],
            "Match Score": f"{candidate['match_score']:.1f}%",
            "Predicted Category": candidate['predicted_category'],
            "Confidence": f"{candidate['confidence']*100:.1f}%",
            "Category Match": "Yes" if candidate['predicted_category'].upper() == req['category'].upper() else "No",
            "Keywords Matched": keyword_info if keyword_info else "N/A",
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
