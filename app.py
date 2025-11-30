"""
Main Landing Page - Resume Screening System
"""

import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 48px !important;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 40px;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    .feature-icon {
        font-size: 60px;
        margin-bottom: 15px;
    }

    .feature-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .feature-desc {
        font-size: 16px;
        opacity: 0.9;
    }

    /* Stats */
    .stat-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }

    .stat-number {
        font-size: 36px;
        font-weight: 800;
        color: #667eea;
    }

    .stat-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero Section
st.markdown('<h1 class="main-title">🎯 AI-Powered Resume Screening System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent candidate matching using SBERT + TF-IDF + Custom Features</p>', unsafe_allow_html=True)

st.markdown("---")

# Feature Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-title">Upload Resumes</div>
        <div class="feature-desc">
            Drag & drop multiple resumes (PDF, DOCX, TXT)<br>
            Get instant AI-powered category predictions
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📤 Go to Resume Upload", key="btn1", use_container_width=True):
        st.switch_page("pages/1_📤_Resume_Upload.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💼</div>
        <div class="feature-title">Job Matching</div>
        <div class="feature-desc">
            Select job category & describe requirements<br>
            Rank candidates by AI-powered matching score
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🎯 Go to Job Matching", key="btn2", use_container_width=True):
        st.switch_page("pages/2_🎯_Job_Matching.py")

st.markdown("---")

# Statistics
st.markdown("### 📊 System Capabilities")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">24</div>
        <div class="stat-label">Job Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">75-85%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">5391</div>
        <div class="stat-label">Features</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">3</div>
        <div class="stat-label">AI Models</div>
    </div>
    """, unsafe_allow_html=True)

# How it works
st.markdown("---")
st.markdown("### 🔍 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1️⃣ Upload Resume")
    st.info("Upload PDF, DOCX, or TXT files. System extracts text automatically.")

with col2:
    st.markdown("#### 2️⃣ AI Analysis")
    st.info("SBERT embeddings + TF-IDF + custom features analyze the resume.")

with col3:
    st.markdown("#### 3️⃣ Get Results")
    st.info("Instant category prediction with confidence scores and rankings.")

# Footer
st.markdown("---")
st.caption("Developed by Chin Pei Fung — Final Year Project 2025")
