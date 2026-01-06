"""
Main Landing Page - Resume Screening System (Minimalist & Pastel Redesign)
"""

import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TalentLens Resume Screening System",
    page_icon="✨",
    layout="centered", # Changed to centered for a focused, clean look
    initial_sidebar_state="collapsed" # Start collapsed for a cleaner main page
)

# ============================================================================
# CUSTOM CSS (HCI & MINIMALIST THEME)
# ============================================================================

# Define a soft, pastel palette
PASTEL_PURPLE = "#CDB4DB" # Light Lavender (Used lightly, mainly for hover/shadow)
PASTEL_BLUE = "#A2D2FF"   # Sky Blue (Used lightly for info boxes)
TEXT_COLOR = "#FFFFFF"    # White text for dark background
ACCENT_COLOR = "#A2D2FF"  # Light blue for accents on dark background
BG_COLOR = "#000000"      # Black background

st.markdown(f"""
<style>
    /* ========== DARK MODE (DEFAULT) ========== */
    /* Overall Page Style */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}

    /* Main title (Focus on Simplicity & Goal) */
    .main-title {{
        font-size: 42px !important;
        font-weight: 700;
        color: {ACCENT_COLOR};
        text-align: center;
        margin-bottom: 5px;
        padding-top: 10px;
    }}

    /* Subtitle (Clear value proposition) */
    .subtitle {{
        font-size: 18px;
        color: #CCCCCC;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }}

    /* Feature/Action Cards (HCI: Use NEUTRALS for background) */
    .action-card {{
        background-color: #1a1a1a; /* Dark gray for cards */
        border: 1px solid #333333; /* Dark border */
        padding: 30px 20px;
        border-radius: 12px;
        color: {TEXT_COLOR};
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 2px 5px rgba(255,255,255,0.1); /* Light shadow for dark theme */
        height: 100%;
        cursor: default; /* Not clickable */
    }}

    .action-icon {{
        font-size: 50px;
        margin-bottom: 15px;
        color: {ACCENT_COLOR}; /* Keep icon colored for visibility */
    }}

    .action-title {{
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
        color: {TEXT_COLOR}; /* Use dark text for title, not the accent color */
    }}

    .action-desc {{
        font-size: 16px;
        color: #CCCCCC; /* Light gray for description on dark background */
        min-height: 40px;
    }}

    /* Streamlit Button Styling (Consistency) */
    .stButton>button {{
        background-color: #4A5FC1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        transition: background-color 0.3s;
        margin-top: 15px;
    }}

    .stButton>button:hover {{
        background-color: #3A4FA1;
    }}

    /* Info/How-It-Works Boxes (Keep light pastel for informative blocks) */
    div[data-testid="stMarkdownContainer"]>div.info {{
        background-color: {PASTEL_BLUE}1A;
        border-left: 5px solid {PASTEL_BLUE};
        color: {TEXT_COLOR};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}

    /* Center headings */
    h3 {{
        text-align: center;
        color: {ACCENT_COLOR};
        margin-top: 40px;
    }}

    /* ========== LIGHT MODE ========== */
    @media (prefers-color-scheme: light) {{
        .stApp {{
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }}

        .main-title {{
            color: #4A5FC1 !important;
        }}

        .subtitle {{
            color: #666666 !important;
        }}

        .action-card {{
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            color: #000000 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            cursor: default !important; /* Not clickable */
        }}

        .action-icon {{
            color: #4A5FC1 !important;
        }}

        .action-title {{
            color: #000000 !important;
        }}

        .action-desc {{
            color: #666666 !important;
        }}

        h3 {{
            color: #4A5FC1 !important;
        }}

        /* Ensure all text is visible in light mode */
        .stApp p, .stApp span, .stApp div {{
            color: #000000;
        }}

        /* Override Streamlit's default styles for light mode */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] div {{
            color: #000000 !important;
        }}

        /* Buttons in light mode */
        .stButton>button {{
            background-color: #4A5FC1 !important;
            color: #FFFFFF !important;
        }}

        .stButton>button:hover {{
            background-color: #3A4FA1 !important;
        }}

        /* Info boxes in light mode */
        .stAlert {{
            background-color: #E8F4FD !important;
            border: 1px solid #4A5FC1 !important;
            color: #000000 !important;
        }}

        /* Captions in light mode */
        .stCaption {{
            color: #666666 !important;
        }}
    }}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero Section (Focus: Clarity and Purpose)
st.markdown('<h1 class="main-title">✨ TalentLens Resume Screening System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quickly identify the best candidates from your resume pool.</p>', unsafe_allow_html=True)

st.markdown("---")

# Feature Cards (Action Grouping)
st.markdown("### Choose Your Next Step")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="action-card">
        <div class="action-icon">📂</div>
        <div class="action-title">Screen Candidates</div>
        <div class="action-desc">
            Upload a batch of resumes and get instant AI categorization for easy sorting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Button placed outside the HTML markdown
    if st.button("📂 Screen Candidates", key="btn1", use_container_width=True):
        st.switch_page("pages/1_📤_Screen_Candidates.py")


with col2:
    st.markdown("""
    <div class="action-card">
        <div class="action-icon">⭐</div>
        <div class="action-title">Rank Candidates</div>
        <div class="action-desc">
            Define job requirements and rank candidates based on an AI-powered matching score.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Button placed outside the HTML markdown
    if st.button("⭐ Rank Candidates", key="btn2", use_container_width=True):
        st.switch_page("pages/2_🎯_Rank_Candidates.py")

# HCI Principle: Visibility and Feedback
st.markdown("---")
st.markdown("### 💡 How It Works")


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1. Upload")
    st.info("Simply drag and drop your resume files. Our system handles PDF, DOCX, or TXT.")

with col2:
    st.markdown("#### 2. Analyze")
    st.info("The system processes the text to understand skills, experience, and roles.")

with col3:
    st.markdown("#### 3. Rank")
    st.info("Receive a score and ranking to show you the most relevant candidates first.")

# Footer (Less prominent, clean)
st.markdown("---")
st.caption("TalentLens Resume Screening System — Designed for a seamless recruitment experience.")