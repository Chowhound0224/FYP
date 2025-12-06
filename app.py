"""
Main Landing Page - Resume Screening System (HCI & Pastel Redesign)
"""

import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Candidate Finder",
    page_icon="✨",
    layout="centered", # Changed to centered for a focused, clean look
    initial_sidebar_state="collapsed" # Start collapsed for a cleaner main page
)

# ============================================================================
# CUSTOM CSS (HCI & PASTEL THEME)
# ============================================================================

# Define a soft, pastel palette
PASTEL_PURPLE = "#CDB4DB" # Light Lavender
PASTEL_PINK = "#FFC8DD"   # Soft Pink
PASTEL_BLUE = "#A2D2FF"   # Sky Blue
PASTEL_GREEN = "#BDECB4"  # Mint Green
TEXT_COLOR = "#333333"    # Dark Grey for readability
ACCENT_COLOR = "#6A4C93"  # Deeper purple for buttons/accents

st.markdown(f"""
<style>
    /* Overall Page Style */
    .stApp {{
        background-color: #FAFAFA; /* Very light neutral background */
        color: {TEXT_COLOR};
    }}

    /* Main title (Focus on Simplicity & Goal) */
    .main-title {{
        font-size: 42px !important;
        font-weight: 700;
        color: {ACCENT_COLOR}; /* Deeper, calming color */
        text-align: center;
        margin-bottom: 5px;
        padding-top: 10px;
    }}

    /* Subtitle (Clear value proposition) */
    .subtitle {{
        font-size: 18px;
        color: #777777;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }}

    /* Feature/Action Cards (HCI: Clear Call-to-Action) */
    .action-card {{
        background-color: {PASTEL_PINK}; /* Soft pastel background */
        border: 2px solid {PASTEL_PURPLE};
        padding: 30px 20px;
        border-radius: 12px;
        color: {TEXT_COLOR};
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); /* Soft shadow */
        transition: all 0.3s ease;
        height: 100%; /* Ensure equal height */
    }}

    .action-card:hover {{
        transform: translateY(-5px); /* Lift on hover */
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        background-color: {PASTEL_PURPLE}; /* Slight color shift */
    }}

    .action-icon {{
        font-size: 50px;
        margin-bottom: 15px;
        color: {ACCENT_COLOR};
    }}

    .action-title {{
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
        color: {ACCENT_COLOR};
    }}

    .action-desc {{
        font-size: 16px;
        color: {TEXT_COLOR};
        min-height: 40px; /* Consistent spacing */
    }}

    /* Streamlit Button Styling (Consistency) */
    .stButton>button {{
        background-color: {ACCENT_COLOR};
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
        background-color: #5A3F7A; /* Slightly darker hover */
    }}

    /* Info/How-It-Works Boxes */
    div[data-testid="stMarkdownContainer"]>div.info {{
        background-color: {PASTEL_BLUE}1A; /* Very light blue tint */
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

</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero Section (Focus: Clarity and Purpose)
st.markdown('<h1 class="main-title">✨ AI Candidate Finder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quickly identify the best candidates from your resume pool.</p>', unsafe_allow_html=True)

st.markdown("---")

# Feature Cards (HCI: Grouping related actions and immediate call to action)
st.markdown("### Choose Your Next Step")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="action-card">
        <div class="action-icon">📂</div>
        <div class="action-title">Screen Resumes</div>
        <div class="action-desc">
            Upload a batch of resumes (PDF, DOCX, TXT) and get instant categorization for easy sorting.
        </div>
        {st.button("📂 Start Screening", key="btn1", use_container_width=True)}
    </div>
    """, unsafe_allow_html=True)
    
    # Check if the button was clicked inside the markdown context for page switch
    if st.session_state.get('btn1'):
         st.switch_page("pages/1_📤_Resume_Upload.py")


with col2:
    st.markdown(f"""
    <div class="action-card">
        <div class="action-icon">⭐</div>
        <div class="action-title">Find Best Match</div>
        <div class="action-desc">
            Define your job requirements and rank your candidates based on an AI-powered matching score.
        </div>
        {st.button("⭐ Find Matches", key="btn2", use_container_width=True)}
    </div>
    """, unsafe_allow_html=True)
    
    # Check if the button was clicked inside the markdown context for page switch
    if st.session_state.get('btn2'):
        st.switch_page("pages/2_🎯_Job_Matching.py")

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
st.caption("AI Candidate Finder — Designed for a seamless recruitment experience.")



