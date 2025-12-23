"""
Page 1: Resume Upload & Prediction
Drag and drop multiple resumes to get AI predictions
"""

import streamlit as st
import tempfile
import docx2txt
from pathlib import Path
from PyPDF2 import PdfReader
import pandas as pd

from src.prediction.predict import predict_resume_category

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄", # Changed icon to a generic document for screening
    layout="wide"
)

# ============================================================================
# CUSTOM CSS (PASTEL & PROFESSIONAL)
# ============================================================================

# Define a soft, pastel palette matching the main page
PASTEL_PURPLE = "#CDB4DB" # Light Lavender
PASTEL_PINK = "#FFC8DD"   # Soft Pink
PASTEL_BLUE = "#A2D2FF"   # Sky Blue
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

    /* Title */
    h1 {{
        color: {ACCENT_COLOR};
        text-align: center;
        font-weight: 700;
        margin-bottom: 5px;
    }}

    h3 {{
        color: {ACCENT_COLOR};
    }}

    /* Upload Section (Softened look) */
    .upload-section {{
        background: #1a1a1a; /* Dark gray for section */
        padding: 30px;
        border-radius: 12px;
        color: {TEXT_COLOR};
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(255,255,255,0.1); /* Light shadow for dark theme */
        border: 1px solid #333333;
    }}

    .upload-section h2 {{
        color: {TEXT_COLOR} !important;
        font-weight: 600;
        margin: 0;
    }}

    /* Result Card (Clean and Clear) */
    .result-card {{
        background: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid {ACCENT_COLOR}; /* Accent border */
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        transition: transform 0.2s;
        border: 1px solid #333333;
    }}

    .result-card:hover {{
        transform: scale(1.005);
        border-left-color: {PASTEL_PINK}; /* Subtle hover feedback */
    }}

    /* Category Badge (Clear and distinct) */
    .category-badge {{
        background: {ACCENT_COLOR};
        color: #000000;
        padding: 6px 15px;
        border-radius: 6px; /* Boxier for a more professional look */
        font-weight: 600;
        font-size: 16px;
        display: inline-block;
        margin: 5px 0 15px 0;
    }}

    /* Confidence Indicators (Using semantic colors) */
    .confidence-high {{
        color: #28a745; /* Green */
        font-weight: 600;
    }}

    .confidence-medium {{
        color: #ffc107; /* Yellow/Orange */
        font-weight: 600;
    }}

    .confidence-low {{
        color: #dc3545; /* Red */
        font-weight: 600;
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

    /* File Uploader Enhancement for Multi-file Drop */
    [data-testid="stFileUploader"] {{
        padding: 20px;
        border: 2px dashed {ACCENT_COLOR};
        border-radius: 12px;
        background: #1a1a1a;
        transition: all 0.3s ease;
    }}

    [data-testid="stFileUploader"]:hover {{
        border-color: {PASTEL_PINK};
        background: #222222;
        transform: scale(1.01);
    }}

    [data-testid="stFileUploader"] section {{
        padding: 10px;
    }}

    /* Make drop zone text more visible */
    [data-testid="stFileUploader"] label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }}

    /* ========== LIGHT MODE ========== */
    @media (prefers-color-scheme: light) {{
        .stApp {{
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }}

        h1, h2, h3 {{
            color: #4A5FC1 !important;
        }}

        .upload-section {{
            background: #FFFFFF !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            border: 1px solid #E0E0E0 !important;
        }}

        .upload-section h2 {{
            color: #000000 !important;
        }}

        .result-card {{
            background: #FFFFFF !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            border: 1px solid #E0E0E0 !important;
            border-left: 5px solid #4A5FC1 !important;
        }}

        .result-card:hover {{
            border-left-color: #FF69B4 !important;
        }}

        .category-badge {{
            background: #4A5FC1 !important;
            color: #FFFFFF !important;
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

        /* File uploader in light mode */
        [data-testid="stFileUploader"] {{
            background: #FFFFFF !important;
            border: 2px dashed #4A5FC1 !important;
        }}

        [data-testid="stFileUploader"]:hover {{
            border-color: #FF69B4 !important;
            background: #F9F9F9 !important;
        }}

        [data-testid="stFileUploader"] label {{
            color: #4A5FC1 !important;
        }}

        /* Buttons in light mode */
        .stButton>button {{
            background-color: #4A5FC1 !important;
            color: #FFFFFF !important;
        }}

        .stButton>button:hover {{
            background-color: #3A4FA1 !important;
        }}

        /* Info/Warning/Success boxes in light mode */
        .stAlert {{
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
        }}

        /* Form elements in light mode */
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox select {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #CCCCCC !important;
        }}

        /* Metrics in light mode */
        [data-testid="stMetricValue"] {{
            color: #000000 !important;
        }}

        /* Tables in light mode */
        [data-testid="stDataFrame"] {{
            background-color: #FFFFFF !important;
        }}

        /* Expander in light mode */
        [data-testid="stExpander"] {{
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
        }}

        /* Progress bars in light mode */
        .stProgress > div > div {{
            background-color: #4A5FC1 !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS (No major changes required, only slight adjustments to extraction error handling)
# ============================================================================
# ... (extract_text_from_file remains the same, adjusted error message slightly) ...
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




def get_confidence_class(confidence: float) -> str:
    """Return CSS class based on confidence level."""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_prediction_card(filename: str, result: dict, index: int):
    """Display a single prediction result as a clean card."""
    confidence = result['confidence_score']
    confidence_class = get_confidence_class(confidence)
    predicted_category = result['predicted_category']

    # Determine border and background colors based on confidence
    if confidence >= 0.70:
        border_color = "#28a745"  # Green
        bg_color = "#1a2e1a"  # Dark green tint
        badge_color = "#28a745"
    elif confidence >= 0.50:
        border_color = "#ffc107"  # Yellow
        bg_color = "#2e2a1a"  # Dark yellow tint
        badge_color = "#ffc107"
    else:
        border_color = "#dc3545"  # Red
        bg_color = "#2e1a1a"  # Dark red tint
        badge_color = "#dc3545"

    with st.container():
        st.markdown(f"""
        <div class="result-card" style="border-left: 5px solid {border_color}; background: {bg_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {TEXT_COLOR};">📄 <strong>{filename}</strong></h4>
                <div style="text-align: right;">
                    <span style="font-size: 14px; color: #999999;">AI Category:</span>
                    <span class="category-badge">{predicted_category}</span>
                </div>
            </div>
                
        </div>
        """, unsafe_allow_html=True)

        # Expandable details
        with st.expander(f"📊 Detailed Analysis for {filename}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top 5 Category Scores:**")
                
                # Filter, sort, and display probabilities neatly
                top_probs = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Check if the highest confidence score is exactly the one predicted
                for cat, prob in top_probs[:5]:
                    # Use BOLD for the predicted category for visual confirmation
                    text_label = f"**{cat}**" if cat == predicted_category else cat
                    st.progress(prob, text=f"{text_label}: {prob*100:.1f}%")

            with col2:
                st.markdown("**Text Extraction Preview:**")
                preview = result['cleaned_text'][:700] + "..." if len(result['cleaned_text']) > 700 else result['cleaned_text']
                st.text_area("Extracted Content", preview, height=200, disabled=True, label_visibility="collapsed", key=f"preview_{index}")
                
# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<h1 style="color: #667eea;">📄 Resume Screening</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-section">
    <h2 style="margin: 0;">Upload Resumes for Instant Categorization</h2>
    <p style="margin-top: 10px; opacity: 0.9;">
        <strong>Drag & drop multiple files at once</strong> (PDF, DOCX, or TXT) or click browse to select files.<br>
        <em style="font-size: 14px; color: #CCCCCC;">💡 Tip: Select multiple files by holding Ctrl (Windows) or Cmd (Mac) when browsing, or drag them all together!</em>
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader with drag & drop - Configured for multiple files
uploaded_files = st.file_uploader(
    "📁 Drop multiple resume files here or click to browse",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple files at once. Supported formats: PDF, DOCX, TXT"
)

if uploaded_files:
    st.success(f"✅ **{len(uploaded_files)}** file(s) uploaded successfully! Click 'Analyze' to start.")

    # Process button
    if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
        results = []
        skipped_files = []  # Track corrupted/empty files

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

            resume_text = extract_text_from_file(uploaded_file)

            if resume_text.strip():
                try:
                    # Use the actual AI model for prediction
                    result = predict_resume_category(resume_text)

                    results.append({
                        "filename": uploaded_file.name,
                        "result": result
                    })
                except Exception as e:
                    st.error(f"❌ Error predicting {uploaded_file.name}: The prediction model failed. ({str(e)})")
                    skipped_files.append(uploaded_file.name)
            else:
                # File is corrupted or empty - already logged in extract_text_from_file
                skipped_files.append(uploaded_file.name)

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()

        # Show summary of skipped files
        if skipped_files:
            st.warning(f"⚠️ **{len(skipped_files)} file(s) skipped** due to corruption, empty content, or errors: {', '.join(skipped_files)}")
            st.info(f"📊 **{len(results)} file(s) successfully analyzed** out of {len(uploaded_files)} uploaded.")

        # Display results
        if results:
            st.markdown("---")
            st.markdown("## 📊 Analysis Summary")

            # Summary statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_confidence = sum(r['result']['confidence_score'] for r in results) / len(results)
                st.metric("Average AI Certainty", f"{avg_confidence*100:.1f}%")

            with col2:
                categories = set(r['result']['predicted_category'] for r in results)
                st.metric("Unique Categories Found", len(categories))

            with col3:
                high_conf = sum(1 for r in results if r['result']['confidence_score'] >= 0.7)
                st.metric("Strong Classifications", f"{high_conf}/{len(results)}")

            st.markdown("---")
            st.markdown("## 📑 Individual Results")

            # Display individual results
            for idx, item in enumerate(results, 1):
                display_prediction_card(item['filename'], item['result'], idx)

            # Download results as CSV
            st.markdown("---")
            st.markdown("### 📥 Export Results")

            export_data = []
            for item in results:
                export_data.append({
                    "Filename": item['filename'],
                    "Predicted Category": item['result']['predicted_category'],
                    "Confidence": f"{item['result']['confidence_score']*100:.2f}%"
                })

            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="resume_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👆 Upload one or more resume files to get started")

# Back to home
st.markdown("---")
if st.button("🏠 Back to Home", use_container_width=True):
    st.switch_page("app.py")
