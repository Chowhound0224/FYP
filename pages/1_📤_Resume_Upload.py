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
    page_title="Resume Upload",
    page_icon="📤",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }

    .result-card {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .category-badge {
        background: #667eea;
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 18px;
        display: inline-block;
        margin: 10px 0;
    }

    .confidence-high {
        color: #28a745;
        font-weight: 700;
    }

    .confidence-medium {
        color: #ffc107;
        font-weight: 700;
    }

    .confidence-low {
        color: #dc3545;
        font-weight: 700;
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
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
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
    """Display a single prediction result as a card."""
    confidence = result['confidence_score']
    confidence_class = get_confidence_class(confidence)

    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <h3>📄 {filename}</h3>
            <span class="category-badge">{result['predicted_category']}</span>
            <p>
                <strong>Confidence:</strong>
                <span class="{confidence_class}">{confidence * 100:.2f}%</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Expandable details
        with st.expander(f"📊 View Details for {filename}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top 5 Predictions:**")
                top_probs = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for cat, prob in top_probs:
                    st.progress(prob, text=f"{cat}: {prob*100:.1f}%")

            with col2:
                st.markdown("**Cleaned Text Preview:**")
                preview = result['cleaned_text'][:500] + "..." if len(result['cleaned_text']) > 500 else result['cleaned_text']
                st.text_area("", preview, height=200, disabled=True, label_visibility="collapsed")

# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<h1 style="color: #667eea;">📤 Resume Upload & Analysis</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-section">
    <h2 style="margin: 0;">Drag & Drop Your Resumes</h2>
    <p style="margin-top: 10px; opacity: 0.9;">
        Upload multiple PDF, DOCX, or TXT files for instant AI-powered classification
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader with drag & drop
uploaded_files = st.file_uploader(
    "Choose resume files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

    # Process button
    if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
        results = []

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

            resume_text = extract_text_from_file(uploaded_file)

            if resume_text.strip():
                try:
                    result = predict_resume_category(resume_text)
                    results.append({
                        "filename": uploaded_file.name,
                        "result": result
                    })
                except Exception as e:
                    st.error(f"Error predicting {uploaded_file.name}: {str(e)}")
            else:
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")

            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()

        # Display results
        if results:
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # Summary statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_confidence = sum(r['result']['confidence_score'] for r in results) / len(results)
                st.metric("Average Confidence", f"{avg_confidence*100:.1f}%")

            with col2:
                categories = set(r['result']['predicted_category'] for r in results)
                st.metric("Unique Categories", len(categories))

            with col3:
                high_conf = sum(1 for r in results if r['result']['confidence_score'] >= 0.7)
                st.metric("High Confidence", f"{high_conf}/{len(results)}")

            st.markdown("---")

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
