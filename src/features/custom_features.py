"""Extract custom handcrafted features from resume text."""

import re
from typing import Dict, Any


def extract_custom_features(resume_text: str) -> Dict[str, Any]:
    """
    Extract engineered features that complement SBERT + TF-IDF.

    Returns:
        A dictionary of 7–10 handcrafted numerical features.
    """
    text_lower = resume_text.lower()
    features = {}

    # --- YEARS OF EXPERIENCE ---
    years_pattern = r'(\d+)\+?\s*(?:years|yrs|y\.?o\.e\.)'
    years_matches = re.findall(years_pattern, text_lower)
    years_list = [int(y) for y in years_matches] if years_matches else []

    features['max_years_exp'] = max(years_list) if years_list else 0
    features['total_years_mentioned'] = sum(years_list) if years_list else 0

    # --- EDUCATION LEVEL ---
    education_level = 0
    if re.search(r'\b(phd|doctorate)\b', text_lower):
        education_level = 3
    elif re.search(r'\b(master|mba|ms|ma|m.sc|mtech)\b', text_lower):
        education_level = 2
    elif re.search(r'\b(bachelor|ba|bs|bsc|btech|b.sc)\b', text_lower):
        education_level = 1
    features['education_level'] = education_level

    # --- TECH SKILLS ---
    tech_skills = [
        # Programming languages
        'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift',
        # Data / ML
        'sql', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit', 'machine learning', 'deep learning',
        # Cloud
        'aws', 'azure', 'gcp', 'cloud',
        # DevOps
        'docker', 'kubernetes', 'jenkins', 'git', 'linux',
        # Web
        'html', 'css', 'react', 'angular', 'vue', 'node', 'nodejs'
    ]

    features['tech_skill_count'] = sum(1 for skill in tech_skills if skill in text_lower)

        # --- SOFT SKILLS ---
    soft_skills = [
        'leadership', 'communication', 'teamwork',
        'problem solving', 'management', 'analytical',
        'collaboration', 'adaptability'
    ]

    features['soft_skill_count'] = sum(1 for skill in soft_skills if skill in text_lower)

    # --- CERTIFICATIONS ---
    cert_keywords = [
        'certified', 'certification', 'certificate', 'licensed',
        'aws certified', 'google certified', 'professional certificate'
    ]

    features['cert_count'] = sum(1 for cert in cert_keywords if cert in text_lower)

    # --- WORD COUNT ---
    features['word_count'] = len(resume_text.split())

    return features



