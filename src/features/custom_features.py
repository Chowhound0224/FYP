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
        # === PROGRAMMING LANGUAGES ===
        'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift',
        'typescript', 'kotlin', 'scala', 'rust', 'r', 'matlab', 'perl', 'shell',
        'bash', 'powershell', 'objective-c', 'dart', 'lua', 'groovy',

        # === WEB DEVELOPMENT ===
        'html', 'css', 'react', 'angular', 'vue', 'node', 'nodejs', 'express',
        'django', 'flask', 'fastapi', 'spring', 'spring boot', 'asp.net', 'laravel',
        'jquery', 'bootstrap', 'tailwind', 'sass', 'webpack', 'next.js', 'nuxt',

        # === MOBILE DEVELOPMENT ===
        'android', 'ios', 'flutter', 'react native', 'xamarin', 'ionic', 'swift ui',

        # === DATABASES ===
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'redis',
        'cassandra', 'dynamodb', 'sqlite', 'mariadb', 'elasticsearch', 'neo4j',

        # === DATA SCIENCE & ML ===
        'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit', 'scikit-learn',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'keras',
        'xgboost', 'lightgbm', 'spark', 'hadoop', 'data analysis', 'statistics',
        'matplotlib', 'seaborn', 'plotly', 'jupyter', 'rstudio',

        # === CLOUD & DEVOPS ===
        'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'jenkins',
        'ci/cd', 'terraform', 'ansible', 'chef', 'puppet', 'circleci',
        'github actions', 'gitlab', 'bitbucket', 'travis ci',

        # === VERSION CONTROL ===
        'git', 'github', 'svn', 'mercurial',

        # === OPERATING SYSTEMS ===
        'linux', 'unix', 'windows', 'macos', 'ubuntu', 'centos', 'debian',

        # === BUSINESS INTELLIGENCE & ANALYTICS ===
        'tableau', 'power bi', 'looker', 'qlik', 'excel', 'google analytics',
        'sas', 'spss', 'alteryx',

        # === PROJECT MANAGEMENT & COLLABORATION ===
        'jira', 'confluence', 'trello', 'asana', 'monday', 'slack', 'teams',
        'agile', 'scrum', 'kanban', 'waterfall',

        # === TESTING & QA ===
        'selenium', 'junit', 'pytest', 'jest', 'cypress', 'postman',
        'unit testing', 'integration testing', 'automation testing',

        # === NETWORKING & SECURITY ===
        'tcp/ip', 'dns', 'vpn', 'firewall', 'encryption', 'ssl', 'oauth',
        'cybersecurity', 'penetration testing', 'ethical hacking',

        # === ERP & CRM ===
        'sap', 'salesforce', 'oracle', 'dynamics', 'crm', 'erp',

        # === DESIGN & MULTIMEDIA ===
        'photoshop', 'illustrator', 'figma', 'sketch', 'adobe xd', 'canva',
        'premiere', 'after effects', 'autocad', 'solidworks', 'ui/ux',

        # === BLOCKCHAIN & EMERGING TECH ===
        'blockchain', 'ethereum', 'solidity', 'web3', 'cryptocurrency',
        'iot', 'edge computing', 'quantum computing',
    ]

    features['tech_skill_count'] = sum(1 for skill in tech_skills if skill in text_lower)

    # --- SOFT SKILLS ---
    soft_skills = [
        # === CORE PROFESSIONAL SKILLS ===
        'leadership', 'communication', 'teamwork', 'collaboration',
        'problem solving', 'critical thinking', 'analytical',
        'decision making', 'time management', 'organization',

        # === INTERPERSONAL SKILLS ===
        'adaptability', 'flexibility', 'creativity', 'innovation',
        'emotional intelligence', 'empathy', 'active listening',
        'conflict resolution', 'negotiation', 'persuasion',

        # === MANAGEMENT & LEADERSHIP ===
        'management', 'project management', 'people management',
        'strategic planning', 'mentoring', 'coaching', 'delegation',

        # === WORK ETHIC ===
        'self-motivated', 'proactive', 'detail-oriented', 'multitasking',
        'work ethic', 'accountability', 'responsibility', 'initiative',

        # === COMMUNICATION ===
        'presentation', 'public speaking', 'writing', 'reporting',
        'stakeholder management', 'client facing',
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



