"""Text cleaning and preprocessing functions (single-cleaner version)."""

import pandas as pd
from nltk.stem import WordNetLemmatizer

from src.config import (
    URL_PATTERN,
    EMAIL_PATTERN,
    MULTI_SPACE_PATTERN,
)


def clean_text(text: str) -> str:
    """
    A SINGLE universal cleaner optimized for BOTH SBERT and TF-IDF.

    Why this works:
    - SBERT needs natural language → we keep everything meaningful.
    - TF-IDF is robust enough to handle some noise.
    - We remove only TRUE noise: URLs, emails, extra spaces.

    Args:
        text: Raw text input.

    Returns:
        Cleaned text string.
    """

    if not isinstance(text, str):
        text = str(text)

    # Normalize whitespace, replace newlines
    text = text.replace("\n", " ").replace("\t", " ")

    # Remove URLs (pure noise)
    text = URL_PATTERN.sub(" ", text)

    # Remove emails (pure noise)
    text = EMAIL_PATTERN.sub(" ", text)

    # DO NOT lowercase → SBERT prefers original casing for acronyms (SQL, AWS)
    # DO NOT remove punctuation → keeps structure for SBERT contextual embeddings
    # DO NOT remove stopwords → SBERT needs grammar (the, is, to...)
    # DO NOT remove numbers → experience years, dates are important
    # DO NOT lemmatize → loses semantic richness, unnecessary for SBERT

    # Collapse excess whitespace
    text = MULTI_SPACE_PATTERN.sub(" ", text)

    return text.strip()


def apply_cleaning(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Apply universal cleaning to a dataframe column.

    Args:
        df: Input dataframe
        text_column: Column containing raw text
        target_column: Output column for cleaned text

    Returns:
        DataFrame with cleaned text column added
    """
    cleaned_df = df.copy()
    cleaned_df[target_column] = cleaned_df[text_column].fillna("").apply(clean_text)
    return cleaned_df



