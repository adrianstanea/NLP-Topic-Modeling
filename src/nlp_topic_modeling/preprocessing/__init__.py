"""Romanian text preprocessing pipeline for topic modeling.

This module provides a preprocessing pipeline optimized for Romanian text
from the MOROCO dataset, designed for use with LDA and BERT-based topic models.

Public API:
    - preprocess_text: Preprocess a single document
    - preprocess_documents: Batch preprocess documents
    - RomanianPreprocessor: Class-based interface for customization
    - PreprocessingConfig: Configuration dataclass

Example:
    >>> from nlp_topic_modeling.preprocessing import preprocess_documents
    >>> from nlp_topic_modeling.data.loaders import load_MOROCO
    >>>
    >>> df, _ = load_MOROCO()
    >>> df_clean = preprocess_documents(df)
    >>> print(df_clean['clean_text'].head())
"""

from .pipeline import (
    PreprocessingConfig,
    PreprocessedDocument,
    RomanianPreprocessor,
    preprocess_text,
    preprocess_documents,
)
from .normalizers import (
    normalize_diacritics,
    normalize_dialect,
    remove_ne_tokens,
    remove_urls,
    remove_html_tags,
    clean_whitespace,
    lowercase,
    keep_romanian_chars,
)
from .stopwords import (
    get_romanian_stopwords,
    get_all_stopwords,
    get_ne_stopwords,
    get_domain_stopwords,
    get_news_boilerplate_stopwords,
    filter_stopwords,
)
from .tokenizer import RomanianTokenizer, Token, TOPIC_POS_TAGS

__all__ = [
    # Main API
    "preprocess_text",
    "preprocess_documents",
    "RomanianPreprocessor",
    "PreprocessingConfig",
    "PreprocessedDocument",
    # Tokenizer
    "RomanianTokenizer",
    "Token",
    "TOPIC_POS_TAGS",
    # Normalizers
    "normalize_diacritics",
    "normalize_dialect",
    "remove_ne_tokens",
    "remove_urls",
    "remove_html_tags",
    "clean_whitespace",
    "lowercase",
    "keep_romanian_chars",
    # Stopwords
    "get_romanian_stopwords",
    "get_all_stopwords",
    "get_ne_stopwords",
    "get_domain_stopwords",
    "get_news_boilerplate_stopwords",
    "filter_stopwords",
]
