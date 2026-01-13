"""TF-IDF + LDA hybrid pipeline for Romanian topic modeling.

This module provides a production-ready topic modeling pipeline that combines
the noise-reduction benefits of TF-IDF with the probabilistic integrity of LDA.

Architecture:
    Input Text → Preprocessing → POS Filter → TF-IDF → Feature Selection → BoW → LDA

Key Features:
    - Filter-then-Feed: TF-IDF selects features, BoW feeds LDA
    - POS Filtering: Focus on NOUN, PROPN, ADJ (topical skeleton)
    - Dialect Normalization: Handles RO/MD î/â variation
    - News Boilerplate Removal: Filters journalism-specific noise

Example:
    >>> from nlp_topic_modeling.lda import train_topic_model
    >>> from nlp_topic_modeling.data.loaders import load_MOROCO
    >>>
    >>> df, _ = load_MOROCO()
    >>> pipeline = train_topic_model(df, n_topics=6)
    >>>
    >>> # Print topics
    >>> pipeline.print_topics(n_words=10)
    >>>
    >>> # Get topic distributions
    >>> doc_topics = pipeline.fit_transform(df)
"""

import pandas as pd

from .config import TFIDFConfig, LDAConfig, PipelineConfig
from .vectorizer import HybridVectorizer
from .model import TopicModel
from .pipeline import TopicModelingPipeline


def train_topic_model(
    df: pd.DataFrame,
    n_topics: int = 6,
    text_column: str = 'sample',
    show_progress: bool = True,
    **config_kwargs
) -> TopicModelingPipeline:
    """Train a topic model on a DataFrame.

    This is a convenience function that creates and fits a TopicModelingPipeline
    with sensible defaults.

    Args:
        df: Input DataFrame with text data
        n_topics: Number of topics to extract (default: 6 for MOROCO categories)
        text_column: Name of the column containing text
        show_progress: Whether to show progress bars
        **config_kwargs: Additional configuration options passed to PipelineConfig

    Returns:
        Fitted TopicModelingPipeline

    Example:
        >>> pipeline = train_topic_model(df, n_topics=6)
        >>> pipeline.print_topics()
    """
    # Build config
    lda_config = LDAConfig(n_topics=n_topics)
    tfidf_config = TFIDFConfig()

    # Apply any overrides
    for key, value in config_kwargs.items():
        if hasattr(lda_config, key):
            setattr(lda_config, key, value)
        elif hasattr(tfidf_config, key):
            setattr(tfidf_config, key, value)

    config = PipelineConfig(
        tfidf=tfidf_config,
        lda=lda_config,
    )

    # Create and fit pipeline
    pipeline = TopicModelingPipeline(config)
    pipeline.fit(df, text_column=text_column, show_progress=show_progress)

    return pipeline


__all__ = [
    # Configuration
    'TFIDFConfig',
    'LDAConfig',
    'PipelineConfig',
    # Components
    'HybridVectorizer',
    'TopicModel',
    # Main pipeline
    'TopicModelingPipeline',
    # Convenience function
    'train_topic_model',
]
