"""Full TF-IDF + LDA hybrid pipeline for Romanian topic modeling.

This module implements the complete topic modeling workflow:
1. Input: Raw text from MOROCO dataset
2. Preprocessing: Normalization, dialect standardization
3. POS Filter: Keep NOUN, PROPN, ADJ (topical skeleton)
4. Vectorization: TF-IDF for feature selection
5. LDA Training: BoW counts for probabilistic topic modeling
6. Output: Topic distributions and interpretable topics
"""

from typing import Self

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nlp_topic_modeling.core.logging import get_logger
from nlp_topic_modeling.preprocessing import (
    RomanianTokenizer,
    normalize_diacritics,
    normalize_dialect,
    remove_ne_tokens,
    clean_whitespace,
    lowercase,
    get_all_stopwords,
    filter_stopwords,
)
from .config import PipelineConfig
from .vectorizer import HybridVectorizer
from .model import TopicModel

logger = get_logger(__name__)


class TopicModelingPipeline:
    """Full TF-IDF + LDA hybrid pipeline for Romanian topic modeling.

    Implements the logical workflow from the design specification:
    1. Input: Lemmatized & Normalized Text
    2. POS Filter: Keep NOUN, PROPN, ADJ
    3. Vectorization: Calculate TF-IDF (1,2-grams)
    4. Vocabulary Selection: Filter by max_df, min_df, stoplist
    5. LDA Training: Pass BoW counts for final features
    6. Output: Topic distributions

    Example:
        >>> from nlp_topic_modeling.lda import TopicModelingPipeline
        >>> from nlp_topic_modeling.data.loaders import load_MOROCO
        >>>
        >>> df, _ = load_MOROCO()
        >>> pipeline = TopicModelingPipeline()
        >>> doc_topics = pipeline.fit_transform(df)
        >>>
        >>> # Get topics
        >>> for i, topic in enumerate(pipeline.get_topics()):
        ...     print(f"Topic {i}: {[w for w, _ in topic[:5]]}")
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()

        # Components - lazy loaded
        self._tokenizer: RomanianTokenizer | None = None
        self._stopwords: set[str] | None = None

        # Core components
        self.vectorizer = HybridVectorizer(self.config.tfidf)
        self.model = TopicModel(self.config.lda)

        # State
        self._is_fitted = False
        self._processed_docs: list[str] | None = None

    @property
    def tokenizer(self) -> RomanianTokenizer:
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = RomanianTokenizer(self.config.spacy_model)
        return self._tokenizer

    @property
    def stopwords(self) -> set[str]:
        """Get the stopwords set."""
        if self._stopwords is None:
            self._stopwords = get_all_stopwords(include_news_boilerplate=self.config.include_stopwords)
        return self._stopwords

    def _preprocess_text(self, text: str) -> str:
        """Preprocess a single document.

        Applies:
        1. Lowercase
        2. Diacritic normalization (cedilla -> comma-below)
        3. Dialect normalization (î -> â mid-word)
        4. Remove $NE$ tokens
        5. Clean whitespace

        Args:
            text: Raw input text

        Returns:
            Normalized text ready for tokenization
        """
        text = lowercase(text)
        text = normalize_diacritics(text)

        if self.config.normalize_dialect:
            text = normalize_dialect(text)

        text = remove_ne_tokens(text)
        text = clean_whitespace(text)

        return text

    def _tokenize_document(self, text: str) -> str:
        """Tokenize and filter a single document.

        Applies:
        1. POS-filtered tokenization (NOUN, PROPN, ADJ)
        2. Stopword filtering

        Args:
            text: Preprocessed text

        Returns:
            Space-joined string of filtered lemmas
        """
        # POS-filtered tokenization
        tokens = self.tokenizer.tokenize_pos_filtered(text, self.config.pos_tags)

        # Stopword filtering
        if self.config.include_stopwords:
            tokens = filter_stopwords(tokens, self.stopwords, min_length=3)

        return ' '.join(tokens)

    def _prepare_documents(
        self,
        df: pd.DataFrame,
        text_column: str,
        show_progress: bool = True
    ) -> list[str]:
        """Prepare documents for vectorization.

        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            show_progress: Whether to show progress bar

        Returns:
            List of preprocessed and tokenized document strings
        """
        logger.info(f"Preparing {len(df)} documents for topic modeling")

        processed_docs = []
        texts = df[text_column].tolist()

        for text in tqdm(texts, desc="Preprocessing", disable=not show_progress):
            # Preprocess
            normalized = self._preprocess_text(text)
            # Tokenize with POS filtering
            tokenized = self._tokenize_document(normalized)
            processed_docs.append(tokenized)

        # Log statistics
        non_empty = sum(1 for doc in processed_docs if doc.strip())
        logger.info(f"Prepared {non_empty}/{len(processed_docs)} non-empty documents")

        return processed_docs

    def fit(
        self,
        df: pd.DataFrame,
        text_column: str = 'sample',
        show_progress: bool = True
    ) -> Self:
        """Fit the full pipeline on a DataFrame.

        Args:
            df: Input DataFrame with text data
            text_column: Name of the column containing text
            show_progress: Whether to show progress bars

        Returns:
            Self for method chaining
        """
        # Prepare documents
        self._processed_docs = self._prepare_documents(df, text_column, show_progress)

        # Fit vectorizer
        logger.info("Fitting TF-IDF vectorizer")
        self.vectorizer.fit(self._processed_docs)

        # Get BoW matrix
        bow_matrix = self.vectorizer.transform(self._processed_docs)

        # Fit LDA
        logger.info("Fitting LDA model")
        self.model.fit(bow_matrix)

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        text_column: str = 'sample',
        show_progress: bool = True
    ) -> np.ndarray:
        """Get topic distributions for documents.

        Args:
            df: Input DataFrame with text data
            text_column: Name of the column containing text
            show_progress: Whether to show progress bars

        Returns:
            Array of shape (n_documents, n_topics) with topic probabilities

        Raises:
            RuntimeError: If pipeline hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        # Prepare documents
        processed_docs = self._prepare_documents(df, text_column, show_progress)

        # Vectorize
        bow_matrix = self.vectorizer.transform(processed_docs)

        # Get topic distributions
        return self.model.transform(bow_matrix)

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_column: str = 'sample',
        show_progress: bool = True
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            df: Input DataFrame with text data
            text_column: Name of the column containing text
            show_progress: Whether to show progress bars

        Returns:
            Array of shape (n_documents, n_topics) with topic probabilities
        """
        self.fit(df, text_column, show_progress)

        # Use already processed docs for efficiency
        bow_matrix = self.vectorizer.transform(self._processed_docs)
        return self.model.transform(bow_matrix)

    def get_topics(self, n_words: int = 10) -> list[list[tuple[str, float]]]:
        """Get top words for each topic.

        Args:
            n_words: Number of top words to return per topic

        Returns:
            List of topics, each containing (word, weight) tuples

        Raises:
            RuntimeError: If pipeline hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before getting topics")

        feature_names = self.vectorizer.get_feature_names()
        return self.model.get_top_words(feature_names, n_words)

    def get_topic_words(self, n_words: int = 10) -> list[list[str]]:
        """Get top words for each topic (without weights).

        Args:
            n_words: Number of top words to return per topic

        Returns:
            List of topics, each containing word strings
        """
        topics = self.get_topics(n_words)
        return [[word for word, _ in topic] for topic in topics]

    def get_document_topics(
        self,
        doc_topic_dist: np.ndarray,
        doc_index: int,
        threshold: float = 0.05
    ) -> list[tuple[int, float]]:
        """Get topic distribution for a single document.

        Args:
            doc_topic_dist: Document-topic distribution from fit_transform()
            doc_index: Index of the document
            threshold: Minimum probability to include a topic

        Returns:
            List of (topic_id, probability) tuples above threshold
        """
        return self.model.get_document_topics(doc_topic_dist, doc_index, threshold)

    def get_dominant_topics(self, doc_topic_dist: np.ndarray) -> np.ndarray:
        """Get the dominant topic for each document.

        Args:
            doc_topic_dist: Document-topic distribution from fit_transform()

        Returns:
            Array of dominant topic indices
        """
        return self.model.get_dominant_topic(doc_topic_dist)

    def get_vocabulary_size(self) -> int:
        """Get the size of the selected vocabulary.

        Returns:
            Number of features in the vocabulary

        Raises:
            RuntimeError: If pipeline hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")

        return len(self.vectorizer.get_feature_names())

    def print_topics(self, n_words: int = 10) -> None:
        """Print topics to console for quick inspection.

        Args:
            n_words: Number of top words to show per topic
        """
        topics = self.get_topic_words(n_words)
        for i, topic_words in enumerate(topics):
            print(f"Topic {i}: {', '.join(topic_words)}")
