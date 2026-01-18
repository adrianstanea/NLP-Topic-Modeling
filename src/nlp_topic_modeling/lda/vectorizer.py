"""Hybrid TF-IDF + BoW vectorizer for LDA topic modeling.

This module implements the "Filter-then-Feed" architecture:
1. Use TF-IDF to rank and select top-K informative features
2. Extract raw BoW counts for only those selected features
3. Pass integer counts to LDA (preserving probabilistic integrity)
"""

from typing import Self

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nlp_topic_modeling.core.logging import get_logger
from .config import TFIDFConfig

logger = get_logger(__name__)


class HybridVectorizer:
    """TF-IDF feature selection + BoW extraction for LDA.

    This class implements the "Filter-then-Feed" architecture that combines
    the noise-reduction benefits of TF-IDF with the probabilistic integrity
    requirements of LDA:

    1. Fit TF-IDF to rank and select top-K features based on importance
    2. Extract the vocabulary of selected features
    3. Build a BoW vectorizer using only that vocabulary
    4. Return integer counts for LDA input

    This approach solves the conflict between TF-IDF's continuous weighting
    and LDA's discrete count requirement.

    Example:
        >>> config = TFIDFConfig(max_features=5000)
        >>> vectorizer = HybridVectorizer(config)
        >>> vectorizer.fit(documents)
        >>> bow_matrix = vectorizer.transform(documents)
        >>> # bow_matrix contains integer counts for LDA
    """

    def __init__(self, config: TFIDFConfig | None = None):
        """Initialize the hybrid vectorizer.

        Args:
            config: TF-IDF configuration. Uses defaults if None.
        """
        self.config = config or TFIDFConfig()
        self._is_fitted = False

        # TF-IDF vectorizer for feature selection
        self.tfidf = TfidfVectorizer(
            max_df=self.config.max_df,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            norm=self.config.norm,
        )

        # BoW vectorizer - will be fitted with TF-IDF vocabulary
        self.bow: CountVectorizer | None = None

    @property
    def vocabulary_(self) -> dict[str, int] | None:
        """Return the vocabulary mapping (feature -> index)."""
        if not self._is_fitted:
            return None
        return self.tfidf.vocabulary_

    def fit(self, documents: list[str]) -> Self:
        """Fit TF-IDF to select features and prepare BoW vectorizer.

        Args:
            documents: List of preprocessed document strings

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting TF-IDF on {len(documents)} documents")

        # Fit TF-IDF to get vocabulary
        self.tfidf.fit(documents)

        # Get the selected vocabulary
        vocabulary = self.tfidf.vocabulary_
        logger.info(f"Selected vocabulary size: {len(vocabulary)}")

        # Create BoW vectorizer with the same vocabulary
        self.bow = CountVectorizer(vocabulary=vocabulary)

        self._is_fitted = True
        return self

    def transform(self, documents: list[str]) -> sparse.csr_matrix:
        """Transform documents to BoW counts using TF-IDF selected features.

        Args:
            documents: List of preprocessed document strings

        Returns:
            Sparse matrix of integer counts (n_documents, n_features)

        Raises:
            RuntimeError: If vectorizer hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")

        # Use BoW to get integer counts
        bow_matrix = self.bow.transform(documents)

        logger.info(f"Transformed {len(documents)} documents to {bow_matrix.shape}")
        return bow_matrix

    def fit_transform(self, documents: list[str]) -> sparse.csr_matrix:
        """Fit and transform in one step.

        Args:
            documents: List of preprocessed document strings

        Returns:
            Sparse matrix of integer counts (n_documents, n_features)
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> list[str]:
        """Return the selected vocabulary as a list.

        Returns:
            List of feature names (terms) in vocabulary order

        Raises:
            RuntimeError: If vectorizer hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted before getting features")

        return self.tfidf.get_feature_names_out().tolist()

    def get_tfidf_matrix(self, documents: list[str]) -> sparse.csr_matrix:
        """Get TF-IDF matrix (for analysis, not for LDA).

        This is useful for analyzing feature importance but should NOT
        be passed to LDA which requires integer counts.

        Args:
            documents: List of preprocessed document strings

        Returns:
            Sparse TF-IDF matrix (n_documents, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer must be fitted first")

        return self.tfidf.transform(documents)

    def get_feature_importance(self, documents: list[str], top_n: int = 20) -> list[tuple[str, float]]:
        """Get top features by average TF-IDF score across documents.

        Args:
            documents: List of preprocessed document strings
            top_n: Number of top features to return

        Returns:
            List of (feature, avg_score) tuples sorted by importance
        """
        tfidf_matrix = self.get_tfidf_matrix(documents)
        feature_names = self.get_feature_names()

        # Calculate mean TF-IDF score per feature
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Get top indices
        top_indices = mean_scores.argsort()[::-1][:top_n]

        return [(feature_names[i], mean_scores[i]) for i in top_indices]
