"""LDA topic model wrapper with analysis utilities."""

from typing import Self

import numpy as np
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation

from nlp_topic_modeling.core.logging import get_logger
from .config import LDAConfig

logger = get_logger(__name__)


class TopicModel:
    """LDA model wrapper with topic analysis utilities.

    Wraps sklearn's LatentDirichletAllocation with convenience methods
    for topic analysis and interpretation.

    Example:
        >>> config = LDAConfig(n_topics=6)
        >>> model = TopicModel(config)
        >>> model.fit(bow_matrix)
        >>> topics = model.get_top_words(feature_names, n_words=10)
        >>> for i, topic in enumerate(topics):
        ...     print(f"Topic {i}: {[w for w, _ in topic]}")
    """

    def __init__(self, config: LDAConfig | None = None):
        """Initialize the topic model.

        Args:
            config: LDA configuration. Uses defaults if None.
        """
        self.config = config or LDAConfig()
        self._is_fitted = False

        self.model = LatentDirichletAllocation(
            n_components=self.config.n_topics,
            max_iter=self.config.max_iter,
            learning_method=self.config.learning_method,
            learning_decay=self.config.learning_decay,
            learning_offset=self.config.learning_offset,
            batch_size=self.config.batch_size,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )

    @property
    def n_topics(self) -> int:
        """Return the number of topics."""
        return self.config.n_topics

    @property
    def components_(self) -> np.ndarray | None:
        """Return the topic-word distribution matrix.

        Shape: (n_topics, n_features)
        Each row is a topic, each column is a word weight.
        """
        if not self._is_fitted:
            return None
        return self.model.components_

    def fit(self, bow_matrix: sparse.csr_matrix) -> Self:
        """Fit LDA to BoW counts.

        Args:
            bow_matrix: Sparse matrix of integer counts (n_documents, n_features)

        Returns:
            Self for method chaining
        """
        n_docs, n_features = bow_matrix.shape
        logger.info(f"Fitting LDA with {self.n_topics} topics on {n_docs} documents, {n_features} features")

        self.model.fit(bow_matrix)
        self._is_fitted = True

        logger.info(f"LDA fitting complete. Perplexity: {self.model.perplexity(bow_matrix):.2f}")
        return self

    def transform(self, bow_matrix: sparse.csr_matrix) -> np.ndarray:
        """Get topic distributions for documents.

        Args:
            bow_matrix: Sparse matrix of integer counts (n_documents, n_features)

        Returns:
            Array of shape (n_documents, n_topics) with topic probabilities

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        return self.model.transform(bow_matrix)

    def fit_transform(self, bow_matrix: sparse.csr_matrix) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            bow_matrix: Sparse matrix of integer counts (n_documents, n_features)

        Returns:
            Array of shape (n_documents, n_topics) with topic probabilities
        """
        self.fit(bow_matrix)
        return self.transform(bow_matrix)

    def get_top_words(
        self,
        feature_names: list[str],
        n_words: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Get top words with weights for each topic.

        Args:
            feature_names: List of feature names from vectorizer
            n_words: Number of top words to return per topic

        Returns:
            List of topics, each containing (word, weight) tuples

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting topics")

        topics = []
        for topic_idx, topic_weights in enumerate(self.components_):
            # Get indices of top words
            top_indices = topic_weights.argsort()[::-1][:n_words]

            # Build list of (word, weight) tuples
            topic_words = [
                (feature_names[i], topic_weights[i])
                for i in top_indices
            ]
            topics.append(topic_words)

        return topics

    def get_topic_words_only(
        self,
        feature_names: list[str],
        n_words: int = 10
    ) -> list[list[str]]:
        """Get top words for each topic (without weights).

        Args:
            feature_names: List of feature names from vectorizer
            n_words: Number of top words to return per topic

        Returns:
            List of topics, each containing word strings
        """
        topics_with_weights = self.get_top_words(feature_names, n_words)
        return [[word for word, _ in topic] for topic in topics_with_weights]

    def get_document_topics(
        self,
        doc_topic_dist: np.ndarray,
        doc_index: int,
        threshold: float = 0.05
    ) -> list[tuple[int, float]]:
        """Get topic distribution for a single document.

        Args:
            doc_topic_dist: Document-topic distribution matrix from transform()
            doc_index: Index of the document
            threshold: Minimum probability to include a topic

        Returns:
            List of (topic_id, probability) tuples above threshold
        """
        doc_topics = doc_topic_dist[doc_index]
        return [
            (topic_id, prob)
            for topic_id, prob in enumerate(doc_topics)
            if prob >= threshold
        ]

    def get_dominant_topic(self, doc_topic_dist: np.ndarray) -> np.ndarray:
        """Get the dominant topic for each document.

        Args:
            doc_topic_dist: Document-topic distribution matrix from transform()

        Returns:
            Array of dominant topic indices for each document
        """
        return doc_topic_dist.argmax(axis=1)

    def get_perplexity(self, bow_matrix: sparse.csr_matrix) -> float:
        """Calculate perplexity on a BoW matrix.

        Lower perplexity indicates better generalization.

        Args:
            bow_matrix: Sparse matrix of integer counts

        Returns:
            Perplexity score

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calculating perplexity")

        return self.model.perplexity(bow_matrix)

    def get_log_likelihood(self, bow_matrix: sparse.csr_matrix) -> float:
        """Calculate log-likelihood on a BoW matrix.

        Higher log-likelihood indicates better fit.

        Args:
            bow_matrix: Sparse matrix of integer counts

        Returns:
            Log-likelihood score

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calculating log-likelihood")

        return self.model.score(bow_matrix)
