"""Configuration dataclasses for TF-IDF + LDA hybrid pipeline."""

from dataclasses import dataclass, field


@dataclass
class TFIDFConfig:
    """TF-IDF vectorizer configuration.

    Implements requirements:
    - R3.1: Pruning via max_df and min_df
    - R3.2: Bigram support via ngram_range
    - Sublinear TF scaling for length normalization
    """
    max_df: float = 0.4
    """Maximum document frequency (R3.1). Terms in >40% of docs are pruned."""

    min_df: int = 5
    """Minimum document frequency (R3.1). Terms in <5 docs are pruned."""

    max_features: int = 5000
    """Maximum vocabulary size. Feature density principle: ~5000 dense lemmas."""

    ngram_range: tuple[int, int] = (1, 2)
    """N-gram range (R3.2). (1,2) = unigrams and bigrams."""

    sublinear_tf: bool = True
    """Use sublinear TF scaling: 1 + log(tf). Normalizes for article length."""

    norm: str = 'l2'
    """Normalization method for TF-IDF vectors."""


@dataclass
class LDAConfig:
    """LDA model configuration."""

    n_topics: int = 6
    """Number of topics to extract. MOROCO has 6 categories."""

    max_iter: int = 750
    """Maximum number of iterations for LDA fitting."""

    learning_method: str = 'online'
    """Learning method: 'online' for large datasets, 'batch' for small."""

    learning_decay: float = 0.5
    """Learning rate decay parameter for online learning."""

    learning_offset: float = 10.0
    """Learning offset for early iterations in online learning."""

    batch_size: int = 128
    """Mini-batch size for online learning."""

    n_jobs: int = -1
    """Number of parallel jobs. -1 = use all CPUs."""

    random_state: int = 42
    """Random seed for reproducibility."""


@dataclass
class PipelineConfig:
    """Full TF-IDF + LDA pipeline configuration.

    Combines TF-IDF feature selection, LDA training, and preprocessing options.
    """

    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    """TF-IDF vectorizer configuration."""

    lda: LDAConfig = field(default_factory=LDAConfig)
    """LDA model configuration."""

    pos_tags: set[str] = field(default_factory=lambda: {'NOUN', 'PROPN', 'ADJ'})
    """POS tags to keep (R4.1). Focus on topical skeleton, not writing style."""

    spacy_model: str = 'ro_core_news_sm'
    """spaCy model for Romanian tokenization and POS tagging."""

    normalize_dialect: bool = True
    """Whether to normalize î/â orthographic variation (R3.3)."""

    include_stopwords: bool = True
    """Whether to filter stopwords including news boilerplate (R4.2)."""
