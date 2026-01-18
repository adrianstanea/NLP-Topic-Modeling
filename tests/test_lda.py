"""Tests for the TF-IDF + LDA hybrid pipeline."""

import pytest
import numpy as np
import pandas as pd

from nlp_topic_modeling.lda.config import TFIDFConfig, LDAConfig, PipelineConfig
from nlp_topic_modeling.lda.vectorizer import HybridVectorizer
from nlp_topic_modeling.lda.model import TopicModel
from nlp_topic_modeling.preprocessing.normalizers import normalize_dialect
from nlp_topic_modeling.preprocessing.stopwords import get_news_boilerplate_stopwords


class TestTFIDFConfig:
    """Tests for TF-IDF configuration."""

    def test_default_max_df(self):
        """Test that default max_df is 0.4 (R3.1)."""
        config = TFIDFConfig()
        assert config.max_df == 0.4

    def test_default_min_df(self):
        """Test that default min_df is 5 (R3.1)."""
        config = TFIDFConfig()
        assert config.min_df == 5

    def test_default_ngram_range(self):
        """Test that default ngram_range supports bigrams (R3.2)."""
        config = TFIDFConfig()
        assert config.ngram_range == (1, 2)

    def test_default_sublinear_tf(self):
        """Test that sublinear TF is enabled by default."""
        config = TFIDFConfig()
        assert config.sublinear_tf is True

    def test_default_max_features(self):
        """Test that default max_features is 5000 (feature density)."""
        config = TFIDFConfig()
        assert config.max_features == 5000


class TestLDAConfig:
    """Tests for LDA configuration."""

    def test_default_n_topics(self):
        """Test that default n_topics is 6 (MOROCO categories)."""
        config = LDAConfig()
        assert config.n_topics == 6

    def test_default_random_state(self):
        """Test that random_state is set for reproducibility."""
        config = LDAConfig()
        assert config.random_state == 42

    def test_custom_n_topics(self):
        """Test that n_topics can be customized."""
        config = LDAConfig(n_topics=10)
        assert config.n_topics == 10


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_pos_tags(self):
        """Test that default POS tags include NOUN, PROPN, ADJ (R4.1)."""
        config = PipelineConfig()
        assert 'NOUN' in config.pos_tags
        assert 'PROPN' in config.pos_tags
        assert 'ADJ' in config.pos_tags
        # Verbs should NOT be included
        assert 'VERB' not in config.pos_tags

    def test_default_normalize_dialect(self):
        """Test that dialect normalization is enabled (R3.3)."""
        config = PipelineConfig()
        assert config.normalize_dialect is True

    def test_nested_configs(self):
        """Test that nested configs are properly initialized."""
        config = PipelineConfig()
        assert isinstance(config.tfidf, TFIDFConfig)
        assert isinstance(config.lda, LDAConfig)


class TestDialectNormalization:
    """Tests for î/â dialect normalization (R3.3)."""

    def test_mid_word_i_to_a(self):
        """Test that mid-word î is converted to â."""
        assert normalize_dialect("sînt") == "sânt"
        assert normalize_dialect("cînd") == "când"
        assert normalize_dialect("vînt") == "vânt"
        assert normalize_dialect("rîu") == "râu"

    def test_word_initial_i_unchanged(self):
        """Test that word-initial î stays unchanged."""
        assert normalize_dialect("început") == "început"
        assert normalize_dialect("întrebare") == "întrebare"
        assert normalize_dialect("înainte") == "înainte"

    def test_word_final_i_unchanged(self):
        """Test that word-final î stays unchanged."""
        assert normalize_dialect("coborî") == "coborî"
        assert normalize_dialect("urî") == "urî"

    def test_full_sentence(self):
        """Test dialect normalization in a full sentence."""
        text = "Eu sînt acasă cînd vîntul bate"
        expected = "Eu sânt acasă când vântul bate"
        assert normalize_dialect(text) == expected


class TestNewsBoilerplateStopwords:
    """Tests for news boilerplate stopwords (R4.2)."""

    def test_verbal_noise_included(self):
        """Test that verbal noise words are included."""
        stopwords = get_news_boilerplate_stopwords()
        verbal_noise = ['declara', 'spune', 'preciza', 'afirma', 'transmite']
        for word in verbal_noise:
            assert word in stopwords, f"'{word}' should be in news boilerplate"

    def test_web_artifacts_included(self):
        """Test that web artifacts are included."""
        stopwords = get_news_boilerplate_stopwords()
        # Check for base forms (with and without diacritics)
        assert 'foto' in stopwords
        assert 'video' in stopwords
        assert 'articol' in stopwords
        assert 'link' in stopwords

    def test_stopwords_not_empty(self):
        """Test that stopwords set is not empty."""
        stopwords = get_news_boilerplate_stopwords()
        assert len(stopwords) > 0


class TestHybridVectorizer:
    """Tests for the HybridVectorizer."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "economie piață bursă acțiuni",
            "politică guvern parlament alegeri",
            "sport fotbal echipă meci",
            "economie bursă investiții capital",
            "politică alegeri vot democrat",
            "sport meci victorie campionat",
        ]

    def test_fit_creates_vocabulary(self, sample_documents):
        """Test that fitting creates a vocabulary."""
        config = TFIDFConfig(min_df=1, max_df=1.0, max_features=100)
        vectorizer = HybridVectorizer(config)
        vectorizer.fit(sample_documents)

        vocab = vectorizer.vocabulary_
        assert vocab is not None
        assert len(vocab) > 0

    def test_transform_returns_integers(self, sample_documents):
        """Test that transform returns integer counts (not floats)."""
        config = TFIDFConfig(min_df=1, max_df=1.0, max_features=100)
        vectorizer = HybridVectorizer(config)
        vectorizer.fit(sample_documents)

        bow_matrix = vectorizer.transform(sample_documents)

        # Check that values are integers
        data = bow_matrix.data
        assert all(val == int(val) for val in data), "BoW should contain integers"

    def test_transform_shape(self, sample_documents):
        """Test that transform returns correct shape."""
        config = TFIDFConfig(min_df=1, max_df=1.0, max_features=100)
        vectorizer = HybridVectorizer(config)
        vectorizer.fit(sample_documents)

        bow_matrix = vectorizer.transform(sample_documents)

        n_docs = len(sample_documents)
        n_features = len(vectorizer.get_feature_names())
        assert bow_matrix.shape == (n_docs, n_features)

    def test_get_feature_names(self, sample_documents):
        """Test that feature names can be retrieved."""
        config = TFIDFConfig(min_df=1, max_df=1.0, max_features=100)
        vectorizer = HybridVectorizer(config)
        vectorizer.fit(sample_documents)

        features = vectorizer.get_feature_names()
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, str) for f in features)

    def test_unfitted_vectorizer_raises(self):
        """Test that using unfitted vectorizer raises error."""
        vectorizer = HybridVectorizer()

        with pytest.raises(RuntimeError):
            vectorizer.transform(["test"])

        with pytest.raises(RuntimeError):
            vectorizer.get_feature_names()


class TestTopicModel:
    """Tests for the TopicModel."""

    @pytest.fixture
    def sample_bow_matrix(self):
        """Sample BoW matrix for testing."""
        from scipy import sparse
        # Create a simple sparse matrix
        data = np.array([
            [5, 3, 0, 0, 2, 0],
            [4, 2, 1, 0, 3, 0],
            [0, 0, 4, 3, 0, 2],
            [0, 1, 5, 4, 0, 3],
            [2, 0, 0, 0, 4, 5],
            [3, 0, 0, 0, 5, 4],
        ])
        return sparse.csr_matrix(data)

    def test_fit_sets_components(self, sample_bow_matrix):
        """Test that fitting sets topic components."""
        config = LDAConfig(n_topics=3, max_iter=10)
        model = TopicModel(config)
        model.fit(sample_bow_matrix)

        assert model.components_ is not None
        assert model.components_.shape[0] == 3  # n_topics
        assert model.components_.shape[1] == 6  # n_features

    def test_transform_returns_distributions(self, sample_bow_matrix):
        """Test that transform returns valid probability distributions."""
        config = LDAConfig(n_topics=3, max_iter=10)
        model = TopicModel(config)
        model.fit(sample_bow_matrix)

        doc_topics = model.transform(sample_bow_matrix)

        # Check shape
        assert doc_topics.shape == (6, 3)  # (n_docs, n_topics)

        # Check that rows sum to 1 (probability distributions)
        row_sums = doc_topics.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(6))

    def test_get_top_words(self, sample_bow_matrix):
        """Test that get_top_words returns correct format."""
        config = LDAConfig(n_topics=3, max_iter=10)
        model = TopicModel(config)
        model.fit(sample_bow_matrix)

        feature_names = ['word1', 'word2', 'word3', 'word4', 'word5', 'word6']
        topics = model.get_top_words(feature_names, n_words=3)

        assert len(topics) == 3  # n_topics
        for topic in topics:
            assert len(topic) == 3  # n_words
            for word, weight in topic:
                assert isinstance(word, str)
                assert isinstance(weight, (int, float))

    def test_get_dominant_topic(self, sample_bow_matrix):
        """Test that get_dominant_topic returns valid indices."""
        config = LDAConfig(n_topics=3, max_iter=10)
        model = TopicModel(config)
        model.fit(sample_bow_matrix)

        doc_topics = model.transform(sample_bow_matrix)
        dominant = model.get_dominant_topic(doc_topics)

        assert len(dominant) == 6  # n_docs
        assert all(0 <= t < 3 for t in dominant)  # Valid topic indices


class TestPublicAPI:
    """Tests for the public API imports."""

    def test_main_imports(self):
        """Test that main components can be imported."""
        from nlp_topic_modeling.lda import (
            TFIDFConfig,
            LDAConfig,
            PipelineConfig,
            HybridVectorizer,
            TopicModel,
            TopicModelingPipeline,
            train_topic_model,
        )

        assert TFIDFConfig is not None
        assert LDAConfig is not None
        assert PipelineConfig is not None
        assert HybridVectorizer is not None
        assert TopicModel is not None
        assert TopicModelingPipeline is not None
        assert train_topic_model is not None

    def test_pipeline_importable(self):
        """Test that TopicModelingPipeline can be instantiated."""
        from nlp_topic_modeling.lda import TopicModelingPipeline

        pipeline = TopicModelingPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
        assert hasattr(pipeline, 'get_topics')


# Integration tests that require spaCy model
@pytest.mark.skipif(
    False,  # Enable when spaCy model is installed
    reason="Requires spaCy Romanian model (ro_core_news_sm)"
)
class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            'sample': [
                "Economia României crește în acest trimestru cu investiții majore.",
                "Guvernul a anunțat noi măsuri economice pentru piața financiară.",
                "Echipa națională de fotbal a câștigat meciul important.",
                "Sportivii români au obținut medalii la campionatul european.",
                "Cercetătorii au descoperit o nouă metodă științifică.",
                "Universitatea a primit fonduri pentru proiecte de cercetare.",
                "Banca Națională a modificat rata dobânzii pentru economie.",
                "Investitorii străini sunt interesați de piața românească.",
                "Antrenorul echipei de fotbal pregătește strategia pentru meci.",
                "Campionatul de tenis aduce sportivi din toată lumea.",
            ]
        })

    def test_pipeline_fit(self, sample_df):
        """Test that pipeline can fit on sample data."""
        from nlp_topic_modeling.lda import TopicModelingPipeline, PipelineConfig, LDAConfig, TFIDFConfig

        config = PipelineConfig(
            lda=LDAConfig(n_topics=3, max_iter=10),
            tfidf=TFIDFConfig(min_df=1, max_df=1.0, max_features=100),
        )
        pipeline = TopicModelingPipeline(config)
        pipeline.fit(sample_df, show_progress=False)

        assert pipeline._is_fitted

    def test_pipeline_get_topics(self, sample_df):
        """Test that topics can be retrieved."""
        from nlp_topic_modeling.lda import TopicModelingPipeline, PipelineConfig, LDAConfig, TFIDFConfig

        config = PipelineConfig(
            lda=LDAConfig(n_topics=3, max_iter=10),
            tfidf=TFIDFConfig(min_df=1, max_df=1.0, max_features=100),
        )
        pipeline = TopicModelingPipeline(config)
        pipeline.fit(sample_df, show_progress=False)

        topics = pipeline.get_topics(n_words=5)

        assert len(topics) == 3  # n_topics
        for topic in topics:
            assert len(topic) <= 5  # n_words

    def test_train_topic_model_convenience(self, sample_df):
        """Test the train_topic_model convenience function."""
        from nlp_topic_modeling.lda import train_topic_model

        pipeline = train_topic_model(
            sample_df,
            n_topics=3,
            max_iter=10,
            min_df=1,
            max_df=1.0,
            max_features=100,
            show_progress=False,
        )

        assert pipeline._is_fitted
        topics = pipeline.get_topic_words(n_words=3)
        assert len(topics) == 3

    def test_vocabulary_size(self, sample_df):
        """Test that vocabulary size is reasonable."""
        from nlp_topic_modeling.lda import TopicModelingPipeline, PipelineConfig, LDAConfig, TFIDFConfig

        config = PipelineConfig(
            lda=LDAConfig(n_topics=3, max_iter=10),
            tfidf=TFIDFConfig(min_df=1, max_df=1.0, max_features=100),
        )
        pipeline = TopicModelingPipeline(config)
        pipeline.fit(sample_df, show_progress=False)

        vocab_size = pipeline.get_vocabulary_size()
        assert vocab_size > 0
        assert vocab_size <= 100  # max_features


class TestPOSFiltering:
    """Tests for POS-filtered tokenization."""

    @pytest.mark.skipif(
        False,  # Enable when spaCy model is installed
        reason="Requires spaCy Romanian model (ro_core_news_sm)"
    )
    def test_pos_filtering_keeps_nouns(self):
        """Test that POS filtering keeps nouns."""
        from nlp_topic_modeling.preprocessing import RomanianTokenizer

        tokenizer = RomanianTokenizer()
        tokens = tokenizer.tokenize_pos_filtered(
            "Ministrul a declarat că economia crește",
            include_pos={'NOUN'}
        )

        # Should contain nouns like "ministru", "economie"
        assert len(tokens) > 0

    @pytest.mark.skipif(
        False,  # Enable when spaCy model is installed
        reason="Requires spaCy Romanian model (ro_core_news_sm)"
    )
    def test_pos_filtering_excludes_verbs(self):
        """Test that POS filtering excludes verbs."""
        from nlp_topic_modeling.preprocessing import RomanianTokenizer

        tokenizer = RomanianTokenizer()
        all_tokens = tokenizer.tokenize_and_lemmatize("Ministrul a declarat că economia crește")
        filtered_tokens = tokenizer.tokenize_pos_filtered(
            "Ministrul a declarat că economia crește",
            include_pos={'NOUN', 'ADJ'}
        )

        # Filtered should have fewer tokens (verbs removed)
        assert len(filtered_tokens) < len(all_tokens)
