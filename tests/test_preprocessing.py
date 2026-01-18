"""Tests for the preprocessing module."""

import pytest
import pandas as pd

from nlp_topic_modeling.preprocessing.normalizers import (
    normalize_diacritics,
    lowercase,
    remove_ne_tokens,
    remove_urls,
    remove_html_tags,
    clean_whitespace,
    keep_romanian_chars,
)
from nlp_topic_modeling.preprocessing.stopwords import (
    get_romanian_stopwords,
    get_all_stopwords,
    get_ne_stopwords,
    filter_stopwords,
)


class TestNormalizeDiacritics:
    """Tests for diacritic normalization."""

    def test_cedilla_to_comma_below_lowercase(self):
        """Test conversion of cedilla to comma-below for lowercase."""
        # ş (U+015F) -> ș (U+0219)
        assert normalize_diacritics('ş') == 'ș'
        # ţ (U+0163) -> ț (U+021B)
        assert normalize_diacritics('ţ') == 'ț'

    def test_cedilla_to_comma_below_uppercase(self):
        """Test conversion of cedilla to comma-below for uppercase."""
        # Ş (U+015E) -> Ș (U+0218)
        assert normalize_diacritics('Ş') == 'Ș'
        # Ţ (U+0162) -> Ț (U+021A)
        assert normalize_diacritics('Ţ') == 'Ț'

    def test_full_text_normalization(self):
        """Test diacritic normalization in full text."""
        text = "Şcoala ţării noastre"
        expected = "Școala țării noastre"
        assert normalize_diacritics(text) == expected

    def test_already_normalized_unchanged(self):
        """Test that already normalized text is unchanged."""
        text = "Școala țării noastre"
        assert normalize_diacritics(text) == text

    def test_mixed_diacritics(self):
        """Test text with mixed old and new diacritics."""
        text = "ş și ț și ţ"
        expected = "ș și ț și ț"
        assert normalize_diacritics(text) == expected


class TestLowercase:
    """Tests for lowercase conversion."""

    def test_basic_lowercase(self):
        """Test basic lowercase conversion."""
        assert lowercase("HELLO") == "hello"
        assert lowercase("Hello World") == "hello world"

    def test_romanian_chars_lowercase(self):
        """Test lowercase with Romanian characters."""
        assert lowercase("ĂÂÎȘȚ") == "ăâîșț"


class TestRemoveNeTokens:
    """Tests for $NE$ token removal."""

    def test_single_ne_token(self):
        """Test removal of single $NE$ token."""
        text = "Merge la $NE$ pentru a studia."
        expected = "Merge la  pentru a studia."
        assert remove_ne_tokens(text) == expected

    def test_multiple_ne_tokens(self):
        """Test removal of multiple $NE$ tokens."""
        text = "$NE$ și $NE$ au mers la $NE$"
        expected = " și  au mers la "
        assert remove_ne_tokens(text) == expected

    def test_no_ne_tokens(self):
        """Test text without $NE$ tokens is unchanged."""
        text = "Text fără marcatori speciali"
        assert remove_ne_tokens(text) == text

    def test_adjacent_ne_tokens(self):
        """Test removal of adjacent $NE$ tokens."""
        text = "$NE$ $NE$ $NE$"
        expected = "  "
        assert remove_ne_tokens(text) == expected


class TestRemoveUrls:
    """Tests for URL removal."""

    def test_http_url(self):
        """Test removal of http URL."""
        text = "Vizitați http://example.com pentru mai multe"
        result = remove_urls(text)
        assert "http://example.com" not in result

    def test_https_url(self):
        """Test removal of https URL."""
        text = "Vizitați https://example.com/page pentru mai multe"
        result = remove_urls(text)
        assert "https://example.com/page" not in result

    def test_www_url(self):
        """Test removal of www URL."""
        text = "Vizitați www.example.com pentru mai multe"
        result = remove_urls(text)
        assert "www.example.com" not in result

    def test_no_url_unchanged(self):
        """Test text without URLs is unchanged."""
        text = "Text fără linkuri"
        assert remove_urls(text) == text


class TestRemoveHtmlTags:
    """Tests for HTML tag removal."""

    def test_simple_tags(self):
        """Test removal of simple HTML tags."""
        text = "<p>Paragraf</p>"
        assert remove_html_tags(text) == "Paragraf"

    def test_nested_tags(self):
        """Test removal of nested HTML tags."""
        text = "<div><p>Text</p></div>"
        assert remove_html_tags(text) == "Text"

    def test_tags_with_attributes(self):
        """Test removal of tags with attributes."""
        text = '<a href="url">Link</a>'
        assert remove_html_tags(text) == "Link"

    def test_no_tags_unchanged(self):
        """Test text without HTML tags is unchanged."""
        text = "Text simplu"
        assert remove_html_tags(text) == text


class TestCleanWhitespace:
    """Tests for whitespace normalization."""

    def test_multiple_spaces(self):
        """Test normalization of multiple spaces."""
        text = "Text   cu   spații   multiple"
        expected = "Text cu spații multiple"
        assert clean_whitespace(text) == expected

    def test_newlines(self):
        """Test normalization of newlines."""
        text = "Text\ncu\nnewlines"
        expected = "Text cu newlines"
        assert clean_whitespace(text) == expected

    def test_mixed_whitespace(self):
        """Test normalization of mixed whitespace."""
        text = "Text  \n\t  cu   \n  whitespace"
        expected = "Text cu whitespace"
        assert clean_whitespace(text) == expected

    def test_leading_trailing_whitespace(self):
        """Test stripping of leading/trailing whitespace."""
        text = "   Text cu padding   "
        expected = "Text cu padding"
        assert clean_whitespace(text) == expected


class TestKeepRomanianChars:
    """Tests for keeping only Romanian characters."""

    def test_removes_punctuation(self):
        """Test removal of punctuation."""
        text = "Salut, lume!"
        result = keep_romanian_chars(text)
        assert "," not in result
        assert "!" not in result

    def test_removes_digits(self):
        """Test removal of digits."""
        text = "Am 123 mere"
        result = keep_romanian_chars(text)
        assert "123" not in result

    def test_keeps_romanian_chars(self):
        """Test that Romanian characters are kept."""
        text = "ăâîșț ĂÂÎȘȚ"
        assert keep_romanian_chars(text) == text

    def test_keeps_spaces(self):
        """Test that spaces are kept."""
        text = "Un text simplu"
        assert keep_romanian_chars(text) == text


class TestStopwords:
    """Tests for stopword functionality."""

    def test_get_romanian_stopwords_not_empty(self):
        """Test that Romanian stopwords set is not empty."""
        stopwords = get_romanian_stopwords()
        assert len(stopwords) > 0

    def test_get_romanian_stopwords_contains_common_words(self):
        """Test that common Romanian stopwords are included."""
        stopwords = get_romanian_stopwords()
        # Use words that are unambiguous across diacritic variations
        # Note: 'și' may be stored as 'şi' (cedilla) in some libraries
        common_words = ['este', 'la', 'de', 'care', 'nu', 'cu']
        for word in common_words:
            assert word in stopwords or word.lower() in stopwords

    def test_get_ne_stopwords(self):
        """Test that NE stopwords are returned."""
        stopwords = get_ne_stopwords()
        assert '$NE$' in stopwords or '$ne$' in stopwords

    def test_get_all_stopwords_combines_sources(self):
        """Test that all stopwords combines multiple sources."""
        all_sw = get_all_stopwords()
        ro_sw = get_romanian_stopwords()

        # Should include Romanian stopwords
        assert len(all_sw) >= len(ro_sw)

    def test_filter_stopwords_removes_stopwords(self):
        """Test that filter_stopwords removes stopwords."""
        tokens = ['acest', 'text', 'este', 'un', 'test']
        stopwords = {'acest', 'este', 'un'}
        result = filter_stopwords(tokens, stopwords)

        assert 'acest' not in result
        assert 'este' not in result
        assert 'un' not in result
        assert 'text' in result
        assert 'test' in result

    def test_filter_stopwords_respects_min_length(self):
        """Test that filter_stopwords respects minimum length."""
        tokens = ['a', 'de', 'text', 'mare']
        result = filter_stopwords(tokens, set(), min_length=3)

        assert 'a' not in result
        assert 'de' not in result
        assert 'text' in result
        assert 'mare' in result


class TestTokenizerImportable:
    """Tests that the tokenizer module is importable."""

    def test_tokenizer_class_importable(self):
        """Test that RomanianTokenizer can be imported."""
        from nlp_topic_modeling.preprocessing.tokenizer import RomanianTokenizer
        assert RomanianTokenizer is not None

    def test_token_namedtuple_importable(self):
        """Test that Token namedtuple can be imported."""
        from nlp_topic_modeling.preprocessing.tokenizer import Token
        assert Token is not None


class TestPipelineImportable:
    """Tests that the pipeline module is importable."""

    def test_preprocessor_class_importable(self):
        """Test that RomanianPreprocessor can be imported."""
        from nlp_topic_modeling.preprocessing.pipeline import RomanianPreprocessor
        assert RomanianPreprocessor is not None

    def test_config_class_importable(self):
        """Test that PreprocessingConfig can be imported."""
        from nlp_topic_modeling.preprocessing.pipeline import PreprocessingConfig
        assert PreprocessingConfig is not None

    def test_convenience_functions_importable(self):
        """Test that convenience functions can be imported."""
        from nlp_topic_modeling.preprocessing.pipeline import (
            preprocess_text,
            preprocess_documents,
        )
        assert preprocess_text is not None
        assert preprocess_documents is not None


class TestPublicAPIImportable:
    """Tests that the public API is importable from the package."""

    def test_main_imports(self):
        """Test main imports from preprocessing package."""
        from nlp_topic_modeling.preprocessing import (
            preprocess_text,
            preprocess_documents,
            RomanianPreprocessor,
            PreprocessingConfig,
        )
        assert preprocess_text is not None
        assert preprocess_documents is not None
        assert RomanianPreprocessor is not None
        assert PreprocessingConfig is not None

    def test_normalizer_imports(self):
        """Test normalizer imports from preprocessing package."""
        from nlp_topic_modeling.preprocessing import (
            normalize_diacritics,
            remove_ne_tokens,
            remove_urls,
        )
        assert normalize_diacritics is not None
        assert remove_ne_tokens is not None
        assert remove_urls is not None

    def test_stopword_imports(self):
        """Test stopword imports from preprocessing package."""
        from nlp_topic_modeling.preprocessing import (
            get_romanian_stopwords,
            get_all_stopwords,
            filter_stopwords,
        )
        assert get_romanian_stopwords is not None
        assert get_all_stopwords is not None
        assert filter_stopwords is not None


class TestPreprocessingConfigDefaults:
    """Tests for PreprocessingConfig default values."""

    def test_default_config_values(self):
        """Test that default config has expected values."""
        from nlp_topic_modeling.preprocessing import PreprocessingConfig

        config = PreprocessingConfig()

        assert config.lowercase is True
        assert config.normalize_diacritics is True
        assert config.remove_ne_tokens is True
        assert config.remove_urls is True
        assert config.remove_html is True
        assert config.lemmatize is True
        assert config.remove_stopwords is True
        assert config.min_token_length == 3
        assert config.spacy_model == "ro_core_news_sm"

    def test_config_override(self):
        """Test that config values can be overridden."""
        from nlp_topic_modeling.preprocessing import PreprocessingConfig

        config = PreprocessingConfig(
            lowercase=False,
            min_token_length=5,
            lemmatize=False,
        )

        assert config.lowercase is False
        assert config.min_token_length == 5
        assert config.lemmatize is False
        # Other values should still be defaults
        assert config.normalize_diacritics is True


# Integration tests that require spaCy model
@pytest.mark.skipif(
    False,  # Enable when spaCy model is installed
    reason="Requires spaCy Romanian model (ro_core_news_sm)"
)
class TestTokenizerIntegration:
    """Integration tests for tokenizer (require spaCy model)."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        from nlp_topic_modeling.preprocessing import RomanianTokenizer

        tokenizer = RomanianTokenizer()
        tokens = tokenizer.tokenize_and_lemmatize("Fetele merg la școală")

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_returns_lemmas(self):
        """Test that tokenization returns lemmas."""
        from nlp_topic_modeling.preprocessing import RomanianTokenizer

        tokenizer = RomanianTokenizer()
        tokens = tokenizer.tokenize_and_lemmatize("Fetele merg")

        # Should contain base forms
        assert all(isinstance(t, str) for t in tokens)


@pytest.mark.skipif(
    False,  # Enable when spaCy model is installed
    reason="Requires spaCy Romanian model (ro_core_news_sm)"
)
class TestPipelineIntegration:
    """Integration tests for the full pipeline (require spaCy model)."""

    def test_preprocess_single_document(self):
        """Test preprocessing a single document."""
        from nlp_topic_modeling.preprocessing import RomanianPreprocessor

        preprocessor = RomanianPreprocessor()
        result = preprocessor.preprocess("Fetele merg la $NE$ pentru a studia.")

        assert 'original' in result
        assert 'cleaned' in result
        assert 'tokens' in result
        assert 'text' in result

        # $NE$ should be removed
        assert '$NE$' not in result['cleaned']
        assert '$ne$' not in result['cleaned'].lower()

    def test_preprocess_dataframe(self):
        """Test preprocessing a DataFrame."""
        from nlp_topic_modeling.preprocessing import RomanianPreprocessor

        df = pd.DataFrame({
            'sample': [
                "Text cu $NE$ și diacritice vechi: şcoală",
                "Alt text de test"
            ]
        })

        preprocessor = RomanianPreprocessor()
        result_df = preprocessor.preprocess_dataframe(df)

        assert 'clean_text' in result_df.columns
        assert 'tokens' in result_df.columns
        assert len(result_df) == 2

    def test_ne_tokens_removed_from_output(self):
        """Test that $NE$ tokens are completely removed."""
        from nlp_topic_modeling.preprocessing import preprocess_documents

        df = pd.DataFrame({
            'sample': [
                "$NE$ a mers la $NE$ pentru $NE$",
            ]
        })

        result = preprocess_documents(df)

        # Should not contain any NE markers
        assert not result['clean_text'].str.contains(r'\$NE\$', regex=True).any()

    def test_diacritics_normalized_in_output(self):
        """Test that diacritics are normalized in output."""
        from nlp_topic_modeling.preprocessing import preprocess_documents

        df = pd.DataFrame({
            'sample': [
                "Şcoala ţării noastre",
            ]
        })

        result = preprocess_documents(df)

        # Should not contain old cedilla characters
        assert not result['clean_text'].str.contains('ş|ţ', regex=True).any()
