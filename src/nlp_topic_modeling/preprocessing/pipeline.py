"""Main preprocessing pipeline for Romanian text."""

from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd
from tqdm.auto import tqdm

from nlp_topic_modeling.core.logging import get_logger
from .normalizers import (
    lowercase,
    normalize_diacritics,
    remove_ne_tokens,
    remove_urls,
    remove_html_tags,
    clean_whitespace,
    keep_romanian_chars,
)
from .stopwords import get_all_stopwords, filter_stopwords
from .tokenizer import RomanianTokenizer

logger = get_logger(__name__)


class PreprocessedDocument(TypedDict):
    """Result of preprocessing a single document."""
    original: str
    cleaned: str
    tokens: list[str]
    text: str


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline.

    Attributes:
        lowercase: Whether to convert text to lowercase
        normalize_diacritics: Whether to standardize Romanian diacritics
        remove_ne_tokens: Whether to remove $NE$ placeholders
        remove_urls: Whether to remove URLs
        remove_html: Whether to strip HTML tags
        keep_only_letters: Whether to remove digits and punctuation
        lemmatize: Whether to perform lemmatization
        remove_stopwords: Whether to filter stopwords
        min_token_length: Minimum token length (shorter tokens are filtered)
        spacy_model: Name of the spaCy model for tokenization
    """
    lowercase: bool = True
    normalize_diacritics: bool = True
    remove_ne_tokens: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    keep_only_letters: bool = False
    lemmatize: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 3
    spacy_model: str = "ro_core_news_sm"
    custom_stopwords: set[str] = field(default_factory=set)


class RomanianPreprocessor:
    """Main preprocessing pipeline for Romanian text.

    This class implements a configurable text preprocessing pipeline
    following the principle of immutability (original text is preserved).

    The pipeline follows this order of operations:
    1. Lowercasing
    2. Diacritic normalization (ș, ț)
    3. Regex cleaning (URLs, $NE$ markers, HTML)
    4. Lemmatization (spaCy with POS tagging)
    5. Stopword removal

    Example:
        >>> config = PreprocessingConfig(min_token_length=2)
        >>> preprocessor = RomanianPreprocessor(config)
        >>> result = preprocessor.preprocess("Fetele merg la $NE$ !")
        >>> print(result['tokens'])
        ['fată', 'merge']
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig()
        self._tokenizer = None
        self._stopwords = None

    @property
    def tokenizer(self) -> RomanianTokenizer:
        """Lazy-load the tokenizer to avoid loading spaCy until needed."""
        if self._tokenizer is None:
            self._tokenizer = RomanianTokenizer(self.config.spacy_model)
        return self._tokenizer

    @property
    def stopwords(self) -> set[str]:
        """Get the combined stopwords set."""
        if self._stopwords is None:
            self._stopwords = get_all_stopwords()
            if self.config.custom_stopwords:
                self._stopwords = self._stopwords.union(self.config.custom_stopwords)
        return self._stopwords

    def _normalize_text(self, text: str) -> str:
        """Apply text normalization steps.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Step 1: Lowercase
        if self.config.lowercase:
            text = lowercase(text)

        # Step 2: Diacritic normalization
        if self.config.normalize_diacritics:
            text = normalize_diacritics(text)

        # Step 3: Regex cleaning
        if self.config.remove_html:
            text = remove_html_tags(text)

        if self.config.remove_urls:
            text = remove_urls(text)

        if self.config.remove_ne_tokens:
            text = remove_ne_tokens(text)

        # Clean up whitespace after removals
        text = clean_whitespace(text)

        return text

    def _tokenize_and_filter(self, text: str) -> list[str]:
        """Tokenize text and apply filtering.

        Args:
            text: Normalized text

        Returns:
            List of filtered tokens
        """
        # Step 4: Lemmatization
        if self.config.lemmatize:
            tokens = self.tokenizer.tokenize_and_lemmatize(text)
        else:
            # Simple whitespace tokenization if lemmatization disabled
            tokens = text.split()

        # Step 5: Stopword removal
        if self.config.remove_stopwords:
            tokens = filter_stopwords(
                tokens,
                self.stopwords,
                self.config.min_token_length
            )
        elif self.config.min_token_length > 0:
            # Still apply length filtering even without stopword removal
            tokens = [t for t in tokens if len(t) >= self.config.min_token_length]

        return tokens

    def preprocess(self, text: str) -> PreprocessedDocument:
        """Preprocess a single document.

        Applies the full preprocessing pipeline while preserving
        the original text (immutability principle).

        Args:
            text: Input document text

        Returns:
            Dictionary with:
            - 'original': Original text
            - 'cleaned': Normalized text before tokenization
            - 'tokens': List of processed tokens
            - 'text': Rejoined clean text for topic modeling
        """
        # Normalize text
        cleaned = self._normalize_text(text)

        # Tokenize and filter
        tokens = self._tokenize_and_filter(cleaned)

        # Rejoin for models that need string input
        processed_text = ' '.join(tokens)

        return PreprocessedDocument(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            text=processed_text
        )

    def preprocess_documents(
        self,
        documents: list[str],
        show_progress: bool = True
    ) -> list[PreprocessedDocument]:
        """Preprocess a list of documents.

        Args:
            documents: List of document texts
            show_progress: Whether to show a progress bar

        Returns:
            List of PreprocessedDocument dictionaries
        """
        results = []
        iterator = tqdm(documents, desc="Preprocessing", disable=not show_progress)

        for doc in iterator:
            results.append(self.preprocess(doc))

        return results

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "sample",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Preprocess a DataFrame, adding new columns for processed text.

        This method follows the immutability principle by adding new
        columns rather than modifying the original text column.

        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            show_progress: Whether to show a progress bar

        Returns:
            DataFrame with added columns:
            - 'clean_text': Normalized and rejoined text
            - 'tokens': List of processed tokens
        """
        logger.info(f"Preprocessing {len(df)} documents from column '{text_column}'")

        # Copy to avoid modifying original
        df = df.copy()

        # Preprocess all documents
        results = self.preprocess_documents(
            df[text_column].tolist(),
            show_progress=show_progress
        )

        # Add new columns
        df['clean_text'] = [r['text'] for r in results]
        df['tokens'] = [r['tokens'] for r in results]

        logger.info("Preprocessing complete")
        return df


def preprocess_text(text: str, **config_kwargs) -> str:
    """Convenience function to preprocess a single text document.

    Args:
        text: Input text
        **config_kwargs: Configuration overrides for PreprocessingConfig

    Returns:
        Preprocessed text as a string
    """
    config = PreprocessingConfig(**config_kwargs)
    preprocessor = RomanianPreprocessor(config)
    result = preprocessor.preprocess(text)
    return result['text']


def preprocess_documents(
    documents: list[str] | pd.DataFrame,
    text_column: str = "sample",
    show_progress: bool = True,
    **config_kwargs
) -> pd.DataFrame:
    """Convenience function to batch preprocess documents.

    Args:
        documents: Either a list of strings or a DataFrame
        text_column: Column name if documents is a DataFrame
        show_progress: Whether to show progress bar
        **config_kwargs: Configuration overrides for PreprocessingConfig

    Returns:
        DataFrame with 'clean_text' and 'tokens' columns
    """
    config = PreprocessingConfig(**config_kwargs)
    preprocessor = RomanianPreprocessor(config)

    if isinstance(documents, pd.DataFrame):
        return preprocessor.preprocess_dataframe(
            documents,
            text_column=text_column,
            show_progress=show_progress
        )
    else:
        # Convert list to DataFrame
        df = pd.DataFrame({'sample': documents})
        return preprocessor.preprocess_dataframe(
            df,
            text_column='sample',
            show_progress=show_progress
        )
