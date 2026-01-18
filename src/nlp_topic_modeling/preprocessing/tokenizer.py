"""spaCy-based tokenization and lemmatization for Romanian text."""

from typing import NamedTuple
import spacy
from spacy.language import Language

from nlp_topic_modeling.core.logging import get_logger

logger = get_logger(__name__)

# Default POS tags for topic modeling (R4.1)
# Focus on "topical skeleton" - nouns and adjectives, not verbs (writing style)
TOPIC_POS_TAGS = {'NOUN', 'PROPN', 'ADJ'}


class Token(NamedTuple):
    """Represents a processed token with linguistic information."""
    text: str
    pos: str
    lemma: str


class RomanianTokenizer:
    """Romanian tokenizer using spaCy for POS-aware lemmatization.

    This class provides tokenization with Part-of-Speech tagging and
    context-aware lemmatization for Romanian text. POS tagging is
    essential for accurate lemmatization in fusional languages like
    Romanian where identical surface forms can have different lemmas
    depending on their grammatical role.

    Example:
        >>> tokenizer = RomanianTokenizer()
        >>> tokens = tokenizer.tokenize_and_lemmatize("Fetele merg la școală")
        >>> print(tokens)
        ['fată', 'merge', 'la', 'școală']
    """

    # Default model - small model for efficiency
    DEFAULT_MODEL = "ro_core_news_sm"

    def __init__(self, model_name: str | None = None):
        """Initialize the tokenizer with a spaCy model.

        Args:
            model_name: Name of the spaCy Romanian model to use.
                       Defaults to 'ro_core_news_sm'.

        Raises:
            OSError: If the spaCy model is not installed.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.nlp = self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> Language:
        """Load the spaCy model.

        Args:
            model_name: Name of the spaCy model to load

        Returns:
            Loaded spaCy Language model

        Raises:
            OSError: If model is not installed, with instructions to install
        """
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError as e:
            error_msg = (
                f"spaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            )
            logger.error(error_msg)
            raise OSError(error_msg) from e

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize text and return tokens with linguistic information.

        Performs tokenization, POS tagging, and lemmatization using spaCy.

        Args:
            text: Input text to tokenize

        Returns:
            List of Token namedtuples with (text, pos, lemma)
        """
        doc = self.nlp(text)
        return [
            Token(text=token.text, pos=token.pos_, lemma=token.lemma_)
            for token in doc
            if not token.is_space
        ]

    def tokenize_and_lemmatize(self, text: str) -> list[str]:
        """Tokenize text and return list of lemmas.

        This is a convenience method that returns only the lemmas,
        which is typically what's needed for topic modeling.

        Args:
            text: Input text to tokenize

        Returns:
            List of lemmas (lowercase)
        """
        doc = self.nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_space and not token.is_punct
        ]

    def tokenize_filtered(
        self,
        text: str,
        exclude_pos: set[str] | None = None
    ) -> list[str]:
        """Tokenize and lemmatize, filtering by POS tags.

        Args:
            text: Input text to tokenize
            exclude_pos: Set of POS tags to exclude (e.g., {'PUNCT', 'SPACE', 'NUM'})

        Returns:
            List of lemmas with specified POS tags excluded
        """
        if exclude_pos is None:
            exclude_pos = {'PUNCT', 'SPACE', 'SYM', 'X'}

        doc = self.nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ not in exclude_pos
        ]

    def get_tokens_with_pos(self, text: str) -> list[tuple[str, str]]:
        """Get tokens with their POS tags (for debugging/analysis).

        Args:
            text: Input text to analyze

        Returns:
            List of (token, pos_tag) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def tokenize_pos_filtered(
        self,
        text: str,
        include_pos: set[str] | None = None
    ) -> list[str]:
        """Tokenize and lemmatize, keeping only specified POS tags.

        For topic modeling, we typically keep NOUN, PROPN, ADJ to focus on
        the "topical skeleton" rather than writing style (verbs). This
        implements the Semantic Compression principle (R4.1).

        Args:
            text: Input text to tokenize
            include_pos: Set of POS tags to include. Defaults to TOPIC_POS_TAGS
                        (NOUN, PROPN, ADJ).

        Returns:
            List of lemmas for tokens matching the specified POS tags

        Example:
            >>> tokenizer = RomanianTokenizer()
            >>> tokenizer.tokenize_pos_filtered("Ministrul a declarat că economia crește")
            ['ministru', 'economie']  # Only nouns, no verbs
        """
        if include_pos is None:
            include_pos = TOPIC_POS_TAGS

        doc = self.nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in include_pos and not token.is_space
        ]
