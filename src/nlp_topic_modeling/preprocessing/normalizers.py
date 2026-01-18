"""Text normalization functions for Romanian preprocessing."""

import re
from typing import Callable

# Diacritic mapping: cedilla (legacy) -> comma-below (Unicode standard)
DIACRITIC_MAP = {
    'ş': 'ș',  # U+015F -> U+0219
    'ţ': 'ț',  # U+0163 -> U+021B
    'Ş': 'Ș',  # U+015E -> U+0218
    'Ţ': 'Ț',  # U+0162 -> U+021A
}

# Regex patterns
NE_TOKEN_PATTERN = re.compile(r'\$NE\$', re.IGNORECASE)
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+|'
    r'www\.[^\s<>"{}|\\^`\[\]]+'
)
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')


def normalize_diacritics(text: str) -> str:
    """Convert cedilla diacritics to comma-below Unicode standard.

    Maps legacy Romanian characters to modern Unicode:
    - ş (U+015F) -> ș (U+0219)
    - ţ (U+0163) -> ț (U+021B)
    - Ş (U+015E) -> Ș (U+0218)
    - Ţ (U+0162) -> Ț (U+021A)

    Args:
        text: Input text with potentially mixed diacritics

    Returns:
        Text with standardized comma-below diacritics
    """
    for old_char, new_char in DIACRITIC_MAP.items():
        text = text.replace(old_char, new_char)
    return text


def lowercase(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text: Input text

    Returns:
        Lowercase text
    """
    return text.lower()


def remove_ne_tokens(text: str) -> str:
    """Remove $NE$ named entity placeholders from MAROCO dataset.

    The MAROCO dataset uses $NE$ as a placeholder for named entities.
    These should be removed to prevent topic pollution.

    Args:
        text: Input text with potential $NE$ tokens

    Returns:
        Text with $NE$ tokens removed
    """
    return NE_TOKEN_PATTERN.sub('', text)


def remove_urls(text: str) -> str:
    """Remove URLs from text.

    Handles both http(s):// and www. prefixed URLs.

    Args:
        text: Input text with potential URLs

    Returns:
        Text with URLs removed
    """
    return URL_PATTERN.sub('', text)


def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text.

    Args:
        text: Input text with potential HTML tags

    Returns:
        Text with HTML tags removed
    """
    return HTML_TAG_PATTERN.sub('', text)


def clean_whitespace(text: str) -> str:
    """Normalize multiple spaces/newlines to single space.

    Args:
        text: Input text with potentially irregular whitespace

    Returns:
        Text with normalized whitespace, stripped of leading/trailing spaces
    """
    return WHITESPACE_PATTERN.sub(' ', text).strip()


def keep_romanian_chars(text: str) -> str:
    """Keep only Romanian alphabet characters and whitespace.

    Preserves: a-z, ă, â, î, ș, ț (and uppercase variants) plus whitespace.
    Removes: digits, punctuation, special characters.

    Args:
        text: Input text

    Returns:
        Text with only Romanian alphabet characters and spaces
    """
    # Pattern matches anything that is NOT a Romanian letter or whitespace
    pattern = re.compile(r'[^a-zA-ZăâîșțĂÂÎȘȚ\s]')
    return pattern.sub('', text)


# Pattern to match mid-word î (not at start or end of word)
# Uses lookbehind and lookahead to ensure î is surrounded by letters
MID_WORD_I_CIRCUMFLEX = re.compile(r'(?<=[a-zA-ZăâșțĂÂȘȚ])î(?=[a-zA-ZăâșțĂÂȘȚ])')
MID_WORD_I_CIRCUMFLEX_UPPER = re.compile(r'(?<=[a-zA-ZăâșțĂÂȘȚ])Î(?=[a-zA-ZăâșțĂÂȘȚ])')


def normalize_dialect(text: str) -> str:
    """Normalize î/â orthographic variation to Romanian standard (â-form).

    In Romanian orthography, both 'î' and 'â' represent the same sound /ɨ/.
    Modern Romanian standard uses:
    - 'î' at the beginning and end of words (e.g., "început", "coborî")
    - 'â' in the middle of words (e.g., "sunt" not "sînt", "când" not "cînd")

    The Moldovan/pre-1993 style uses 'î' everywhere. This function normalizes
    to the modern Romanian standard for cross-dialect consistency in MAROCO.

    Args:
        text: Input text with potentially mixed î/â usage

    Returns:
        Text with mid-word î normalized to â (Romanian standard)

    Examples:
        >>> normalize_dialect("sînt")
        'sânt'
        >>> normalize_dialect("cînd")
        'când'
        >>> normalize_dialect("început")  # Word-initial î stays
        'început'
    """
    # Replace mid-word î with â
    text = MID_WORD_I_CIRCUMFLEX.sub('â', text)
    text = MID_WORD_I_CIRCUMFLEX_UPPER.sub('Â', text)
    return text


def compose(*functions: Callable[[str], str]) -> Callable[[str], str]:
    """Compose multiple text transformation functions.

    Args:
        *functions: Variable number of str -> str functions

    Returns:
        A single function that applies all transformations in order
    """
    def composed(text: str) -> str:
        for func in functions:
            text = func(text)
        return text
    return composed
