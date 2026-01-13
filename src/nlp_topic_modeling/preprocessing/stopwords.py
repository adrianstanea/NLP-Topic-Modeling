"""Stopword list management for Romanian text preprocessing."""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_romanian_stopwords() -> set[str]:
    """Get combined Romanian stopwords from NLTK, stop-words, and stopwordsiso.

    This function caches the result to avoid repeated computation.

    Returns:
        Set of Romanian stopwords from multiple sources
    """
    stopwords_set = set()

    # NLTK stopwords
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        import nltk
        try:
            stopwords_set.update(nltk_stopwords.words('romanian'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stopwords_set.update(nltk_stopwords.words('romanian'))
    except ImportError:
        pass

    # stop-words library
    try:
        from stop_words import get_stop_words
        stopwords_set.update(get_stop_words('romanian'))
    except ImportError:
        pass

    # stopwordsiso library
    try:
        import stopwordsiso
        stopwords_set.update(stopwordsiso.stopwords('ro'))
    except ImportError:
        pass

    return stopwords_set


def get_ne_stopwords() -> set[str]:
    """Get MAROCO-specific stopwords.

    Returns:
        Set containing the $NE$ placeholder and variants
    """
    return {'$ne$', 'ne', '$NE$'}


def get_domain_stopwords() -> set[str]:
    """Get domain-specific noise words common in Romanian news text.

    These are words that frequently appear but don't contribute
    to topic discrimination.

    Returns:
        Set of domain-specific stopwords
    """
    return {
        # Common discourse markers
        'apoi', 'deci', 'totuși', 'totusi', 'însă', 'insa',
        'așadar', 'asadar', 'prin urmare', 'de asemenea',
        # Temporal markers
        'acum', 'ieri', 'azi', 'astăzi', 'astazi', 'mâine', 'maine',
        # Common verbs that don't add meaning
        'spune', 'spus', 'zice', 'zis', 'face', 'făcut', 'facut',
        # Filler words
        'poate', 'chiar', 'doar', 'tocmai', 'deja',
    }


def get_all_stopwords(min_length: int = 0) -> set[str]:
    """Get combined stopword list from all sources.

    Args:
        min_length: Minimum token length to include (tokens shorter than
                   this will be considered stopwords). Set to 0 to disable.

    Returns:
        Combined set of all stopwords
    """
    all_stopwords = set()
    all_stopwords.update(get_romanian_stopwords())
    all_stopwords.update(get_ne_stopwords())
    all_stopwords.update(get_domain_stopwords())

    # Normalize to lowercase
    all_stopwords = {word.lower() for word in all_stopwords}

    return all_stopwords


def is_stopword(token: str, stopwords: set[str] | None = None, min_length: int = 0) -> bool:
    """Check if a token is a stopword.

    Args:
        token: Token to check
        stopwords: Set of stopwords to check against. If None, uses get_all_stopwords().
        min_length: Tokens shorter than this are considered stopwords.

    Returns:
        True if token is a stopword, False otherwise
    """
    if stopwords is None:
        stopwords = get_all_stopwords()

    token_lower = token.lower()

    # Check length constraint
    if min_length > 0 and len(token_lower) < min_length:
        return True

    return token_lower in stopwords


def filter_stopwords(
    tokens: list[str],
    stopwords: set[str] | None = None,
    min_length: int = 0
) -> list[str]:
    """Filter stopwords from a list of tokens.

    Args:
        tokens: List of tokens to filter
        stopwords: Set of stopwords. If None, uses get_all_stopwords().
        min_length: Tokens shorter than this are filtered out.

    Returns:
        List of tokens with stopwords removed
    """
    if stopwords is None:
        stopwords = get_all_stopwords()

    return [
        token for token in tokens
        if not is_stopword(token, stopwords, min_length)
    ]
