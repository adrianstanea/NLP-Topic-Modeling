ROMANIAN = 'romanian'

SUPPORTED_LANGUAGES = [ROMANIAN]

# None indicates that it should be configured via the get_lda_pipeline_params function
LDA_PIPELINE_PARAMS_WORDS_CFG = {
    'stopwords__stopwords': None,
    'stemmer__language': None,
    'count_vect__max_df': 0.98,
    'count_vect__min_df': 2,
    'count_vect__max_features': 10000,
    'count_vect__ngram_range': (1, 2),
    'count_vect__strip_accents': None,
    'lda__n_components': None,
    'lda__max_iter': 750,
    'lda__learning_decay': 0.5,
    'lda__learning_method': 'online',
    'lda__learning_offset': 10,
    'lda__batch_size': 25,
    'lda__n_jobs': -1,  # Use all CPUs
}


def get_lda_pipeline_params(n_topics: int, language: str = None, stopwords: list | None = None) -> dict:
    """ Get LDA pipeline parameters with specified number of topics and stopwords.

    :param n_topics: number of topics to extract
    :param language: language of the documents
    :param stopwords: list of stopwords to use

    :return: dictionary of LDA pipeline parameters
    """
    params = LDA_PIPELINE_PARAMS_WORDS_CFG.copy()

    if n_topics is not None:
        assert 'lda__n_components' in list(
            params.keys()), "LDA parameter 'n_components' not found in the configuration."
        params['lda__n_components'] = n_topics

    if language is not None:
        assert 'stemmer__language' in list(
            params.keys()), "Stemmer parameter 'language' not found in the configuration."
        params['stemmer__language'] = language.lower()

    if stopwords is not None:
        assert 'stopwords__stopwords' in list(
            params.keys()), "Stopwords parameter 'stopwords' not found in the configuration."
        params['stopwords__stopwords'] = stopwords

    return params
