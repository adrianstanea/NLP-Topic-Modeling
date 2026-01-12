
import numpy as np
from sklearn.pipeline import Pipeline
from artifici_lda.logic.stop_words_remover import StopWordsRemover
from artifici_lda.logic.stemmer import Stemmer
from artifici_lda.logic.count_vectorizer import CountVectorizer
from artifici_lda.logic.lda import LDA

from .config import SUPPORTED_LANGUAGES, get_lda_pipeline_params
from .utils import (
    get_word_weightings,
    link_topics_and_weightings,
    get_top_comments,
    split_1_grams_from_n_grams,
)


def lda_pipeline(docs_list, n_topics, language=None, stopwords=None):
    """ Train an LDA and transform the comments.

    :param docs_list: list of documents (texts)
    :param n_topics: number of topics to extract
    :param language: language of the documents
    :param stopwords: list of stopwords to use

    :return: trained LDA model and transformed documents

    :raises ValueError: if the language is not supported
    """
    assert language in SUPPORTED_LANGUAGES, f"Language '{language}' is not supported. Supported languages: {SUPPORTED_LANGUAGES}"
    assert isinstance(docs_list, list) and all(isinstance(doc, str)
                                               for doc in docs_list), "docs_list must be a list of strings."
    assert isinstance(n_topics, int) and n_topics > 0, "n_topics must be a positive integer."
    assert stopwords is None or isinstance(stopwords, list), "stopwords must be a list or None."

    lda_params = get_lda_pipeline_params(
        n_topics=n_topics,
        language=language,
        stopwords=stopwords
    )

    lda_pipeline = Pipeline([
        ('stopwords', StopWordsRemover()),
        ('stemmer', Stemmer()),
        ('count_vect', CountVectorizer()),
        ('lda', LDA()),
    ]).set_params(**lda_params)

    # Fit the data
    transformed_docs = lda_pipeline.fit_transform(docs_list)

    top_comments = get_top_comments(docs_list, transformed_docs)

    # Extract information about data
    topic_words = lda_pipeline.inverse_transform(Xt=None)
    topic_words_weighting = get_word_weightings(lda_pipeline)
    topics_words_and_weightings = link_topics_and_weightings(topic_words, topic_words_weighting)

    # Cleanup return values
    _1_grams, _2_grams = split_1_grams_from_n_grams(topics_words_and_weightings)
    return transformed_docs, top_comments, _1_grams, _2_grams
