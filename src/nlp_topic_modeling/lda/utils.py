def get_word_weightings(lda_pipeline):
    lda = lda_pipeline.named_steps['lda']
    topics = lda.components_
    topic_words_weighting = [list(reversed(sorted(t))) for t in topics]
    return topic_words_weighting


def link_topics_and_weightings(topic_words, topic_words_weighting):
    """
    Pair every words with their weightings for topics into tuples of (), for each topic.

    :param topic_words: a 2D array of shape [topics, top_words]
    :param topic_words_weighting: a 2D array of shape [topics, top_words_weightings]
    :return: A list of list containing tuples of shape [topics, list((top_word, top_words_weighting) for each word)].
        In other words, it is a 3D list of shape [topics, number_of_top_words, 2] where the 2 is two entries such as
        (top_word, top_words_weighting).
    """

    topics_and_words = [
        [
            (word, weightings)
            for word, weightings in zip(word_list, weightings_list)
        ]
        for word_list, weightings_list in zip(topic_words, topic_words_weighting)
    ]
    return topics_and_words


def get_top_comments(comments, transformed_comments):
    """
    :param comments: a list of comments.
    :param transformed_comments: a list of the class probabilies for comments.
        The class probabilities could be like [0.2, 0.98] in the inner dimension.
    :return: a list of the #1 top comment for each topic.
    """
    top_comments_idx = transformed_comments.argmax(0)  # top probability's index for each topic
    top_comments_strings = np.array(comments)[top_comments_idx].tolist()
    return top_comments_strings


def split_1_grams_from_n_grams(topics_weightings):
    """
    Pair every words with their weightings for topics into dicts, for each topic.

    :param topics_weightings: it is a 3D list of shape [topics, number_of_top_words, 2] where
        the 2 is two entries such as (top_word, top_words_weighting).
    :return: Two arrays similar to the input array where the 1-grams were splitted from the
        n-grams. The n-grams are in the second array and the 1-grams in the first one.
    """
    _1_grams = [[] for _ in range(len(topics_weightings))]
    _n_grams = [[] for _ in range(len(topics_weightings))]

    for i, topic_words_list in enumerate(topics_weightings):
        for word, weighting in topic_words_list:
            tuple_entries = (word, weighting)
            if ' ' in word:
                _n_grams[i].append(tuple_entries)
            else:
                _1_grams[i].append(tuple_entries)
    return _1_grams, _n_grams
