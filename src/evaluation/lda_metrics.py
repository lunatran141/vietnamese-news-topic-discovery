
"""Evaluation metrics for M08 LDA topic experiments."""

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def extract_topic_words(model, feature_names, top_n=10):
    """Extract top words of each learned LDA topic."""
    feature_names = np.asarray(feature_names)

    if top_n < 1:
        raise ValueError("top_n must be positive")

    if model.components_.shape[1] != len(feature_names):
        raise ValueError("Model and vocabulary dimensions differ")

    topics = []

    for component in model.components_:
        top_indices = np.argsort(-component)[:top_n]
        topics.append(feature_names[top_indices].tolist())

    return topics


def topic_diversity(topics):
    """Ratio of unique words among all topic keywords."""
    words = [word for topic in topics for word in topic]

    if not words:
        raise ValueError("No topic words provided")

    return len(set(words)) / len(words)


def prepare_coherence_corpus(texts, vectorizer):
    """Tokenize validation texts using the trained BoW vocabulary."""
    analyzer = vectorizer.build_analyzer()
    vocabulary = set(vectorizer.get_feature_names_out())

    tokenized_texts = [
        [word for word in analyzer(text) if word in vocabulary]
        for text in texts
    ]

    tokenized_texts = [tokens for tokens in tokenized_texts if tokens]

    if not tokenized_texts:
        raise ValueError("Coherence corpus is empty")

    dictionary = Dictionary(tokenized_texts)

    return tokenized_texts, dictionary


def calculate_coherence(
    topics,
    tokenized_texts,
    dictionary,
    metric="c_v",
    processes=1,
):
    """Calculate coherence with Gensim on validation documents."""
    if metric not in {"c_v", "c_npmi"}:
        raise ValueError(f"Unsupported coherence metric: {metric}")

    valid_topics = [
        [word for word in topic if word in dictionary.token2id]
        for topic in topics
    ]

    if any(len(topic) < 2 for topic in valid_topics):
        raise ValueError(
            "A topic has fewer than two keywords "
            "present in the coherence corpus"
        )

    model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence=metric,
        processes=processes,
    )

    return float(model.get_coherence())
