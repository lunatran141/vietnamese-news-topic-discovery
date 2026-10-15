"""Feature extraction modules."""

from src.features.article_text import build_article_text
from src.features.tfidf import (
    create_tfidf_vectorizer,
    evaluate_validation_vocabulary,
    fit_tfidf_pipeline,
    load_stopwords,
    load_tfidf_artifacts,
    save_tfidf_artifacts,
    transform_tfidf_pipeline,
)

__all__ = [
    "build_article_text",
    "create_tfidf_vectorizer",
    "fit_tfidf_pipeline",
    "transform_tfidf_pipeline",
    "evaluate_validation_vocabulary",
    "save_tfidf_artifacts",
    "load_tfidf_artifacts",
    "load_stopwords",
]

