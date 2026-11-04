"""Section 6.3 model: TF-IDF (word + char n-grams) with Logistic Regression."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def make_tfidf_pipeline(class_weight: str | None = "balanced", C: float = 1.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats),
                     ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000))])
