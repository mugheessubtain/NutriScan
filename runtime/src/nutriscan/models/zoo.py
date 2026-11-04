"""Section 7 model zoo: baselines (7.2), improved models (7.3), imbalance variants (7.3.6), tuning spaces and
ensembles (7.4). Every model is an sklearn estimator whose input X is a list/Series of normalised ingredient strings,
so all of them plug into the same predictors, VotingClassifier and evaluation code.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .. import config
from ..features import TextFeatures, text_features
from .tfidf_lr import make_tfidf_pipeline


def _tfidf_union() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])


# ---------------------------------------------------------------- baselines (7.2)
class KeywordRule(BaseEstimator, ClassifierMixin):
    """'Contains any additive marker or E-number' rule. Learns one class for marker rows and one for marker-free rows
    (the mode of each group in training), which for NOVA is the plan's 'any additive or E-number means NOVA 4'."""

    def fit(self, X, y):
        y = np.asarray(y)
        has = np.array([self._has_marker(t) for t in X])
        self.classes_ = np.unique(y)
        mode = lambda v: (v if len(v) else y)
        self.marker_class_ = _mode(mode(y[has]))
        self.clean_class_ = _mode(mode(y[~has]))
        return self

    @staticmethod
    def _has_marker(t: str) -> bool:
        f = text_features(t)
        return f[1] > 0 or f[8] > 0          # n_e_numbers, n_markers

    def predict(self, X):
        return np.array([self.marker_class_ if self._has_marker(t) else self.clean_class_ for t in X])

    def predict_proba(self, X):
        pred = self.predict(X)
        return (pred[:, None] == self.classes_[None, :]).astype(float)


def _mode(v):
    vals, counts = np.unique(v, return_counts=True)
    return vals[np.argmax(counts)]


def _count_features(X):
    return np.array([text_features(t)[:2] for t in X], dtype=float)   # n_ingredients, n_e_numbers


def make_baseline(name: str):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "keyword_rule":
        return KeywordRule()
    if name == "count_features_lr":
        return Pipeline([("feats", FunctionTransformer(_count_features)), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000))])
    raise ValueError(name)


BASELINES = ["majority", "keyword_rule", "count_features_lr"]


# ---------------------------------------------------------------- improved models (7.3)
def make_model(name: str, class_weight: str | None = "balanced", seed: int = config.RANDOM_STATE):
    """name: tfidf_lr | tfidf_svm | rf_features | gbm_features_lsa | voting | stacking"""
    if name == "tfidf_lr":
        return make_tfidf_pipeline(class_weight=class_weight)
    if name == "tfidf_svm":
        # LinearSVC has no predict_proba; calibration adds it so the model can join a soft-voting ensemble
        return Pipeline([("tfidf", _tfidf_union()),
                         ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight=class_weight), cv=3))])
    if name == "rf_features":
        return Pipeline([("feats", TextFeatures()),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                        class_weight=class_weight, random_state=seed))])
    if name == "gbm_features_lsa":
        # gradient boosting on engineered features + a 100-d latent semantic (TF-IDF -> SVD) text embedding
        feats = FeatureUnion([("eng", TextFeatures()),
                              ("lsa", Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                                                ("svd", TruncatedSVD(n_components=100, random_state=seed))]))])
        return Pipeline([("feats", feats),
                         ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                                                class_weight=class_weight, random_state=seed))])
    if name == "voting":
        return make_ensemble("voting", class_weight=class_weight, seed=seed)
    if name == "stacking":
        return make_ensemble("stacking", class_weight=class_weight, seed=seed)
    raise ValueError(name)


IMPROVED = ["tfidf_lr", "tfidf_svm", "rf_features", "gbm_features_lsa"]


def make_ensemble(kind: str, members: dict | None = None, class_weight="balanced", seed: int = config.RANDOM_STATE):
    """Soft voting or stacking over (already tuned, unfitted) member estimators. Member names are kept so the UI can
    reach the linear member ('tfidf_lr') for per-product explanations."""
    members = members or {n: make_model(n, class_weight, seed) for n in IMPROVED}
    est = [(n, clone(m)) for n, m in members.items()]
    if kind == "voting":
        return VotingClassifier(est, voting="soft", n_jobs=1)
    if kind == "stacking":
        return StackingClassifier(est, final_estimator=LogisticRegression(max_iter=2000, class_weight=class_weight),
                                  stack_method="predict_proba", cv=3, n_jobs=1)
    raise ValueError(kind)


# ---------------------------------------------------------------- tuning spaces (7.4)
PARAM_SPACES = {
    "tfidf_lr": {"clf__C": loguniform(0.1, 30)},
    "tfidf_svm": {"clf__estimator__C": loguniform(0.05, 5)},
    "rf_features": {"clf__n_estimators": randint(150, 500), "clf__max_depth": [None, 20, 40],
                    "clf__min_samples_leaf": [1, 2, 5], "clf__max_features": ["sqrt", 0.5]},
    "gbm_features_lsa": {"clf__learning_rate": loguniform(0.03, 0.3), "clf__max_leaf_nodes": randint(15, 63),
                         "clf__l2_regularization": loguniform(1e-3, 1.0)},
}


# ---------------------------------------------------------------- imbalance variants (7.3.6)
def make_imbalance_variant(model: str, strategy: str, seed: int = config.RANDOM_STATE, smote_k: int = 5):
    """strategy: none | class_weights | random_oversampling | smote.
    SMOTE is only offered for the dense engineered-feature model (interpolating sparse TF-IDF rows is not meaningful).
    smote_k must be below the size of the rarest class (step6 passes min(5, n_rarest - 1))."""
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    if strategy in ("none", "class_weights"):
        return make_model(model, class_weight="balanced" if strategy == "class_weights" else None, seed=seed)
    base = make_model(model, class_weight=None, seed=seed)
    steps = list(base.steps)
    if strategy == "random_oversampling":
        sampler = RandomOverSampler(random_state=seed)
        if model == "rf_features":                         # oversample after featurising (dense)
            return ImbPipeline([steps[0], ("sample", sampler), steps[1]])
        return ImbPipeline([("to_2d", FunctionTransformer(_to_2d)), ("sample", sampler),
                            ("from_2d", FunctionTransformer(_from_2d)), *steps])
    if strategy == "smote":
        if model != "rf_features":
            raise ValueError("SMOTE is only used on dense engineered features (rf_features)")
        return ImbPipeline([steps[0], ("sample", SMOTE(k_neighbors=smote_k, random_state=seed)), steps[1]])
    raise ValueError(strategy)


def _to_2d(X):
    return np.asarray(list(X), dtype=object).reshape(-1, 1)


def _from_2d(X):
    return np.asarray(X).ravel()


def linear_member(model):
    """Return the TF-IDF + linear Pipeline inside a model (itself, or the 'tfidf_lr' member of an ensemble), else None."""
    if isinstance(model, Pipeline) and "tfidf" in model.named_steps and hasattr(model.named_steps["clf"], "coef_"):
        return model
    named = getattr(model, "named_estimators_", None)
    if named and "tfidf_lr" in named:
        return named["tfidf_lr"]
    return None
