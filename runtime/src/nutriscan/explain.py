"""Section 7.5 explainability (SHAP).

Linear models (TF-IDF + logistic regression): with an independent-feature (interventional) background, the SHAP value
of feature j for class k is exactly  coef[k, j] * (x_j - E[x_j]),  so it is computed directly here (no `shap`
dependency, no sampling error). E[x] is the mean TF-IDF row of the training data, stored by the predictor.

Tree models (rf_features, gbm_features_lsa): `shap.TreeExplainer` on the engineered features when the `shap` package
is installed, otherwise permutation importance as a clearly labelled fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models.zoo import linear_member


def _clean(name: str) -> str:
    return name.replace("word__", "").replace("char__", "chars:")


def _member_label(model, pipe, class_label):
    """Voting/stacking ensembles fit their members on label-ENCODED targets (0..n-1); translate the label for them."""
    enc = getattr(model, "le_", None) or getattr(model, "_label_encoder", None)
    return enc.transform([class_label])[0] if enc is not None and pipe is not model else class_label


def _coef_row(clf, class_label) -> np.ndarray:
    classes = list(clf.classes_)
    if len(classes) == 2:                     # binary: one coef row for classes_[1]
        return clf.coef_[0] * (1.0 if class_label == classes[1] else -1.0)
    return clf.coef_[classes.index(class_label)]


def linear_shap(model, text: str, class_label, background: np.ndarray | None = None, k: int = 8,
                words_only: bool = False) -> list[tuple[str, float]]:
    """Top-k features pushing `text` TOWARD `class_label` (largest positive SHAP values).

    words_only drops character n-gram features ("chars: e1"): still part of the model, but meaningless to a reader.
    Without a background (older artifacts) this falls back to coef * x, the contribution relative to an empty list.
    """
    pipe = linear_member(model)
    if pipe is None:
        return []
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = np.asarray(tfidf.transform([text]).todense()).ravel()
    base = background if background is not None and len(background) == len(x) else 0.0
    phi = _coef_row(clf, _member_label(model, pipe, class_label)) * (x - base)
    phi = np.where(x > 0, phi, 0.0)           # report features present in this product; absent ones are background terms
    names = tfidf.get_feature_names_out()
    if words_only:
        phi = np.where(np.char.startswith(names.astype(str), "char__"), 0.0, phi)
    idx = np.argsort(-phi)[:k]
    return [(_clean(names[i]), float(phi[i])) for i in idx if phi[i] > 0]


# kept for backwards compatibility with earlier code and tests
def top_terms(pipeline, text: str, class_label, k: int = 8) -> list[tuple[str, float]]:
    return linear_shap(pipeline, text, class_label, None, k)


def linear_shap_summary(model, texts: list[str], class_label, background: np.ndarray | None = None,
                        top: int = 20, chunk: int = 64) -> pd.DataFrame:
    """Global importance for one class: mean |SHAP| over `texts` (use a few hundred rows).
    `toward_class` is the coefficient sign: True when the feature, if present, pushes toward the class."""
    pipe = linear_member(model)
    if pipe is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "toward_class"])
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    coef = _coef_row(clf, _member_label(model, pipe, class_label))
    X = tfidf.transform(list(texts))
    base = background if background is not None else np.asarray(X.mean(axis=0)).ravel()
    abs_sum = np.zeros_like(coef)
    for s in range(0, X.shape[0], chunk):
        abs_sum += np.abs((np.asarray(X[s:s + chunk].todense()) - base) * coef).sum(axis=0)
    n = max(X.shape[0], 1)
    names = tfidf.get_feature_names_out()
    idx = np.argsort(-abs_sum)[:top]
    return pd.DataFrame({"feature": [_clean(names[i]) for i in idx], "mean_abs_shap": abs_sum[idx] / n,
                         "toward_class": coef[idx] > 0})


def tree_shap_summary(model, texts: list[str], y=None, class_label=None, top: int = 20) -> tuple[pd.DataFrame, str]:
    """Global importance for a feature-based tree pipeline (steps 'feats' -> 'clf').

    Returns (table, method) where method is 'shap.TreeExplainer' or 'permutation_importance (shap not installed)'.
    """
    feats, clf = model.named_steps["feats"], model.named_steps["clf"]
    Xf = feats.transform(list(texts))
    Xf = Xf.toarray() if hasattr(Xf, "toarray") else np.asarray(Xf)
    try:
        names = list(feats.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xf.shape[1])]
    try:
        import shap
        sv = shap.TreeExplainer(clf).shap_values(Xf, check_additivity=False)
        classes = list(clf.classes_)
        ci = classes.index(class_label) if class_label in classes else 0
        vals = sv[ci] if isinstance(sv, list) else (sv[..., ci] if np.ndim(sv) == 3 else sv)
        imp, method = np.abs(vals).mean(axis=0), "shap.TreeExplainer"
    except ImportError:
        from sklearn.inspection import permutation_importance
        if y is None:
            raise ValueError("permutation fallback needs y")
        r = permutation_importance(clf, Xf, np.asarray(y), n_repeats=3, random_state=0, scoring="f1_macro")
        imp, method = r.importances_mean, "permutation_importance (shap not installed)"
    idx = np.argsort(-imp)[:top]
    return pd.DataFrame({"feature": [_clean(str(names[i])) for i in idx], "importance": imp[idx]}), method
