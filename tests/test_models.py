import numpy as np
import pytest

from nutriscan import config
from nutriscan.explain import linear_shap, linear_shap_summary
from nutriscan.features import FEATURE_NAMES, feature_frame, text_features
from nutriscan.models import NovaPredictor
from nutriscan.models.zoo import BASELINES, IMPROVED, KeywordRule, make_baseline, make_imbalance_variant, make_model


def test_out_root_is_sandboxed():
    assert config.OUT_ROOT != config.PROJECT_ROOT          # conftest: tests never write into the project
    assert str(config.ARTIFACTS).startswith(str(config.OUT_ROOT))


def test_text_features():
    f = dict(zip(FEATURE_NAMES, text_features("sugar, wheat flour, palm oil, e471, flavouring")))
    assert f["n_ingredients"] == 5 and f["n_e_numbers"] == 1
    assert f["sugar_pos"] == 1 and f["sugar_in_first3"] == 1 and f["fat_pos"] == 3
    assert f["has_palm"] == 1 and f["has_flavouring"] == 1 and f["has_emulsifier"] == 1
    assert feature_frame(["", "potatoes"]).shape == (2, len(FEATURE_NAMES))


def test_keyword_rule():
    X = ["sugar, e102", "potatoes", "water, flavouring", "rolled oats"]
    m = KeywordRule().fit(X, [4, 1, 4, 1])
    assert list(m.predict(["salt, e330", "carrots"])) == [4, 1]


@pytest.mark.parametrize("name", BASELINES + IMPROVED)
def test_every_model_fits_and_predicts(sample_df, name):
    df = sample_df[sample_df["nova_group"].notna()].head(600)
    X, y = df["ingredients_norm"].tolist(), df["nova_group"].astype(int)
    model = (make_baseline(name) if name in BASELINES else make_model(name)).fit(X, y)
    pred = model.predict(X[:20])
    assert len(pred) == 20 and set(pred) <= set(y)


@pytest.mark.parametrize("model,strategy", [("tfidf_lr", "random_oversampling"), ("rf_features", "smote"),
                                            ("rf_features", "class_weights")])
def test_imbalance_variants(sample_df, model, strategy):
    df = sample_df[sample_df["nova_group"].notna()].head(600)
    df = df[df["nova_group"] != 2]                         # SMOTE needs > k samples per class
    m = make_imbalance_variant(model, strategy).fit(df["ingredients_norm"].tolist(), df["nova_group"].astype(int))
    assert len(m.predict(df["ingredients_norm"].tolist()[:5])) == 5


def test_linear_shap_is_exact(sample_df):
    """Sum of SHAP values over ALL features = logit(x) - logit(background) for the explained class."""
    p = NovaPredictor().train(sample_df)
    text = "sugar, glucose syrup, palm oil, e471, flavouring"
    tfidf, clf = p.model.named_steps["tfidf"], p.model.named_steps["clf"]
    x = np.asarray(tfidf.transform([text]).todense()).ravel()
    k = list(clf.classes_).index(4)
    phi = clf.coef_[k] * (x - p.background_)
    expected = (x @ clf.coef_[k]) - (p.background_ @ clf.coef_[k])
    assert np.isclose(phi.sum(), expected)
    top = linear_shap(p.model, text, 4, p.background_)
    assert top and all(v > 0 for _, v in top)
    summ = linear_shap_summary(p.model, sample_df["ingredients_norm"].head(50).tolist(), 4, p.background_, top=5)
    assert len(summ) == 5 and {"feature", "mean_abs_shap", "toward_class"} <= set(summ.columns)


def test_predictor_save_load_roundtrip(sample_df, tmp_path):
    p = NovaPredictor(kind="tfidf_lr").train(sample_df)
    path = tmp_path / "nova.joblib"
    p.save(path)
    q = NovaPredictor.load(path)
    assert q.kind == "tfidf_lr" and q.background_ is not None
    assert list(q.predict(["sugar, e102"])) == list(p.predict(["sugar, e102"]))


def test_shap_works_inside_voting_ensemble(sample_df):
    """Regression: ensemble members are fit on encoded labels (0..n-1), the app passes real labels (1..4 / 'poor')."""
    from nutriscan.models import TierPredictor
    for pred_cls, label in ((NovaPredictor, 4), (TierPredictor, "poor")):
        p = pred_cls(kind="voting").train(sample_df.head(800))
        drivers = linear_shap(p.model, "sugar, glucose syrup, palm oil, e471, flavouring", label, p.background_)
        assert drivers and all(v > 0 for _, v in drivers)
        assert not linear_shap_summary(p.model, sample_df["ingredients_norm"].head(30).tolist(), label, p.background_).empty
