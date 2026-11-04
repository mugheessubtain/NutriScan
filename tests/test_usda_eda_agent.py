import numpy as np
import pandas as pd

from nutriscan import agent, eda
from nutriscan.alternatives import build_gallery
from nutriscan.evaluation import group_metrics, inject_ocr_noise
from nutriscan.expert_system import ExpertSystem
from nutriscan.parser import parse_ingredients
from nutriscan.pipeline import NutriScan
from nutriscan.usda import map_category, nutriscore_2017


def test_usda_category_mapping():
    assert map_category("Candy") == "sugary-snacks"
    assert map_category("Milk chocolate bars") == "sugary-snacks"
    assert map_category("Cheese") == "milk-and-dairy-products"
    assert map_category("Nut & Seed Butters") == "fats-and-sauces"
    assert map_category("Chips, Pretzels & Snacks") == "salty-snacks"
    assert map_category("Frozen Dinners & Entrees") == "composite-foods"
    assert map_category("Soda") == "beverages"
    assert map_category(None) is None


def test_nutriscore_2017():
    # plain oats-like profile -> a; sweet biscuit -> d/e; missing sodium -> None
    assert nutriscore_2017(370, 1, 1.2, 5, 10, 13) == "a"
    assert nutriscore_2017(480, 35, 12, 300, 3, 6) in ("d", "e")
    assert nutriscore_2017(480, 35, 12, None, 3, 6) is None
    assert nutriscore_2017(42, 10, 0, 10, 0, 0, beverage=True) in ("d", "e")


def test_rules_only_nova():
    es = ExpertSystem()
    nova = lambda t: es.rules_only_nova(parse_ingredients(t))
    assert nova("water, aspartame, e150d") == 4
    assert nova("sugar") == 2
    assert nova("tomatoes, salt") == 3
    assert nova("rolled oats") == 1


def test_ocr_noise_is_reproducible_and_scaled():
    t = "sugar, wheat flour, palm oil, emulsifier (soy lecithin), salt" * 3
    assert inject_ocr_noise(t, 0.0, np.random.default_rng(0)) == t
    a = inject_ocr_noise(t, 0.2, np.random.default_rng(1))
    b = inject_ocr_noise(t, 0.2, np.random.default_rng(1))
    assert a == b and a != t


def test_group_metrics():
    df = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 5, "y": [1, 2] * 20 + [1] * 5, "p": [1, 2] * 20 + [2] * 5})
    tbl = group_metrics(df, "g", "y", "p", min_n=10)
    assert list(tbl["g"]) == ["a"] and tbl.loc[0, "accuracy"] == 1.0


def test_eda_helpers(sample_df, tmp_path):
    df = eda.add_text_features(sample_df.head(300))
    assert "n_ingredients" in df and eda.imbalance_ratio(df["nova_group"]) > 1
    assert not eda.top_additives(df["ingredients_norm"]).empty
    p = eda.plot_class_distribution(df["nova_group"].dropna().astype(int), "t_class", "test")
    assert p.exists()


def test_pipeline_gallery_and_agent_template(sample_df):
    g = build_gallery(sample_df, max_rows=200)
    assert len(g) <= 200 and g["category"].notna().all()
    ns = NutriScan.train(sample_df.head(1500))
    r = ns.analyze("Sugar, wheat flour, palm oil, glucose syrup, emulsifier (soy lecithin), colour (E102)")
    assert r["tier_drivers"] is not None and "tier_confidence" in r
    text = agent.template_answer(r)
    assert f"NOVA group {r['nova_final']}" in text and "not medical advice" in text
    ctx = agent.context_from_result(r)
    assert ctx["ingredients"] and "findings" in ctx
