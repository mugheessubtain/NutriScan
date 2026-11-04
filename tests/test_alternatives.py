import importlib.util
from pathlib import Path

import pandas as pd

from nutriscan.alternatives import AlternativesEngine, off_tag_violations
from nutriscan.expert_system import ExpertSystem


def _gallery():
    return pd.DataFrame({
        "product_name": ["Plain Oats", "Oat Palm Bar", "Sweet Cereal"],
        "brand": ["a", "b", "c"], "category": ["cereal"] * 3,
        "ingredients_text": ["oats", "oats, palm oil", "sugar, oats, flavouring"],
        "ingredients_norm": ["oats", "oats, palm oil", "sugar, oats, flavouring"],
        "nova_group": [1.0, 3.0, 4.0], "health_tier": ["good", "medium", "poor"],
        "analysis_tags": ["en:palm-oil-free,en:vegan", "en:palm-oil", "en:palm-oil-free"],
        "unique_scans_n": [5, 500, 50],
    })


def test_off_tag_violations():
    assert off_tag_violations("en:palm-oil,en:vegan", ["no_palm_oil"]) == ["no_palm_oil"]
    assert off_tag_violations("en:palm-oil-free", ["no_palm_oil"]) == []
    assert off_tag_violations(None, ["vegan"]) == []


def test_recommend_uses_preferences_and_ranks():
    eng = AlternativesEngine(_gallery(), ExpertSystem())
    res = eng.recommend("sugar, oats, flavouring", "cereal", 4, "poor", prefs=["no_palm_oil"])
    names = [a.name for a in res]
    assert "Plain Oats" in names and "Oat Palm Bar" not in names and "Sweet Cereal" not in names


def test_prepare_off_query_builder():
    spec = importlib.util.spec_from_file_location("prepare_off", Path(__file__).resolve().parents[1] / "training" / "prepare_off.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    types = {"code": "VARCHAR", "ingredients_text": "STRUCT(lang VARCHAR, \"text\" VARCHAR)[]", "nova_group": "BIGINT",
             "nutriscore_grade": "VARCHAR", "food_groups_tags": "VARCHAR[]", "lang": "VARCHAR",
             "brands": "VARCHAR", "countries_tags": "VARCHAR[]"}
    q = m.build_query("food.parquet", types, "en", 1000, False)
    assert "x.lang = 'main'" in q and "IN ('a','b','c','d','e')" in q and "food_groups_tags[1]" in q
    assert "lang = 'en'" in q and "LIMIT 1000" in q and "unknown" not in q
