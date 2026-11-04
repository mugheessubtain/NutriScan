import math

import pandas as pd

from nutriscan.evaluation import (char_error_rate, confusion_df, ingredient_extraction_accuracy, merge_nova12,
                                  ordinal_metrics, per_class_report, word_error_rate)
from nutriscan.leakage import check_features, forbidden_features


def test_wer_cer():
    assert word_error_rate("sugar salt oil", "sugar salt oil") == 0
    assert math.isclose(word_error_rate("sugar salt oil", "sugar sal oil"), 1 / 3)
    assert char_error_rate("palm oil", "pa1m oil") > 0


def test_ingredient_extraction_f1():
    assert ingredient_extraction_accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert ingredient_extraction_accuracy(["a", "b"], ["c"]) == 0.0


def test_ordinal_and_merge():
    m = ordinal_metrics([1, 2, 3, 4, 4], [1, 2, 3, 4, 3])
    assert m["qwk"] > 0.8 and m["spearman"] > 0.8
    assert list(merge_nova12([1, 2, 3, 4])) == [1, 1, 3, 4]


def test_per_class_and_confusion():
    rep = per_class_report([1, 3, 4, 4], [1, 4, 4, 4])
    assert set(rep["class"]) == {1, 3, 4} and rep.loc[rep["class"] == 3, "recall"].iloc[0] == 0
    assert confusion_df([1, 3, 4, 4], [1, 4, 4, 4]).loc["true_4", "pred_4"] == 2


def test_leakage_guard():
    assert forbidden_features(["n_ingredients", "additives_n"], "nova_group") == ["additives_n"]
    assert forbidden_features(["sugars_100g", "n_ingredients"], "health_tier") == ["sugars_100g"]
    check_features(["n_ingredients", "n_e_numbers"], "nova_group")      # engineered features are clean
    try:
        check_features(["ingredients_analysis_tags"], "nova_group")
        assert False
    except ValueError:
        pass
