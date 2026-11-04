"""Section 7.3 engineered features: dense, interpretable numbers computed from the normalised ingredient string only.

Used by the Random Forest / gradient-boosting models, the SMOTE study (SMOTE needs dense features) and the EDA.
Only ingredient TEXT goes in, so the leakage rules in leakage.py hold (no OFF-derived columns, no nutriments).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

E_NUM = re.compile(r"\be\d{3,4}[a-z]?\b")

# marker groups: plain keyword lists, matched as whole words/phrases inside an ingredient name
MARKERS = {
    "syrup": ["syrup", "glucose", "fructose", "dextrose", "maltodextrin", "invert sugar"],
    "flavouring": ["flavouring", "flavourings", "flavoring", "flavorings", "flavour", "flavor", "aroma"],
    "emulsifier": ["emulsifier", "lecithin", "mono- and diglycerides", "e471", "e322", "polysorbate"],
    "thickener": ["modified starch", "modified maize starch", "xanthan", "guar", "carrageenan", "pectin", "e415", "e407"],
    "sweetener": ["aspartame", "acesulfame", "sucralose", "saccharin", "stevia", "steviol", "e950", "e951", "e955"],
    "colour": ["colour", "color", "caramel", "e102", "e110", "e129", "e150", "e171", "carmine", "annatto"],
    "preservative": ["preservative", "benzoate", "sorbate", "nitrite", "nitrate", "sulphite", "sulfite", "e202", "e211", "e250"],
    "enhancer": ["glutamate", "msg", "e621", "yeast extract", "inosinate", "guanylate"],
    "protein_isolate": ["isolate", "hydrolysed", "hydrolyzed", "whey powder", "casein", "gluten"],
    "hydrogenated": ["hydrogenated", "interesterified"],
    "palm": ["palm"],
}
SUGAR_WORDS = ["sugar", "syrup", "dextrose", "glucose", "fructose", "honey", "maltodextrin"]
FAT_WORDS = ["oil", "fat", "butter", "lard", "margarine", "shortening"]

FEATURE_NAMES = (["n_ingredients", "n_e_numbers", "mean_name_len", "has_brackets", "sugar_pos", "fat_pos", "salt_pos",
                  "sugar_in_first3", "n_markers"] + [f"has_{k}" for k in MARKERS])


def _first_pos(items: list[str], words: list[str]) -> int:
    """1-based position of the first ingredient containing one of `words`; 0 when absent."""
    for i, it in enumerate(items, 1):
        if any(w in it for w in words):
            return i
    return 0


def text_features(norm: str) -> list[float]:
    norm = norm or ""
    items = [s.strip() for s in norm.split(",") if s.strip()]
    sugar = _first_pos(items, SUGAR_WORDS)
    flags = [int(any(re.search(rf"(?<![a-z]){re.escape(w)}", norm) for w in words)) for words in MARKERS.values()]
    return [
        len(items),
        len(E_NUM.findall(norm)),
        float(np.mean([len(s) for s in items])) if items else 0.0,
        int("(" in norm),
        sugar,
        _first_pos(items, FAT_WORDS),
        _first_pos(items, ["salt"]),
        int(0 < sugar <= 3),
        sum(flags),
        *flags,
    ]


def feature_frame(texts) -> pd.DataFrame:
    return pd.DataFrame([text_features(t) for t in texts], columns=FEATURE_NAMES)


class TextFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: iterable of normalised ingredient strings -> dense engineered-feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return feature_frame(list(X)).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
