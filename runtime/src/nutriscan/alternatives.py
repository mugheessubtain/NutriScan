"""Section 8 alternatives engine.

VectorIndex is an interface so you can swap SklearnIndex for another VectorIndex implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import config
from .expert_system import ExpertSystem
from .parser import parse_ingredients


class VectorIndex:
    def fit(self, texts: list[str]): raise NotImplementedError
    def query(self, text: str, k: int) -> tuple[np.ndarray, np.ndarray]: raise NotImplementedError  # (indices, similarities)


class SklearnIndex(VectorIndex):
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.nn = NearestNeighbors(metric="cosine")

    def fit(self, texts):
        X = self.vec.fit_transform(texts)
        self.nn.fit(X)
        self.n = X.shape[0]
        return self

    def query(self, text, k):
        k = min(k, self.n)
        dist, idx = self.nn.kneighbors(self.vec.transform([text]), n_neighbors=k)
        return idx[0], 1.0 - dist[0]


# Dense-embedding and Milvus index implementations were removed; the runtime uses SklearnIndex.

GALLERY_COLS = ["code", "product_name", "brand", "category", "nova_group", "health_tier", "ingredients_text",
                "ingredients_norm", "analysis_tags", "unique_scans_n"]


def build_gallery(df: pd.DataFrame, max_rows: int = config.GALLERY_MAX_ROWS) -> pd.DataFrame:
    """Products the app may recommend: labelled (category + NOVA or tier), named, most-scanned first.

    Kept small and pre-normalised so the app does not re-parse hundreds of thousands of rows at start-up."""
    g = df[df["ingredients_norm"].fillna("").ne("") & df["category"].notna()].copy()
    g = g[g["nova_group"].notna() | g.get("health_tier", pd.Series("unknown", index=g.index)).isin(list(config.TIER_ORDER))]
    if "product_name" in g:
        g = g[g["product_name"].fillna("").str.strip().ne("")]
    else:
        g["product_name"] = "unnamed product"
    if "unique_scans_n" in g:
        g = g.sort_values("unique_scans_n", ascending=False, na_position="last")
    for c in GALLERY_COLS:
        if c not in g:
            g[c] = np.nan
    g["brand"] = g["brand"].fillna("unknown")
    return g[GALLERY_COLS].head(max_rows).reset_index(drop=True)


# OFF-derived tags (ingredients_analysis_tags) are good FILTERS for preferences but must never be NOVA features.
OFF_TAG_RULES = {"no_palm_oil": "en:palm-oil", "vegan": "en:non-vegan", "vegetarian": "en:non-vegetarian"}  # verify tag names


def off_tag_violations(tags: str | None, prefs: list[str]) -> list[str]:
    if not isinstance(tags, str) or not tags:
        return []
    have = set(tags.split(","))
    return [p for p in prefs if OFF_TAG_RULES.get(p) in have]


@dataclass
class Alternative:
    name: str
    brand: str
    category: str
    similarity: float
    nova_group: float
    health_tier: str
    reason: str


class AlternativesEngine:
    def __init__(self, gallery: pd.DataFrame, es: ExpertSystem, index: VectorIndex | None = None):
        self.gallery = gallery.reset_index(drop=True)
        self.es = es
        self.index = (index or SklearnIndex()).fit(self.gallery["ingredients_norm"].tolist())
        self._pop_max = (np.log1p(self.gallery["unique_scans_n"].max())
                         if "unique_scans_n" in self.gallery and self.gallery["unique_scans_n"].notna().any() else 0.0)

    def recommend(self, ingredients_norm: str, category: str, nova: float, tier: str,
                  prefs: list[str] | None = None, k: int = 5, pool: int = 50,
                  popularity_weight: float = 0.05) -> list[Alternative]:
        """Rank = similarity + 0.3*improvement + popularity_weight*normalised log(unique_scans_n) when available."""
        prefs = prefs or []
        idx, sims = self.index.query(ingredients_norm, pool)
        out: list[Alternative] = []
        for i, s in zip(idx, sims):
            row = self.gallery.iloc[i]
            if row["category"] != category or row["ingredients_norm"] == ingredients_norm:
                continue                               # other food group, or the scanned product itself
            better_nova = pd.notna(row.get("nova_group")) and pd.notna(nova) and row["nova_group"] < nova
            better_tier = (config.TIER_ORDER.get(row.get("health_tier"), -1) > config.TIER_ORDER.get(tier, -1))
            if not (better_nova or better_tier):
                continue
            if prefs and (self.es.violates_preference(parse_ingredients(row["ingredients_text"]), prefs)
                          or off_tag_violations(row.get("analysis_tags"), prefs)):
                continue
            why = []
            if better_nova:
                why.append(f"less processed (NOVA {int(row['nova_group'])} vs {int(nova)})")
            if better_tier:
                why.append(f"better health tier ({row['health_tier']} vs {tier})")
            improvement = float(better_nova) + float(better_tier)
            pop = 0.0
            if self._pop_max and pd.notna(row.get("unique_scans_n")):
                pop = np.log1p(max(float(row["unique_scans_n"]), 0)) / self._pop_max
            out.append((s + 0.3 * improvement + popularity_weight * pop, Alternative(row["product_name"], row["brand"], row["category"], float(s),
                        float(row["nova_group"]) if pd.notna(row["nova_group"]) else float("nan"),
                        str(row.get("health_tier", "unknown")), "; ".join(why))))
        out.sort(key=lambda t: -t[0])
        return [a for _, a in out[:k]]
