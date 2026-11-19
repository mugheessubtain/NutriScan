"""End-to-end NutriScan pipeline: text -> parse -> ML (NOVA, tier, category) -> expert rules -> hybrid -> alternatives."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import __version__, config
from .alternatives import AlternativesEngine, build_gallery
from .data import load_products
from .expert_system import ExpertSystem
from .explain import linear_shap
from .models import CategoryPredictor, NovaPredictor, TierPredictor
from .parser import extract_ingredient_section, normalized_string, parse_ingredients

TRAIN_META = config.ARTIFACTS / "train_meta.json"


class NutriScan:
    def __init__(self, nova_pred: NovaPredictor, tier_pred: TierPredictor, category_pred: CategoryPredictor,
                 gallery: pd.DataFrame, data_source: str = "unknown", gallery_synthetic: bool = False):
        self.gallery_synthetic = gallery_synthetic      # True when the gallery is the synthetic sample (UI hides it)
        self.nova_pred = nova_pred
        self.tier_pred = tier_pred
        self.category_pred = category_pred
        self.gallery = gallery
        self.es = ExpertSystem()
        self.alt = AlternativesEngine(gallery, self.es) if len(gallery) else None
        self.data_source = data_source
        self.data_meta = json.loads(config.OFF_META.read_text(encoding="utf-8")) if config.OFF_META.exists() else {}
        self.train_meta = json.loads(TRAIN_META.read_text(encoding="utf-8")) if TRAIN_META.exists() else {}
        # Backward compat: models dict interface
        self.models = {"nova": nova_pred.model, "tier": tier_pred.model, "category": category_pred.model}

    # ---------- training ----------
    @classmethod
    def train(cls, df: pd.DataFrame | None = None, data_source: str = "sample", save: bool = False) -> "NutriScan":
        """Train all three predictors on `df` (default: the synthetic sample). Nothing is written unless save=True,
        so a demo fallback can never overwrite real artifacts."""
        df = load_products(config.SAMPLE_CSV) if df is None else df
        preds = [NovaPredictor().train(df), TierPredictor().train(df), CategoryPredictor().train(df)]
        if save:
            for p in preds:
                p.save()
        return cls(*preds, build_gallery(df), data_source)

    @classmethod
    def load_or_train(cls) -> "NutriScan":
        """Saved artifacts + pre-built gallery when available (fast start); otherwise an in-memory demo on the sample."""
        paths = [NovaPredictor.model_path, TierPredictor.model_path, CategoryPredictor.model_path]
        if all(p.exists() for p in paths):
            meta = json.loads(TRAIN_META.read_text(encoding="utf-8")) if TRAIN_META.exists() else {}
            source = ("synthetic sample" if meta.get("synthetic", "sample" in str(meta.get("data_path", "")))
                      else ("off" if meta else "unknown"))
            synthetic_gallery = not config.GALLERY.exists()
            if config.GALLERY.exists():
                gallery = pd.read_parquet(config.GALLERY)
            elif config.SAMPLE_CSV.exists():
                gallery = build_gallery(load_products(config.SAMPLE_CSV))
            else:
                gallery = pd.DataFrame()
            return cls(NovaPredictor.load(), TierPredictor.load(), CategoryPredictor.load(), gallery, source,
                       gallery_synthetic=synthetic_gallery)
        return cls.train(data_source="sample (in-memory demo, no artifacts found)")

    # ---------- inference ----------
    def analyze(self, raw_text: str, prefs: list[str] | None = None, k_alt: int = 5) -> dict:
        text = extract_ingredient_section(raw_text) if "ingredient" in raw_text.lower() else raw_text
        ings = parse_ingredients(text)
        norm = normalized_string(text)
        if not norm:
            return {"error": "No ingredients could be parsed."}

        nova_proba = self.nova_pred.predict_proba(norm)[0]
        nova_pred = int(self.nova_pred.model.classes_[int(np.argmax(nova_proba))])
        tier_proba = self.tier_pred.predict_proba(norm)[0]
        tier_pred = str(self.tier_pred.model.classes_[int(np.argmax(tier_proba))])
        cat_pred = str(self.category_pred.predict(norm)[0])

        result = self.es.analyze(ings)
        nova_final, hybrid_reasons = self.es.hybrid_nova(nova_pred, result)
        names = [i.name for i in ings]
        # word bigrams can straddle two ingredients ("flour palm"); only show terms that occur inside one ingredient
        readable = lambda terms: [(t, v) for t, v in terms if any(t in n for n in names)][:8]
        drivers = readable(linear_shap(self.nova_pred.model, norm, nova_pred, self.nova_pred.background_, k=20,
                                       words_only=True))
        tier_drivers = readable(linear_shap(self.tier_pred.model, norm, tier_pred, self.tier_pred.background_, k=20,
                                            words_only=True))
        alts = self.alt.recommend(norm, cat_pred, nova_final, tier_pred, prefs, k=k_alt) if self.alt else []

        return {
            "ingredients": [i.name for i in ings],
            "ingredients_norm": norm,
            "nova_ml": nova_pred,
            "nova_confidence": float(np.max(nova_proba)),
            "nova_final": nova_final,
            "hybrid_reasons": hybrid_reasons,
            "health_tier": tier_pred,
            "tier_confidence": float(np.max(tier_proba)),
            "category": cat_pred,
            "findings": result.findings,
            "allergens": result.allergens,
            "drivers": drivers,
            "tier_drivers": tier_drivers,
            "alternatives": alts,
            "versions": {"app": __version__, "kb": self.es.kb_version, "rules": self.es.rules_version,
                         "model": self.nova_pred.kind, "data_source": self.data_source,
                         "trained_at": self.train_meta.get("trained_at"),
                         "off_export_date": self.data_meta.get("export_date"),
                         "off_schema_versions": self.data_meta.get("schema_versions")},
        }
