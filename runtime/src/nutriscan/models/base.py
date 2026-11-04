"""Shared train/predict/save/load logic for the three predictors (NOVA, health tier, category).

`kind` picks the estimator from models/zoo.py (default "tfidf_lr", the fast linear model the app was built on;
"voting" is the tuned soft-voting ensemble). Saved blobs also keep `kind` and the mean TF-IDF vector of the training
data, which explain.py uses as the SHAP background for linear models.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .zoo import linear_member, make_model


class TextPredictor:
    target: str = ""
    model_path: Path

    def __init__(self, model=None, kind: str = "tfidf_lr"):
        self.model = model
        self.kind = kind
        self.classes_: list = []
        self.background_ = None          # mean TF-IDF row of training data (linear SHAP background)

    # subclasses override when the label needs cleaning (e.g. NOVA -> int)
    def labelled(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.target].notna()]

    def labels(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target]

    def train(self, df: pd.DataFrame, model=None) -> "TextPredictor":
        """Train on a dataframe with 'ingredients_norm' and the target column. `model` overrides `kind` (e.g. a tuned estimator)."""
        df = self.labelled(df)
        self.model = (model if model is not None else make_model(self.kind)).fit(df["ingredients_norm"].tolist(), self.labels(df))
        self.classes_ = [c.item() if hasattr(c, "item") else c for c in self.model.classes_]
        lin = linear_member(self.model)
        if lin is not None:
            X = lin.named_steps["tfidf"].transform(df["ingredients_norm"].tolist())
            self.background_ = np.asarray(X.mean(axis=0)).ravel()
        return self

    def _x(self, ingredients_norm):
        return [ingredients_norm] if isinstance(ingredients_norm, str) else list(ingredients_norm)

    def predict(self, ingredients_norm: str | list[str]) -> np.ndarray:
        return self.model.predict(self._x(ingredients_norm))

    def predict_proba(self, ingredients_norm: str | list[str]) -> np.ndarray:
        return self.model.predict_proba(self._x(ingredients_norm))

    def save(self, path=None):
        path = Path(path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_, "kind": self.kind, "background": self.background_}, path)

    @classmethod
    def load(cls, path=None) -> "TextPredictor":
        blob = joblib.load(path or cls.model_path)
        p = cls(blob["model"], blob.get("kind", "tfidf_lr"))
        p.classes_ = blob["classes"]
        p.background_ = blob.get("background")
        return p
