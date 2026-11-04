"""Category predictor - classifies products into coarse food groups (OFF food_groups level 1) from ingredient text.

The category defines "same type of product" for the alternatives engine.
"""
from __future__ import annotations

import pandas as pd

from nutriscan import config
from nutriscan.models.base import TextPredictor

MODEL_PATH = config.ARTIFACTS / "category_model.joblib"


class CategoryPredictor(TextPredictor):
    target = "category"
    model_path = MODEL_PATH

    def labels(self, df: pd.DataFrame) -> pd.Series:
        return df["category"].astype(str)
