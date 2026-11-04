"""Health tier predictor - classifies products into Nutri-Score tiers (good / medium / poor) from ingredient text.

Nutriments are never features (leakage.py): the label comes from the nutrition table, the input is the ingredient list.
"""
from __future__ import annotations

import pandas as pd

from nutriscan import config
from nutriscan.models.base import TextPredictor

MODEL_PATH = config.ARTIFACTS / "tier_model.joblib"


class TierPredictor(TextPredictor):
    target = "health_tier"
    model_path = MODEL_PATH

    def labelled(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["health_tier"].notna() & (df["health_tier"] != "unknown")]
