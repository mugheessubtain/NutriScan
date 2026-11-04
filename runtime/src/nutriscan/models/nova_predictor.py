"""NOVA predictor - classifies products into the 4 NOVA groups (1-4) from ingredient text."""
from __future__ import annotations

import pandas as pd

from nutriscan import config
from nutriscan.models.base import TextPredictor

MODEL_PATH = config.ARTIFACTS / "nova_model.joblib"


class NovaPredictor(TextPredictor):
    target = "nova_group"
    model_path = MODEL_PATH

    def labels(self, df: pd.DataFrame) -> pd.Series:
        return df["nova_group"].astype(int)   # real OFF data stores NOVA as float
