"""Dataset loading, cleaning (Section 5 rules) and grouped splitting (split by brand to avoid leakage)."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .parser import normalized_string


def load_products(path=None, lang: str | None = None) -> pd.DataFrame:
    """Load processed OFF parquet if present, else the synthetic sample CSV.

    lang: keep only this language when a `lang` column exists (e.g. 'en'). The OFF modelling set is only ~48%
    English, so decide the language strategy explicitly and report it.
    """
    if path is None:
        path = config.OFF_PROCESSED if config.OFF_PROCESSED.exists() else config.SAMPLE_CSV
    path = str(path)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype={"code": str})
    if lang and "lang" in df.columns:
        df = df[df["lang"] == lang]
    return clean_products(df)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Rules from the plan: drop missing ingredients, drop missing NOVA, dedupe on barcode then ingredient string."""
    df = df.copy()
    df = df[df["ingredients_text"].notna() & (df["ingredients_text"].str.strip() != "")]
    df["ingredients_norm"] = df["ingredients_text"].map(normalized_string)
    df = df[df["ingredients_norm"] != ""]
    if "code" in df:
        df = df.drop_duplicates("code")
    df = df.drop_duplicates("ingredients_norm")
    if "code" in df and "code_canon" not in df:
        from .external import canonical_barcode
        df["code_canon"] = df["code"].map(canonical_barcode)
    if "nova_group" in df:
        df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce")
        df["brand"] = df.get("brand", pd.Series("unknown", index=df.index)).fillna("unknown")
    if "nutriscore_grade" in df:
        df["health_tier"] = df["nutriscore_grade"].map(config.health_tier)
    return df.reset_index(drop=True)


def supervised_view(df: pd.DataFrame, target: str = "nova_group") -> pd.DataFrame:
    """Rows usable for supervised training of `target` (missing label -> dropped, kept in gallery elsewhere)."""
    col = "health_tier" if target == "health_tier" else target
    out = df[df[col].notna()]
    if col == "health_tier":
        out = out[out[col] != "unknown"]
    return out.reset_index(drop=True)


def grouped_split(df: pd.DataFrame, group_col: str = "brand", test_size: float = 0.25, seed: int = config.RANDOM_STATE):
    """Split by brand so near-identical recipes from one brand never land in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df[group_col]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
