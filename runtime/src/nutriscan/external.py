"""USDA <-> OFF external test design (validation report section 1).

Reported overlap is ~85% of USDA barcodes (not "tiny"), which makes a naive external test CONTAMINATED:
the same physical products sit in OFF training data. So we build three explicit evaluation views:

  (a) HEADLINE external: train on OFF *minus* every USDA-overlapping barcode; test on the overlapping products using
      USDA ingredient text + OFF labels. Product-disjoint from training.
  (b) ROBUSTNESS bonus: train on ALL of OFF; test on the same USDA texts. This is "same product, independent
      transcription source" (manufacturer data vs crowd/OCR), NOT generalisation to new products. Report it as such.
  (c) USDA products absent from OFF have NO NOVA label (USDA has none), so they can only serve as an unlabeled gallery
      for alternatives, not as a scored test set, unless you create proxy labels.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

USDA_COLS = ["fdc_id", "brand_owner", "brand_name", "gtin_upc", "ingredients", "branded_food_category",
             "modified_date", "market_country", "discontinued_date"]


def canonical_barcode(x) -> str | None:
    """Digits only, leading zeros stripped (UPC-A / EAN-13 / GTIN-14 variants of one product collapse together)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    digits = re.sub(r"\D", "", str(x)).lstrip("0")
    return digits or None


def read_usda_branded(path: str | Path, usecols=USDA_COLS) -> pd.DataFrame:
    """Read branded_food.csv from a plain CSV or from the FDC zip download."""
    path = Path(path)
    kwargs = dict(usecols=lambda c: c in set(usecols), dtype=str, low_memory=False)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.endswith("branded_food.csv"))
            with z.open(member) as f:
                return pd.read_csv(f, **kwargs)
    return pd.read_csv(path, **kwargs)


def normalize_usda(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe to one row per canonical barcode (latest modified_date) and rename to NutriScan columns."""
    df = df.copy()
    df["code_canon"] = df["gtin_upc"].map(canonical_barcode)
    df = df[df["code_canon"].notna()]
    df = df[df["ingredients"].notna() & (df["ingredients"].str.strip() != "")]   # keep the latest row WITH ingredients
    if "modified_date" in df:
        df = df.sort_values("modified_date")
    df = df.drop_duplicates("code_canon", keep="last")
    out = pd.DataFrame({
        "fdc_id": df.get("fdc_id"),
        "code_canon": df["code_canon"],
        "code": df["gtin_upc"],
        "brand": df.get("brand_owner", pd.Series(index=df.index, dtype=str)).fillna("unknown").str.lower(),
        "product_name": df.get("brand_name"),
        "ingredients_text": df["ingredients"],
        "usda_category": df.get("branded_food_category"),
        "market_country": df.get("market_country"),
        "discontinued_date": df.get("discontinued_date"),
    })
    out = out[out["ingredients_text"].notna() & (out["ingredients_text"].str.strip() != "")]
    return out.reset_index(drop=True)


def add_code_canon(off: pd.DataFrame) -> pd.DataFrame:
    off = off.copy()
    if "code_canon" not in off:
        off["code_canon"] = off["code"].map(canonical_barcode)
    return off


def match_to_off(usda: pd.DataFrame, off: pd.DataFrame,
                 label_cols=("nova_group", "nutriscore_grade", "health_tier", "category")) -> pd.DataFrame:
    """USDA ingredient TEXT (input) + OFF LABELS (target) for products present in both."""
    off = add_code_canon(off).drop_duplicates("code_canon")
    keep = ["code_canon"] + [c for c in label_cols if c in off.columns]
    merged = usda.merge(off[keep], on="code_canon", how="inner")
    return merged[merged["nova_group"].notna()].reset_index(drop=True) if "nova_group" in merged else merged


def external_eval_sets(off: pd.DataFrame, usda: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the training pools and test/gallery sets described in the module docstring."""
    off = add_code_canon(off)
    matched = match_to_off(usda, off)
    overlap = set(usda["code_canon"]) & set(off["code_canon"].dropna())
    return {
        "train_disjoint": off[~off["code_canon"].isin(overlap)].reset_index(drop=True),   # (a) training pool
        "train_full": off.reset_index(drop=True),                                          # (b) training pool
        "external_matched": matched,                                                       # test for (a) and (b)
        "usda_unlabeled": usda[~usda["code_canon"].isin(overlap)].reset_index(drop=True),  # (c) gallery only
    }


def exclude_overlap(df: pd.DataFrame, usda_codes) -> pd.DataFrame:
    """Drop products whose canonical barcode is in USDA (use for the headline internal+external protocol)."""
    df = add_code_canon(df)
    return df[~df["code_canon"].isin(set(usda_codes))].reset_index(drop=True)
