"""Step 1 - load, select and top-level dedupe the modelling table.

USAGE
  python training/steps/step1_load_data.py --src data/processed/off_en.parquet --lang en --out data/processed/loaded.parquet
  python training/steps/step1_load_data.py --src data/sample/products_sample.csv

Reads a parquet table (or the sample CSV), keeps the modelling columns that
exist, then removes exact duplicate rows and duplicate barcodes. This is the
file to edit when you want to change which columns or dedup rules go into the
pipeline. Cleaning/normalisation happens in step2, not here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402

MODELLING_COLUMNS = [
    "code", "product_name", "ingredients_text", "nova_group", "nutriscore_grade", "category",
    "brand", "lang", "countries", "is_us", "analysis_tags", "allergens", "unique_scans_n",
    # nutrients are kept for EDA and alternatives ranking only; leakage.py forbids them as model features
    "energy_kcal_100g", "sugars_100g", "fat_100g", "saturated_fat_100g", "salt_100g", "proteins_100g", "fiber_100g",
]




def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"code": str})


def main() -> None:
    ap = argparse.ArgumentParser(description="Load, select and dedupe the modelling table.")
    ap.add_argument("--src", default="data/processed/off_en.parquet",
                    help="parquet or CSV input (relative to Nutriscan_Final/)")
    ap.add_argument("--lang", default=None, help="keep one language when a 'lang' column exists")
    ap.add_argument("--out", default="data/processed/loaded.parquet")
    a = ap.parse_args()

    src = config.resolve(a.src)
    if not src.exists():
        sys.exit(f"Input not found: {src}")

    df = read_table(src)
    rows_in = len(df)
    print(f"Read {rows_in:,} rows from {src}")

    if a.lang and "lang" in df.columns:
        df = df[df["lang"] == a.lang]
        print(f"lang={a.lang!r}: kept {len(df):,} rows")
    elif a.lang:
        print(f"lang={a.lang!r} requested but there is no 'lang' column; skipped")

    missing = [c for c in MODELLING_COLUMNS if c not in df.columns]
    keep = [c for c in MODELLING_COLUMNS if c in df.columns]
    df = df[keep].copy()
    print(f"Missing modelling columns: {missing or 'none'}")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Exact-row duplicates dropped: {before - len(df):,}")

    if "code" in df.columns:
        before = len(df)
        df = df.drop_duplicates("code")
        print(f"Duplicate barcodes dropped: {before - len(df):,}")

    out = config.resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(out, index=False)
    print(f"Rows in: {rows_in:,} | rows out: {len(df):,}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
