"""Step 2 - thorough parse + normalize + filter.

USAGE
  python training/steps/step2_preprocess.py --src data/processed/loaded.parquet --out data/processed/off_clean.parquet --min-chars 8

Reuses `data.clean_products` (normalises ingredient text via
`parser.normalized_string`, dedupes on the normalised string, derives
`health_tier` from Nutri-Score) and `parser.parse_ingredients`. On top of that
it applies two explicit filters you can tune:
  * drop ingredient text shorter than --min-chars
  * drop rows with zero parsed top-level ingredients
and adds `n_ingredients` / `n_e_numbers` for inspection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.data import clean_products  # noqa: E402
from nutriscan.parser import E_NUMBER, parse_ingredients  # noqa: E402




def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"code": str})


def top_level_count(text: str) -> int:
    return sum(1 for ing in parse_ingredients(text) if ing.depth == 0)


def e_number_count(text: str) -> int:
    return len(E_NUMBER.findall(text or ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse, normalise and filter the loaded table.")
    ap.add_argument("--src", default="data/processed/loaded.parquet")
    ap.add_argument("--out", default="data/processed/off_clean.parquet")
    ap.add_argument("--min-chars", type=int, default=8, help="drop ingredient text shorter than this")
    a = ap.parse_args()

    src = config.resolve(a.src)
    if not src.exists():
        sys.exit(f"Input not found: {src} (run step1 first)")

    df = read_table(src)
    rows_in = len(df)
    print(f"Read {rows_in:,} rows from {src}")

    df = clean_products(df)
    print(f"After clean_products (missing/empty text + dedupe): {len(df):,} rows")

    before = len(df)
    df = df[df["ingredients_text"].str.len() >= a.min_chars]
    print(f"Dropped < {a.min_chars} chars: {before - len(df):,}")

    df = df.copy()
    df["n_ingredients"] = df["ingredients_text"].map(top_level_count)
    df["n_e_numbers"] = df["ingredients_text"].map(e_number_count)

    before = len(df)
    df = df[df["n_ingredients"] > 0]
    print(f"Dropped zero parsed top-level ingredients: {before - len(df):,}")

    print("\nnova_group:")
    print(df["nova_group"].value_counts(dropna=False).sort_index().to_string())
    if "health_tier" in df.columns:
        print("\nhealth_tier:")
        print(df["health_tier"].value_counts(dropna=False).to_string())
    if "category" in df.columns:
        print("\nTop categories:")
        print(df["category"].value_counts(dropna=False).head(10).to_string())
    for col in ("nova_group", "nutriscore_grade", "category"):
        if col in df.columns:
            print(f"Unlabelled {col}: {df[col].isna().sum():,}")

    out = config.resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(out, index=False)
    print(f"\nRows in: {rows_in:,} | rows out: {len(df):,}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
