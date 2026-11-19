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
from nutriscan.cleaning_log import record  # noqa: E402
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

    rules = {
        "empty_text": ("Missing or empty ingredient list", "Drop the product",
                       "Without ingredients there is nothing to classify"),
        "unparseable_text": ("Ingredient text with no usable ingredient after cleaning", "Drop the product",
                             "Text such as '-', '...' or only a percentage carries no information"),
        "duplicate_barcode": ("Same barcode after loading", "Keep one row per barcode",
                              "Avoids counting a product twice"),
        "duplicate_ingredient_list": ("Different products with an identical ingredient list (after normalising)",
                                      "Keep one row per normalised list",
                                      "Identical inputs in train and test would inflate the scores"),
    }

    def log(rule_id, before, after):
        issue, fix, why = rules[rule_id]
        record("step2", rule_id, dataset="Open Food Facts", issue=issue, fix=fix, why=why, before=before, after=after)

    df = clean_products(df, log=log)
    print(f"After clean_products (missing/empty text + dedupe): {len(df):,} rows")

    before = len(df)
    df = df[df["ingredients_text"].str.len() >= a.min_chars]
    print(f"Dropped < {a.min_chars} chars: {before - len(df):,}")
    record("step2", "too_short", dataset="Open Food Facts", issue=f"Ingredient text shorter than {a.min_chars} characters",
           fix="Drop the product", why="Fragments like 'salt' typed into the wrong field or OCR debris",
           before=before, after=len(df))

    df = df.copy()
    df["n_ingredients"] = df["ingredients_text"].map(top_level_count)
    df["n_e_numbers"] = df["ingredients_text"].map(e_number_count)

    before = len(df)
    df = df[df["n_ingredients"] > 0]
    print(f"Dropped zero parsed top-level ingredients: {before - len(df):,}")
    record("step2", "no_ingredients_parsed", dataset="Open Food Facts", issue="No ingredient left after parsing",
           fix="Drop the product", why="Nothing for the models to read", before=before, after=len(df))
    for col, label in (("nova_group", "NOVA group"), ("health_tier", "Nutri-Score tier"), ("category", "food group")):
        if col in df.columns:
            missing = int((df[col].isna() | (df[col] == "unknown")).sum())
            record("step2", f"missing_{col}", dataset="Open Food Facts", issue=f"Missing {label} label",
                   fix="Keep the product; leave it out only of the model that needs this label",
                   why="The product is still useful for the other two targets", before=missing, after=missing,
                   unit="labels", kind="kept")

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
