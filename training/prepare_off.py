"""Filter the Open Food Facts Parquet export to a modelling table with DuckDB.

USAGE
  python training/prepare_off.py --describe                       # print the real schema
  python training/prepare_off.py --check                          # verify expected columns/types, no data pulled
  python training/prepare_off.py --src food.parquet --lang en --limit 0 --export-date 2026-10-30

Schema facts from the validation report (111-column export, hf://datasets/openfoodfacts/product-database/food.parquet):
  * ingredients_text / product_name are STRUCT(lang, text)[]; the main-language entry has lang = 'main'.
  * nutriments is STRUCT(name, value, "100g", ...)[]  (only exported with --with-nutriments; NOT a model feature).
  * nutriscore_grade contains junk values ('unknown', 'not-applicable') -> set to NULL here.
  * No pnns_groups_*; category comes from food_groups_tags (first tag).
This script inspects column TYPES with DESCRIBE and builds the SELECT accordingly, and stops with a clear message if a
required column is missing. It has NOT been run against the real 8 GB file by the author of this repo: run --check first.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.cleaning_log import record  # noqa: E402
from nutriscan.external import canonical_barcode  # noqa: E402

HF_SRC = "hf://datasets/openfoodfacts/product-database/food.parquet"
DEFAULT_SRC = str(config.OFF_RAW) if config.OFF_RAW.exists() else HF_SRC   # data/raw/food.parquet when downloaded
REQUIRED = ["code", "ingredients_text", "nova_group", "nutriscore_grade", "food_groups_tags"]
OPTIONAL = ["product_name", "brands", "lang", "categories_tags", "countries_tags", "ingredients_analysis_tags",
            "unique_scans_n", "ingredients_n", "additives_n", "allergens_tags", "schema_version", "nutriments"]
NUTRIMENTS = ["energy-kcal", "sugars", "fat", "saturated-fat", "salt", "proteins", "fiber"]


def text_main(col: str, types: dict) -> str:
    t = types[col]
    if t.startswith("STRUCT"):
        return f"list_filter({col}, lambda x: x.lang = 'main')[1].text"
    return col  # plain VARCHAR


def list_to_str(col: str, types: dict) -> str:
    return f"array_to_string({col}, ',')" if types[col].endswith("[]") else col


def build_query(src: str, types: dict, lang: str | None, limit: int, with_nutriments: bool) -> str:
    sel = ["code", f"{text_main('ingredients_text', types)} AS ingredients_text",
           "CAST(nova_group AS DOUBLE) AS nova_group",
           "CASE WHEN lower(CAST(nutriscore_grade AS VARCHAR)) IN ('a','b','c','d','e') "
           "THEN lower(CAST(nutriscore_grade AS VARCHAR)) END AS nutriscore_grade"]
    fg = types["food_groups_tags"]
    sel.append(("replace(food_groups_tags[1], 'en:', '')" if fg.endswith("[]") else "replace(food_groups_tags, 'en:', '')")
               + " AS category")
    if "product_name" in types:
        sel.append(f"{text_main('product_name', types)} AS product_name")
    if "brands" in types:
        sel.append(f"lower(trim(split_part({list_to_str('brands', types)}, ',', 1))) AS brand")
    for c, alias in (("lang", "lang"), ("unique_scans_n", "unique_scans_n"), ("ingredients_n", "ingredients_n"),
                     ("additives_n", "additives_n"), ("schema_version", "schema_version")):
        if c in types:
            sel.append(f"{c} AS {alias}")
    for c, alias in (("categories_tags", "categories"), ("ingredients_analysis_tags", "analysis_tags"),
                     ("allergens_tags", "allergens")):
        if c in types:
            sel.append(f"{list_to_str(c, types)} AS {alias}")
    if "countries_tags" in types:
        sel.append("list_contains(countries_tags, 'en:united-states') AS is_us")
        sel.append(f"{list_to_str('countries_tags', types)} AS countries")      # EDA E11 + fairness by region
    if with_nutriments and "nutriments" in types:
        for n in NUTRIMENTS:
            sel.append(f"list_filter(nutriments, lambda x: x.name = '{n}')[1].\"100g\" AS {n.replace('-', '_')}_100g")
    where = [f"{text_main('ingredients_text', types)} IS NOT NULL",
             f"trim({text_main('ingredients_text', types)}) <> ''"]
    if lang and "lang" in types:
        where.append(f"lang = '{lang}'")
    q = f"SELECT\n  " + ",\n  ".join(sel) + f"\nFROM read_parquet('{src}')\nWHERE " + "\n  AND ".join(where)
    return q + (f"\nLIMIT {limit}" if limit else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--lang", default="en", help="keep one language ('' = all). Modelling set is ~48%% English")
    ap.add_argument("--export-date", default="unknown", help="date of the OFF export you downloaded (audit log)")
    ap.add_argument("--no-nutriments", dest="with_nutriments", action="store_false",
                    help="skip the *_100g nutrient columns (they are kept by default for EDA; never a model feature)")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    import duckdb
    con = duckdb.connect()
    if a.src.startswith(("hf://", "http")):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{a.src}') LIMIT 1").df()
    types = dict(zip(desc["column_name"], desc["column_type"]))
    if a.describe:
        print(desc.to_string())
        return
    missing = [c for c in REQUIRED if c not in types]
    print("columns:", len(types), "| missing required:", missing or "none",
          "| missing optional:", [c for c in OPTIONAL if c not in types] or "none")
    if missing:
        sys.exit(f"Required columns missing: {missing}. Run --describe and adapt REQUIRED/build_query.")
    if a.check:
        print(build_query(a.src, types, a.lang or None, a.limit, a.with_nutriments))
        return

    con.execute("SET lambda_syntax='ENABLE_SINGLE_ARROW'")
    df = con.execute(build_query(a.src, types, a.lang or None, a.limit, a.with_nutriments)).df()
    df["code_canon"] = df["code"].map(canonical_barcode)
    total = con.execute(f"SELECT count(*) FROM read_parquet('{a.src}')").fetchone()[0]
    record("prepare", "export_filter", dataset="Open Food Facts",
           issue="Full export: 8 GB, 111 columns, all languages, many products without ingredients",
           fix=f"Read only the needed columns and products with a non-empty {a.lang or 'any'}-language ingredient list (DuckDB)",
           why="The full table does not fit in memory and most rows cannot be used", before=total, after=len(df))
    if "nutriscore_grade" in types:
        junk = con.execute(
            f"SELECT count(*) FROM read_parquet('{a.src}') WHERE lower(CAST(nutriscore_grade AS VARCHAR)) "
            "IN ('unknown', 'not-applicable')").fetchone()[0]
        record("prepare", "nutriscore_junk", dataset="Open Food Facts",
               issue="Nutri-Score values 'unknown' and 'not-applicable'", fix="Set to missing",
               why="They are not grades and would become a fake fourth tier", before=junk, after=0,
               unit="values", kind="fixed")
    out = config.OFF_PROCESSED
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "source": a.src, "export_date": a.export_date, "processed_at": datetime.now(timezone.utc).isoformat(),
        "lang_filter": a.lang or None, "rows": int(len(df)),
        "rows_with_nova": int(df["nova_group"].notna().sum()),
        "rows_with_nutriscore": int(df["nutriscore_grade"].notna().sum()),
        "nova_distribution": {str(k): int(v) for k, v in df["nova_group"].value_counts().sort_index().items()},
        "schema_versions": ({str(k): int(v) for k, v in df["schema_version"].value_counts().head(5).items()}
                            if "schema_version" in df else None),
    }
    config.OFF_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()