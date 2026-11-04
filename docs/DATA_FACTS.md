# Data facts (from NUTRISCAN_VALIDATION_REPORT.md, measured 2026-10-30/31 by its author)
These numbers were reported by the independent validation, NOT re-measured in this repo. Re-check them in your own EDA
notebook before quoting them in the report, and record the export date in the audit log.

## Open Food Facts `food.parquet` (hf://datasets/openfoodfacts/product-database)
- 4,884,051 food rows, 111 columns, ~8.0 GB. Modelling set (ingredients AND NOVA): 1,125,878 rows.
- Modelling-set NOVA: 1 = 11.9%, **2 = 1.8%**, 3 = 19.3%, **4 = 67.0%**. Use macro-F1, per-class F1, and the 3-class variant.
- Language of the modelling set: en 47.9%, fr 27.8%, rest mostly es/de/it/nl (539k English). Decide the language strategy explicitly.
- Nutri-Score: real letters for ~29% of products; the export also contains junk values `unknown` / `not-applicable`
  (set to NULL by `prepare_off.py`). OFF migrated to the 2023 algorithm with no per-product version flag, so labels are mixed-vintage.
- Categories: no `pnns_groups_*` columns; use `food_groups_tags` (first tag), seeded categories: sugary-snacks, fish-meat-eggs,
  cereals-and-potatoes, milk-and-dairy, beverages, fats-and-sauces, fruits-and-vegetables, composite-foods, salty-snacks, alcoholic-beverages.
- Handy columns: `ingredients_analysis_tags` (palm oil / vegan / vegetarian), `unique_scans_n` (popularity), `images`
  (OCR test set), `schema_version`, `allergens_tags`, `traces_tags`. OFF-derived ingredient columns are FILTERS only, never NOVA features.

## USDA FoodData Central Branded (Oct-2026 CSV)
- 2,010,212 rows collapse to ~443k distinct barcodes (multiple `fdc_id` per product; keep latest `modified_date`).
- Reported overlap with OFF: ~85% of USDA barcodes (363,767 / 421,666 well-formed GTIN-13 = 86.3%). ~313k overlapping products have an OFF NOVA label.
- Ingredient strings are ALL-CAPS, `INGREDIENTS:` prefix, `CONTAINS LESS THAN 2% OF:`, nested brackets, `**` footnotes. The parser handles these.
- If you filter OFF to English only, the overlap will be lower than the full-export figure.

## Consequence for evaluation
85% overlap means a naive "external" set is not product-disjoint from OFF training data. See `runtime/src/nutriscan/external.py` and
`training/eval_external.py`: (a) headline = train without USDA-overlap barcodes, (b) bonus = same product, independent transcription.
USDA products absent from OFF have no NOVA label, so they are a gallery for alternatives, not a scored test set.
