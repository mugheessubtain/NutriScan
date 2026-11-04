# NutriScan: context for Claude Code

AI-powered ingredient label analyzer for **CT-361 AI & Expert Systems (NED University), CCA + CCP**.
Input: label photo or text -> parse ingredients -> predict NOVA (1-4), health tier, product category ->
expert-system rules flag ingredients (cited) -> hybrid decision -> recommend healthier same-category products.
Full plan: `NutriScan Final Project Plan v2` (PDF, one folder up). Data facts: `docs/DATA_FACTS.md`. Report limitations: `docs/LIMITATIONS.md`.

## Commands
```
py -3.14 -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
pytest                                                 # 52 tests; conftest sends all outputs to a temp dir
python training/run_all.py --demo                      # whole pipeline on the synthetic sample -> ./_demo_out
python training/run_all.py [--from N]                  # real data (data/raw/food.parquet + USDA zip)
streamlit run runtime/app/streamlit_app.py
```
Steps: prepare_off -> 1 load -> 2 preprocess -> 2b EDA -> 3 USDA join -> 3b USDA labels -> 6 compare models ->
4 train final -> 5 unseen eval -> 7 ablation/robustness/fairness/SHAP -> 8 alternatives eval.
`NUTRISCAN_OUT=<dir>` redirects every generated file (data/processed, artifacts, reports). Never run smoke tests
without it: real artifacts live in `artifacts/`.

## Architecture (runtime/src/nutriscan/)
- `parser.py`: cleaning + bracket-aware parsing -> `Ingredient(name, position, depth, parent)`.
- `features.py`: dense engineered features from the normalised string (counts, marker flags, sugar/fat/salt position).
- `models/zoo.py`: baselines, 4 improved models, ensembles, tuning spaces, imbalance variants. `models/base.py`:
  shared predictor (train/predict/save/load, `kind`, SHAP background). nova/tier/category predictors subclass it.
- `expert_system.py` + `knowledge_base/`: KB matching, rules (`when.type`: all_categories_present,
  ingredient_present, category_in_first_n, category_count_at_least, nova4_markers_at_least), allergens, preferences,
  `hybrid_nova` (rules only RAISE to 4, with a reason), `rules_only_nova` (ablation).
- `explain.py`: exact linear SHAP (coef * (x - mean)), summaries, TreeExplainer for tree pipelines.
- `eda.py`: summaries + plots E1-E11, confusion/metric bar plots. `usda.py`: category mapping, Nutri-Score 2017.
- `data.py`, `external.py` (USDA<->OFF, barcodes), `evaluation.py` (metrics, group metrics, OCR noise, WER/CER).
- `alternatives.py`: `build_gallery`, `SklearnIndex`, ranking. `ocr.py`: RapidOCR default. `agent.py`: optional chat.
- `pipeline.py`: `NutriScan.analyze(text, prefs)` single entry point; `load_or_train` never writes artifacts.

## Hard rules (do not break)
1. No medical claims. Wording is "associated with" + a source. Keep the "not medical advice" footer.
2. Allergen warnings are **always rule-based**, never LLM-generated.
3. KB entries are written in our own words with `source` + `source_url`; set `verified: true` only after actually reading the source. Do not bulk-copy CSPI or any site. All current entries are `verified: false` drafts.
4. Every user-facing result lists which rules fired and the KB/rules/data versions.
5. Uploaded images are processed in memory and discarded. No personal data stored.
6. Split by brand, never random-by-row, when reporting results.
7. Headline metric is macro-F1 (NOVA 4 dominates).

## Status (v0.3.0, see CHANGELOG.md)
Code for every plan-v2 must-have exists and runs end to end on the synthetic sample (`run_all.py --demo`, 52 tests).
NOT yet done with real data: every number in the report. The v0.2 real-data artifacts were lost; v0.2 tables are in
`reports/member_v0.2/`. OCR needs the VC++ Redistributable on Windows. KB entries are all `verified: false`.

## Known risks (re-check with real data)
- **NOVA leakage is confirmed.** OFF computes NOVA from ingredients; report gains over the keyword-rule baseline and
  lead with the health tier, the USDA tests and robustness.
- **USDA<->OFF overlap ~85%**: train on `off_train_disjoint.parquet` (step3) for any USDA number.
- **NOVA 2 ~1.8%**: per-class F1, `--nova-merge12` rows in cross_dataset tables; step6 drops classes with < 10 rows.
- USDA tier uses Nutri-Score 2017 with fruit/veg = 0; OFF grades are mixed 2017/2023 vintages.
- Windows Smart App Control blocks new compiled wheels: keep pandas < 3 and scikit-learn 1.7 (requirements.txt).

## Next steps (in order)
1. Download data (README section 2), `python training/run_all.py`; check overlap, class counts and timings.
2. Read `reports/eda_findings.md` and `reports/tables/*`; fill the technical report with the real numbers.
3. Verify KB entries against their sources (set `verified: true` only after reading).
4. OCR: install VC++ Redistributable, photograph 30-50 labels into data/ocr_test/, run `training/eval_ocr.py`.

## Conventions
Python 3.11, type hints, small pure functions, tests for every parser/rule change, no notebooks as the only source of logic (notebooks import from `src/`).
