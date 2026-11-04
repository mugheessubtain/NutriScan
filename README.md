# NutriScan
AI-powered ingredient label analyzer with healthier-alternative recommendations (CT-361 CCA + CCP; also mapped to the
CT-264 CCA, see plan v2 Section 2.1).

Paste or photograph an ingredient list and get: NOVA processing group, Nutri-Score health tier, food group, flagged
ingredients with cited reasons, rule-based allergen warnings, SHAP explanations and ranked healthier alternatives in the
same food group. Hybrid system: ML classifiers + a YAML knowledge-base expert system (58 entries, 12 rules).

## 1. Setup (Windows)
```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt          # or: pip install -r requirements.lock.txt for the exact tested versions
pytest                                   # 52 tests, all outputs go to a temp folder
```
RapidOCR (photo upload) also needs the **Microsoft Visual C++ Redistributable (x64)**. Everything else works without it.

## 2. Data (two datasets)
| File | Where to get it | Put it in |
|---|---|---|
| `food.parquet` (~8 GB) | huggingface.co/datasets/openfoodfacts/product-database → Files | `data/raw/food.parquet` |
| Branded Foods **CSV** zip | fdc.nal.usda.gov/download-datasets → Branded Foods | `data/raw/` (keep it zipped) |

## 3. Run everything
```bash
python training/run_all.py                 # real data: prepare_off -> steps 1-8 (resume with --from N)
python training/run_all.py --demo          # synthetic sample, writes only to ./_demo_out (smoke test)
streamlit run runtime/app/streamlit_app.py
```

| Step | Script | Output |
|---|---|---|
| 0 | `training/prepare_off.py` | `off_en.parquet` (+ audit `off_en.meta.json`) |
| 1 | `steps/step1_load_data.py` | select columns, drop duplicate rows/barcodes |
| 2 | `steps/step2_preprocess.py` | parse + normalise ingredients → `off_clean.parquet` |
| 2b | `steps/step2b_eda.py` | EDA plots E1-E11, tables, `reports/eda_findings.md` |
| 3 | `steps/step3_make_unseen_set.py` | USDA ↔ OFF join, unseen test set, overlap-safe training pool |
| 3b | `steps/step3b_usda_targets.py` | USDA's own labels: mapped food group + Nutri-Score tier from USDA nutrients |
| 6 | `steps/step6_compare_models.py` | baselines vs 4 models vs tuned vs voting/stacking, imbalance table, USDA cross-dataset |
| 4 | `steps/step4_train_models.py` | final models → `artifacts/`, app gallery |
| 5 | `steps/step5_evaluate.py` | final models on the unseen USDA set |
| 7 | `steps/step7_ablation_robustness.py` | ML vs rules vs hybrid, OCR-noise robustness, fairness, SHAP |
| 8 | `steps/step8_eval_alternatives.py` | coverage, precision@5, diversity of recommendations |

Notebooks `notebooks/01_eda_off`, `02_eda_usda`, `03_results` show the same results step by step (they import the
package; no logic lives only in a notebook).

## 4. Layout
```
runtime/knowledge_base/   additives.yaml (58 entries), rules.yaml (12 rules, allergens, preferences)
runtime/src/nutriscan/    parser, features, models/ (zoo + 3 predictors), expert_system, explain (SHAP),
                          alternatives, eda, usda, external, evaluation, ocr, agent (optional Claude API), pipeline
runtime/app/              streamlit_app.py
training/                 prepare_off.py, run_all.py, make_sample_data.py, eval_ocr.py, eval_external.py, steps/
notebooks/                EDA + results notebooks
reports/                  tables/, figures/ (generated); member_v0.2/ (first real-data results, kept for comparison)
docs/                     DATA_FACTS.md, LIMITATIONS.md
```
Set `NUTRISCAN_OUT=<folder>` to send every generated file somewhere else (tests and `--demo` do this automatically).

## 5. Optional chat (external API)
Set `ANTHROPIC_API_KEY` to let the chat panel answer follow-up questions with the Claude API, grounded only in the
analysis result. Without a key (or offline) it returns a template explanation. The API never classifies products or
produces allergen warnings.

## Disclaimer
Uses established classifications (NOVA, Nutri-Score) and published ratings. Not medical advice.
Contains data from Open Food Facts, available under the Open Database License (https://world.openfoodfacts.org).
USDA FoodData Central is public domain (https://fdc.nal.usda.gov).
