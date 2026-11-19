# Changelog

## 0.4.0: web app, real models, data-cleaning page
- New web interface (`web/`, Next.js 16 + React 19): Scan, Data cleaning, Model performance and Method pages; three.js
  hero whose colour follows the NOVA result; animated verdict card with SHAP bars. Python API in `runtime/api/server.py`
  (FastAPI); `start.bat` starts both. The Streamlit UI is kept as a fallback.
- Trained voting ensembles (254,169 products, tuned, USDA barcodes removed) run in the app; README and report use the
  real tuned parameters.
- Fixed: SHAP crashed for ensembles (members are fit on label-encoded targets); explanations now show readable words
  only (no character n-grams, no bigrams that straddle two ingredients).
- Data cleaning page: every cleaning step records its before/after counts in `reports/tables/cleaning_log.json`
  (`nutriscan/cleaning_log.py`); `run_all.py --until` runs the cleaning steps alone.
- Parser: function-word prefixes removed (`emulsifier: soy lecithin` -> `soy lecithin`, `colour e102` -> `e102`).
- OCR: tries the photo as taken, straightened by the median text-line angle, and preprocessed, and keeps the cleanest
  read; all 8 sample labels in `samples/` read with WER 0.
- Optional Gemini suggestions of healthier products available in Pakistan (`nutriscan/llm_suggest.py`), labelled in
  the UI and never used as a prediction.
- Windows Smart App Control: pyarrow pinned below 23. Next.js upgraded to 16.4.5 (security advisories in 15.x).
- 58 tests.

## 0.3.0: complete plan v2 (CT-361 CCA + CCP, CT-264 core)
Built against "NutriScan Final Project Plan v2". Everything below is tested on the synthetic sample only; real
numbers come from `python training/run_all.py` once `data/raw/` is filled.

**New**
- EDA (plan Section 6): `nutriscan/eda.py`, `steps/step2b_eda.py` (plots E1-E11 for OFF and USDA, summary tables,
  auto-written `reports/eda_findings.md`), notebooks `01_eda_off`, `02_eda_usda`, `03_results`.
- USDA as a labelled dataset (Section 4.1): `nutriscan/usda.py` + `steps/step3b_usda_targets.py`
  (USDA category → OFF food group mapping, Nutri-Score 2017 tier from USDA nutrients, agreement check vs OFF).
- Model zoo (Section 7): `models/zoo.py` with 3 baselines (majority, keyword rule, count-feature LR), 4 improved models
  (TF-IDF LR, calibrated TF-IDF linear SVM, RF on engineered features, gradient boosting on engineered features + LSA),
  soft-voting and stacking ensembles, tuning spaces, imbalance variants (none / class weights / random oversampling /
  SMOTE on dense features). `features.py` holds the engineered features.
- `steps/step6_compare_models.py`: brand-held-out comparison, RandomizedSearchCV with StratifiedGroupKFold by brand,
  imbalance table, USDA cross-dataset tests (OFF labels and independent USDA labels), `--save-best`.
- `steps/step7_ablation_robustness.py`: ML vs rules-only vs hybrid ablation, OCR-noise robustness, fairness per food
  group/language/country, SHAP summaries and per-product examples.
- `steps/step8_eval_alternatives.py`: coverage, same-group share, better share, precision@k, brand diversity.
- SHAP (Section 7.5): exact linear SHAP with a training-mean background (`explain.py`), TreeExplainer for tree models.
- Expert system: `rules_only_nova` (for the ablation), new rule type `category_count_at_least`, KB 27 → 58 entries,
  rules 5 → 12 (all entries still `verified: false`).
- External API (Section 14.1): `agent.py`, optional Claude API chat grounded in the result, template fallback.
- `training/run_all.py` (whole pipeline, `--demo`, `--from`), `training/make_sample_data.py` (synthetic OFF + USDA zip).
- `NUTRISCAN_OUT` / `config.resolve`: generated files can be redirected; tests always use a temp folder.

**Changed / fixed**
- Predictors share `models/base.py`; `kind` selects any zoo model; saved blobs keep `kind` and the SHAP background.
- App: fast start (pre-built `gallery.parquet`, 60k products, instead of re-parsing the whole dataset), no crash
  without data, Data insights and About tabs, tier SHAP, chat panel, online/offline indicator, synthetic-data banner.
- `pipeline.train` no longer writes artifacts unless `save=True` (a demo fallback can't overwrite real models).
- step1 kept no `product_name`, so alternatives had no names: fixed (also keeps countries + nutrients for EDA).
- `prepare_off.py`: reads `data/raw/food.parquet` when present, exports countries and nutrients, no row limit by default.
- `normalize_usda` keeps the latest USDA row *with* ingredients and carries `fdc_id`.
- `eval_ocr.py`: `a.no-preprocess` bug (always crashed) → `a.no_preprocess`.
- OCR: RapidOCR default (no system install), deskew, upscaling, CLAHE, cached readers.
- Alternatives exclude the scanned product itself.
- Requirements pinned for Windows Smart App Control (pandas < 3, scikit-learn 1.7).

**Note:** the v0.2 artifacts (`*.joblib`) were overwritten during testing and are not in this folder; retrain with
step4 (or `run_all.py`). The v0.2 result tables are kept in `reports/member_v0.2/`.

## Unreleased (v0.2.x): cleanup
- Removed evaluation-only scaffolding (`ensemble.py`, `baseline.py`, `rf_features.py`, `features.py`, `embeddings.py`,
  `train_baselines.py`), the Linear SVM branch, `rules_only_nova`, `EmbeddingIndex`/`MilvusIndex`. (0.3.0 re-adds
  baselines, SVM, RF, ensembles and the rules-only ablation in `models/zoo.py`, because the plan's must-have list
  requires them.)

## Unreleased (v0.2.x): 5-step learner pipeline
- Added `training/steps/step1_load_data.py` → `step5_evaluate.py`; `step4` writes `artifacts/train_meta.json`.

## 0.2.0: validation-report update
External test redesigned for the ~85% USDA↔OFF overlap, `prepare_off.py` rewritten for the real schema, leakage guard,
per-class/QWK metrics, RapidOCR + `eval_ocr.py`, USDA parser fixes, ODbL attribution.

## 0.1.0
Initial base repo.
