# To do (updated for v0.3.0)

## Answers to the previous list
- **check UI**: done. Three tabs (Analyze, Data insights, About & ethics), tier SHAP, chat, online/offline status.
  Checked headless with Streamlit's AppTest (no exceptions; metrics, allergens, chat all render).
- **choose one OCR**: RapidOCR is the default (`ocr.DEFAULT_ENGINE`). pip-only, no PATH setup. Tesseract/EasyOCR remain
  optional for the comparison in `training/eval_ocr.py`.
- **tesseract needs PATH variables?**: yes (binary + PATH), which is why it is no longer the default.
- **UI and app taking too long to load**: the app used to load and re-parse the entire OFF table and fit a TF-IDF index
  on it at every start. It now reads `data/processed/gallery.parquet` (60k most-scanned products, pre-normalised,
  written by step4) and the three saved models.
- **cleanup the files**: generated files can be redirected with `NUTRISCAN_OUT`; tests and `run_all.py --demo` never
  write into the project. `.gitignore` added.
- **check and clean md files**: README, CLAUDE.md, CHANGELOG, LIMITATIONS updated. `docs/plan.md` is the old
  5-step plan from the member's tool; keep it as history or delete it (your call).
- **separate runtime src from training files**: runtime code is `runtime/` (package + app + KB); training code is
  `training/` and imports the package. Training needs the same parser/features as runtime, so sharing `src` is
  intentional: a separate copy would drift.
- **what are the kb additives and rules yaml files**: the expert system's knowledge. `additives.yaml` = facts about
  ingredients (aliases, E-numbers, category, rating, reason, source). `rules.yaml` = IF-THEN rules over those facts
  (e.g. IF a colour AND a flavouring AND an emulsifier THEN +2 evidence for NOVA 4), plus allergen keywords and
  user-preference filters. `expert_system.py` reads both; editing YAML changes behaviour without touching code.
- **why are the model files joblib**: `joblib` saves a trained scikit-learn pipeline (TF-IDF vocabulary + model weights)
  to disk so the app can load it in seconds instead of retraining. They are large (50-130 MB) because the TF-IDF
  vocabulary has hundreds of thousands of word and character n-grams.

## Open items
- [ ] Download `food.parquet` + USDA Branded zip into `data/raw/`, run `python training/run_all.py`.
- [ ] Install the Microsoft Visual C++ Redistributable (x64) so RapidOCR loads; photograph 30-50 labels into
      `data/ocr_test/` (photo + typed `.txt`), run `training/eval_ocr.py`.
- [ ] Verify KB entries against their sources; set `verified: true` only after reading each one.
- [ ] Optional: chat groundedness check (30-50 questions), hand-labelled NOVA audit set (100-200 products).
- [ ] Write the technical report with the real numbers (outline: plan v2 Section 18).
