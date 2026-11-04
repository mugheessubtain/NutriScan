  Current Plan
     /home/shaheer/.openclaude/plans/memoized-prancing-dijkstra.md

     Plan: Turn NutriScan into a 5-step, learner-editable pipeline

     Context

     You want a small, transparent pipeline you can personally edit while
     learning: a
      data folder, one file that loads/selects/dedupes, one thorough
     preprocessing
     file, three one-file-per-model predictors, one file that evaluates saved
     models
     on unseen USDA data, and a UI file with OCR. You also asked whether the
     current
     structure needs changing.

     Assessment: it mostly does not need rewriting. Four of the five pieces
     already exist:
     - loading/cleaning/dedup: runtime/src/nutriscan/data.py — load_products
     (:11), clean_products (:26; dedupes code :33 then ingredients_norm :34)
     - preprocessing: runtime/src/nutriscan/parser.py — normalize_text (:86:
     lowercase, %, E-numbers, brackets), parse_ingredients (:137),
     normalized_string (:153)
     - three model files:
     runtime/src/nutriscan/models/{nova_predictor,tier_predicto
     r,category_predictor}.py (each TF-IDF + LogisticRegression
     class_weight="balanced", .train/.predict/.save/.load, saving
     artifacts/{nova,tier,category}_model.joblib)
     - UI + OCR: runtime/app/streamlit_app.py, runtime/src/nutriscan/ocr.py
     (read_label)
     - unseen-set helpers: runtime/src/nutriscan/external.py (canonical_barcode
     :25, read_usda_branded :33, normalize_usda :45, match_to_off :74,
     external_eval_sets :83, exclude_overlap :96); metrics in evaluation.py
     (summary :15, ordinal_metrics :23, per_class_report, confusion_df,
     merge_nova12)

     What is missing: a clear, ordered set of thin "step" scripts that call this     logic, and an evaluator that scores the saved artifacts (existing
     eval_external.py retrains a fresh model instead). Decisions locked with
     you:
     classification base models, thin wrappers (do not rewrite the package),
     parse+normalize+filter preprocessing, USDA branded CSV zip.

     Target structure (additions only)

     Nutriscan_Final/
     ├── data/
     │   ├── raw/                     # YOU put food.parquet +
     FoodData_Central_branded_food_csv_<date>.zip here
     │   └── processed/               # off_en.parquet (existing),
     loaded.parquet,
     off_clean.parquet,
     │                                # usda_branded.parquet,
     usda_matched.parquet,
     off_train_disjoint.parquet
     ├── training/
     │   ├── prepare_off.py           # existing: 8 GB raw -> off_en.parquet
     (run
     once; already done)
     │   └── steps/                   # NEW: the five files you will edit
     │       ├── step1_load_data.py
     │       ├── step2_preprocess.py
     │       ├── step3_make_unseen_set.py
     │       ├── step4_train_models.py
     │       └── step5_evaluate.py
     ├── runtime/                     # unchanged: package, app, knowledge_base
     ├── artifacts/                   # {nova,tier,category}_model.joblib +
     train_meta.json
     └── reports/tables/              # unseen_metrics.csv
     All step files start with the same sys.path.insert(0, .../runtime/src) line     used by existing training scripts.

     The five step files

     step1_load_data.py — load + select + top-level dedup

     - CLI: python training/steps/step1_load_data.py --src
     data/processed/off_en.parquet --lang en --out data/processed/loaded.parquet     - Reads parquet (or sample CSV with --src data/sample/products_sample.csv);     optional --lang filter.
     - Selects modelling columns: code, ingredients_text, nova_group,
     nutriscore_grade, category, brand, lang, analysis_tags, allergens,
     unique_scans_n (keep those present; report missing).
     - Top-level dedup: exact-row duplicates, then duplicate code (explicit,
     visible lines — this is what you'll tweak).
     - Prints: rows in/out, each dedup count, missing columns. Writes
     loaded.parquet.

     step2_preprocess.py — thorough parse + normalize + filter

     - CLI: python training/steps/step2_preprocess.py --src
     data/processed/loaded.parquet --out data/processed/off_clean.parquet
     --min-chars  8
     - Reuses data.clean_products (normalizes via parser.normalized_string,
     dedupes on ingredients_norm, derives health_tier) and
     parser.parse_ingredients.
     - Extra explicit filters: drop ingredients_text shorter than --min-chars;
     drop rows with zero parsed top-level ingredients; add n_ingredients,
     n_e_numbers for inspection.
     - Prints: rows dropped per rule; value counts for nova_group, health_tier,
     and top categorys; unlabeled counts. Writes off_clean.parquet.

     step3_make_unseen_set.py — build unseen test set + overlap-safe training

     pool
     - CLI: python training/steps/step3_make_unseen_set.py --usda-zip
     data/raw/FoodData_Central_branded_food_csv_<date>.zip --off
     data/processed/off_clean.parquet
     - Reuses external.read_usda_branded → normalize_usda, then
     external_eval_sets(off_df, usda_df) returning train_disjoint,
     external_matched, usda_unlabeled; ensure category is carried as a label
     via match_to_off(..., label_cols=("category",)) if needed.
     - Writes: usda_branded.parquet, usda_matched.parquet (label-bearing unseen
     set), off_train_disjoint.parquet (OFF minus USDA barcodes).
     - Prints: USDA barcodes with ingredients, overlap as % of USDA barcodes
     (not
      % of OFF), matched rows with non-null nova_group/health_tier/category,
     and train_disjoint row count.
     - Consolidates what the old prepare_usda.py did, adds the persisted disjoint pool
     that it only printed. (prepare_usda.py was replaced by step3 and removed.)

     step4_train_models.py — the runner for the three predictors

     - CLI: python training/steps/step4_train_models.py --data
     data/processed/off_train_disjoint.parquet [--only nova|tier|category]
     - Default input: off_train_disjoint.parquet if present, else
     off_clean.parquet with a loud "NOT overlap-safe" warning.
     - Reuses NovaPredictor/TierPredictor/CategoryPredictor .train(df).save()
     (each file already filters unlabeled rows: nova :23, tier :23-24, category
     :24-25).
     - Writes artifacts/{nova,tier,category}_model.joblib +
     artifacts/train_meta.json (data_path, overlap_safe, rows, trained_at,
     classes).
     - Prints: data path, overlap-safe status, rows/classes per target, artifact     paths.

     step5_evaluate.py — score the saved artifacts on unseen data

     - CLI: python training/steps/step5_evaluate.py --matched
     data/processed/usda_matched.parquet [--nova-merge12]
     - Loads via .load(); runs data.clean_products on the matched set to add
     ingredients_norm/health_tier; filters labeled rows per target (categories:
     drop NaN).
     - Metrics: NOVA → summary + ordinal_metrics (accuracy, macro-F1, QWK,
     Spearman); tier/category → summary + per_class_report; optional
     confusion_df, merge_nova12.
     - Reads artifacts/train_meta.json and warns if overlap_safe is false.
     Writes reports/tables/unseen_metrics.csv (+ per-class).
     - If matched set is missing, exits with the exact step3 command.

     Workflow (order is load-bearing)

     1. Put food.parquet + USDA branded zip in data/raw/. Run prepare_off.py
     once → off_en.parquet (already done).
     2. step1_load_data.py → loaded.parquet
     3. step2_preprocess.py → off_clean.parquet
     4. step3_make_unseen_set.py → usda_matched.parquet +
     off_train_disjoint.parquet
     5. step4_train_models.py --data .../off_train_disjoint.parquet → artifacts
     (overlap_safe: true)
     6. step5_evaluate.py → metrics; then streamlit run
     runtime/app/streamlit_app.py

     Why step 4 must use off_train_disjoint: disjointness in external.py is
     barcode-only (:87-92). Training artifacts on full OFF makes usda_matched
     partly seen data, so the "unseen" score would be inflated. If you also want      an
     app demo model on all OFF, train it to a separate artifacts dir — never
     report
     it as unseen.

     Predictor files

     Keep nova_predictor.py, tier_predictor.py, category_predictor.py exactly
     as they are (each is already its own train/save/load file — the thing
     you'll
     edit to try a different model). step4 is only the runner; no __main__ added     to the predictors, so imports and the 30 tests stay untouched.

     What NOT to change

     Runtime package, tfidf_lr.py, predictor logic, pipeline.py, UI/OCR,
     existing
      training/*.py, artifact paths, tests.

     Optional tidy-ups (not required for the 5 steps): wildcard imports in
     models/__init__.py pull training-only modules; NutriScan.save() no-ops
     unless a path is passed (its "saved nutriscan_v0.joblib" print is
     misleading);
     normalize_usda dedupes before ingredient filtering.

     Independence from the old lowercase nutriscan/ folder (safe to delete)

     Verified today: Nutriscan_Final has no dependency on
     /mnt/d/Downloads/NutriScan/nutriscan:
     - No source file references that path, nor any absolute D:\ or /mnt/d/
     path.
     - Every training/*.py bootstraps the package via
     Path(__file__).resolve().parent.parent / "runtime" / "src"; the app uses
     parents[1] / "src" — all resolve inside Nutriscan_Final.
     - runtime/src/nutriscan/config.py uses only __file__-relative parents
     (parents[2]/parents[3]).
     - pyproject.toml: pythonpath = ["runtime/src"], testpaths = ["tests"].
     - The venv has no editable install, .pth, .egg-link, or .egg-info
     pointing at the old folder (its nutriscan package is not installed; imports     come from runtime/src).
     - The venv lives inside Nutriscan_Final/venv, so deleting the old folder
     cannot break it.

     Rules the new step files must follow to keep this true:
     - Each training/steps/stepN_*.py begins with:
     sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" /
     "src"))
     (file sits in training/steps/, so parents[2] = Nutriscan_Final).
     - No absolute paths — read from data/…, write to data/…, artifacts/,
     reports/.
     - Run all commands from inside Nutriscan_Final, so runtime/src wins even if     the old folder is on PYTHONPATH.

     Delete-safety checklist (run before removing the old folder):
     1. Guard grep (code only): grep -rnE
     "NutriScan[\\\\/]nutriscan|D:\\\\|/mnt/d/|
     scripts/|src/nutriscan/|app/streamlit_app.py" --include=*.py
     --include=*.toml --include=*.yaml . | grep -v /venv/ → no hits in code.
     2. From Nutriscan_Final, add runtime/src explicitly (the package is not
     installed): python -c "import sys; sys.path.insert(0, 'runtime/src');
     import nutriscan; print(nutriscan.__file__)" → must print
     ...\Nutriscan_Final\runtime\src\nutriscan\__init__.py. A bare
     "import nutriscan" raises ModuleNotFoundError; that is expected and is
     NOT a dependency on the old folder.
     3. Temporarily rename the old folder (e.g. nutriscan_old), then run pytest
     -q and the five steps → all pass, then delete permanently.
     4. Stale docs (done): README.md, CLAUDE.md and CHANGELOG.md now point at
     training/, runtime/app/streamlit_app.py and runtime/src/nutriscan/ instead
     of scripts/, app/ and src/nutriscan/.

     Honest reporting notes

     - OFF derives NOVA from ingredients → TF-IDF NOVA scores are partly label
     reconstruction (leakage). Report the Nutri-Score tier and the unseen-USDA
     result
      as the headline.
     - NOVA 2 is ~0.8% of data (weak F1); offer --nova-merge12.
     - Category has no benchmark yet; step5 gives its first real unseen number.

     Verification

     1. pytest -q → 30 passed.
     2. python training/steps/step1_load_data.py --src
     data/sample/products_sample.csv → ~40 rows, loaded.parquet.
     3. python training/steps/step2_preprocess.py → non-empty
     nova_group/health_tier/category counts.
     4. python training/steps/step3_make_unseen_set.py --usda-zip <zip> --off
     .../off_clean.parquet → overlap %, three parquet paths.
     5. python training/steps/step4_train_models.py --data
     .../off_train_disjoint.parquet → overlap_safe: true, three joblibs.
     6. python training/steps/step5_evaluate.py → three metric blocks,
     unseen_metrics.csv, no contamination warning.
     7. streamlit run runtime/app/streamlit_app.py → loads the three artifacts.
     8. Import isolation: python -c "import sys; sys.path.insert(0,
     'runtime/src'); import nutriscan; print(nutriscan.__file__)"
     prints ...\Nutriscan_Final\runtime\src\nutriscan\__init__.py. (A bare
     "import nutriscan" fails because runtime/src is not on sys.path.)
     9. Guard grep (see delete-safety checklist) has no code hits.
     10. Delete-safety: rename the old lowercase folder, re-run pytest -q + the
     five steps, confirm green, then delete it.

     "/plan open" to edit this plan in VS Code