"""Run the whole NutriScan pipeline in order (prepare -> steps 1-8).

  python training/run_all.py --usda-zip data/raw/FoodData_Central_branded_food_csv_<date>.zip     # real data
  python training/run_all.py --demo                  # synthetic sample, all outputs go to ./_demo_out (never the project)
  python training/run_all.py --from 6 --usda-zip ... # resume from step 6
  python training/run_all.py --from 1 --until 3.5      # cleaning steps only (fills the Data cleaning page)

Real-data prerequisites: data/raw/food.parquet (Open Food Facts) and the USDA Branded Foods CSV zip in data/raw/.
Each step is a separate script in training/steps/; this file only calls them with sensible arguments.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usda-zip", default=None, help="FoodData Central branded zip (default: newest zip in data/raw)")
    ap.add_argument("--demo", action="store_true", help="synthetic sample, outputs in ./_demo_out")
    ap.add_argument("--from", dest="start", type=float, default=0, help="first step to run (0 = prepare_off)")
    ap.add_argument("--until", type=float, default=99, help="last step to run, e.g. 3.5 = cleaning only (steps 1-3b)")
    ap.add_argument("--model", default="voting", help="final model for step4 (tfidf_lr is faster)")
    ap.add_argument("--max-train", default="150000")
    a = ap.parse_args()

    env = dict(os.environ)
    if a.demo:
        env["NUTRISCAN_OUT"] = str(ROOT / "_demo_out")
        subprocess.run([PY, "training/make_sample_data.py"], cwd=ROOT, check=True, env=env)
        off_src, usda = "data/sample/products_sample.csv", "data/sample/usda_branded_sample.zip"
        quick = ["--quick"]
    else:
        zips = sorted((ROOT / "data" / "raw").glob("*.zip"))
        usda = a.usda_zip or (str(zips[-1]) if zips else None)
        if not usda:
            sys.exit("No USDA zip found in data/raw/. Download 'Branded Foods' (CSV) from fdc.nal.usda.gov.")
        off_src, quick = "data/processed/off_en.parquet", []

    steps = [
        (0, ["training/prepare_off.py", "--lang", "en"]),
        (1, ["training/steps/step1_load_data.py", "--src", off_src, "--lang", "en"]),
        (2, ["training/steps/step2_preprocess.py"]),
        (3, ["training/steps/step3_make_unseen_set.py", "--usda-zip", usda]),
        (3.5, ["training/steps/step3b_usda_targets.py", "--usda-zip", usda]),
        (2.5, ["training/steps/step2b_eda.py"]),
        (6, ["training/steps/step6_compare_models.py", "--max-train", a.max_train, *quick]),
        (4, ["training/steps/step4_train_models.py", "--model", "tfidf_lr" if a.demo else a.model, "--tuned", "--add-usda"]),
        (5, ["training/steps/step5_evaluate.py"]),
        (7, ["training/steps/step7_ablation_robustness.py", "--max-train", a.max_train]),
        (8, ["training/steps/step8_eval_alternatives.py"]),
    ]
    for num, cmd in steps:
        if num < a.start or num > a.until or (a.demo and num == 0):
            continue
        print(f"\n{'=' * 78}\n>>> step {num}: {' '.join(cmd)}\n{'=' * 78}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([PY, *cmd], cwd=ROOT, env=env)
        if r.returncode:
            sys.exit(f"step {num} failed (exit {r.returncode}). Fix it, then resume with --from {num}")
        print(f"<<< step {num} done in {time.perf_counter() - t0:.0f}s", flush=True)
    out = env.get("NUTRISCAN_OUT", str(ROOT))
    print(f"\nAll steps finished. Outputs under {out}. Start the app:  streamlit run runtime/app/streamlit_app.py")


if __name__ == "__main__":
    main()
