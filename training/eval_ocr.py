"""Compare OCR engines (Tesseract, RapidOCR, EasyOCR) on your own photos. Decide the primary engine EMPIRICALLY.

Folder layout:  <dir>/photo1.jpg + <dir>/photo1.txt  (ground-truth ingredient list typed by hand)
  python training/eval_ocr.py --dir data/ocr_test --engines tesseract rapidocr easyocr

Reports mean WER/CER on the extracted ingredient section and ingredient-extraction F1. Saves reports/tables/ocr_eval.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.evaluation import char_error_rate, ingredient_extraction_accuracy, word_error_rate  # noqa: E402
from nutriscan.ocr import read_label  # noqa: E402
from nutriscan.parser import extract_ingredient_section, ingredient_names  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--engines", nargs="+", default=["tesseract", "rapidocr", "easyocr"])
    ap.add_argument("--no-preprocess", action="store_true")
    a = ap.parse_args()
    imgs = sorted(p for p in Path(a.dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.with_suffix(".txt").exists())
    if not imgs:
        sys.exit("No image+txt pairs found.")
    rows = []
    for eng in a.engines:
        wers, cers, f1s, failed = [], [], [], 0
        for p in imgs:
            ref = p.with_suffix(".txt").read_text(encoding="utf-8")
            try:
                hyp = extract_ingredient_section(read_label(p.read_bytes(), engine=eng, do_preprocess=not a.no_preprocess))
            except Exception as e:  # engine not installed / failed
                print(f"[{eng}] {p.name}: {e}"); failed += 1; continue
            wers.append(word_error_rate(ref, hyp)); cers.append(char_error_rate(ref, hyp))
            f1s.append(ingredient_extraction_accuracy(ingredient_names(ref), ingredient_names(hyp)))
        n = max(len(wers), 1)
        rows.append({"engine": eng, "images": len(imgs), "failed": failed, "WER": sum(wers) / n,
                     "CER": sum(cers) / n, "ingredient_F1": sum(f1s) / n})
    table = pd.DataFrame(rows).round(3)
    out = config.REPORTS / "tables" / "ocr_eval.csv"
    table.to_csv(out, index=False); print(table.to_string(index=False)); print(f"Saved {out}")


if __name__ == "__main__":
    main()