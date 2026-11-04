"""External evaluation (validation report section 1): headline (a) product-disjoint vs bonus (b) same-product transcription.

  python training/eval_external.py [--target nova_group|health_tier] [--nova-merge12] [--lang en]

Writes reports/tables/external_<target>.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.data import clean_products, load_products, supervised_view  # noqa: E402
from nutriscan.evaluation import merge_nova12, per_class_report, summary  # noqa: E402
from nutriscan.external import add_code_canon, external_eval_sets  # noqa: E402
from nutriscan.models.tfidf_lr import make_tfidf_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="nova_group", choices=["nova_group", "health_tier"])
    ap.add_argument("--nova-merge12", action="store_true")
    ap.add_argument("--lang", default=None)
    a = ap.parse_args()
    if not (config.USDA_PROCESSED.exists() and config.OFF_PROCESSED.exists()):
        sys.exit("Run training/steps/step2_preprocess.py and training/steps/step3_make_unseen_set.py first.")

    off = add_code_canon(load_products(config.OFF_PROCESSED, lang=a.lang))
    usda = pd.read_parquet(config.USDA_PROCESSED)
    sets = external_eval_sets(off, usda)
    test = supervised_view(clean_products(sets["external_matched"]), a.target)

    def prep(y):
        y = y.astype(int) if a.target == "nova_group" else y
        return merge_nova12(y) if (a.nova_merge12 and a.target == "nova_group") else y

    rows = []
    for name, pool in (("(a) train_disjoint -> USDA text", sets["train_disjoint"]),
                       ("(b) train_full -> USDA text [same-product, independent transcription]", sets["train_full"])):
        tr = supervised_view(pool, a.target)
        model = make_tfidf_pipeline(class_weight="balanced").fit(tr["ingredients_norm"], prep(tr[a.target]))
        pred = model.predict(test["ingredients_norm"])
        yt = prep(test[a.target])
        rows.append({"setting": name, "n_train": len(tr), "n_test": len(test),
                     **summary(yt, pred, ordinal=a.target == "nova_group")})
        print(name); print(per_class_report(yt, pred).to_string(index=False))
    table = pd.DataFrame(rows).round(3)
    out = config.REPORTS / "tables" / f"external_{a.target}.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False)); print(f"Saved {out}")


if __name__ == "__main__":
    main()