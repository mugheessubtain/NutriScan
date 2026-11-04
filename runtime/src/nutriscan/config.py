"""Central paths and constants. Import from here instead of hard-coding paths.

Set the environment variable NUTRISCAN_OUT to a folder to send every GENERATED file (data/processed, artifacts,
reports) there instead of the project. Tests and smoke runs on the synthetic sample use this so they can never
overwrite real models or results. Inputs (data/raw, data/sample, knowledge_base) always come from the project.
"""
import os
from pathlib import Path

# In the new runtime/ structure:
# - config.py is at runtime/src/nutriscan/config.py
# parents[0] = nutritional_final/runtime/src/nutriscan
# parents[1] = nutritional_final/runtime/src
# parents[2] = nutritional_final/runtime
# parents[3] = nutritional_final
# - knowledge_base is at runtime/knowledge_base/
# - data/artifacts/reports are at Nutriscan_Final/ (sibling of runtime/)
RUNTIME_ROOT = Path(__file__).resolve().parents[2]   # runtime/
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Nutriscan_Final/

OUT_ROOT = Path(os.environ["NUTRISCAN_OUT"]).resolve() if os.environ.get("NUTRISCAN_OUT") else PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
PROCESSED = OUT_ROOT / "data" / "processed"
SAMPLE_CSV = DATA / "sample" / "products_sample.csv"
OFF_PROCESSED = PROCESSED / "off_en.parquet"      # produced by training/prepare_off.py
USDA_PROCESSED = PROCESSED / "usda_branded.parquet"  # produced by training/steps/step3_make_unseen_set.py
KB_DIR = RUNTIME_ROOT / "knowledge_base"
ARTIFACTS = OUT_ROOT / "artifacts"
REPORTS = OUT_ROOT / "reports"


def resolve(path) -> Path:
    """CLI path -> absolute. Generated locations (data/processed, artifacts, reports) go to OUT_ROOT, the rest to the project."""
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    generated = parts[:1] in (("artifacts",), ("reports",)) or parts[:2] == ("data", "processed")
    return (OUT_ROOT if generated else PROJECT_ROOT) / p

KB_VERSION = "0.2.0"
RANDOM_STATE = 42

# Health tier derived from Nutri-Score grade (T2 target)
def health_tier(grade: str) -> str:
    g = str(grade).strip().lower()
    if g in ("a", "b"):
        return "good"
    if g == "c":
        return "medium"
    if g in ("d", "e"):
        return "poor"
    return "unknown"

TIER_ORDER = {"poor": 0, "medium": 1, "good": 2}
DISCLAIMER = ("NutriScan uses established classifications (NOVA, Nutri-Score) and published ratings. "
              "It is not medical advice.")

# ---- v0.2 additions (validation report) ----
USDA_MATCHED = PROCESSED / "usda_matched.parquet"   # USDA text + OFF labels (external test set)
OFF_META = PROCESSED / "off_en.meta.json"           # export date, schema_version, counts (audit log)

# ---- v0.3 additions (plan v2: EDA, USDA targets, model comparison, gallery) ----
RAW = DATA / "raw"
OFF_RAW = RAW / "food.parquet"                                  # downloaded from huggingface.co/datasets/openfoodfacts/product-database
OFF_CLEAN = PROCESSED / "off_clean.parquet"            # step2 output
OFF_TRAIN_DISJOINT = PROCESSED / "off_train_disjoint.parquet"  # step3 output (OFF minus USDA barcodes)
USDA_LABELED = PROCESSED / "usda_labeled.parquet"      # step3b: USDA + mapped category + nutrient-based tier
GALLERY = PROCESSED / "gallery.parquet"                # small, pre-normalised product gallery for the app
USDA_SAMPLE_ZIP = DATA / "sample" / "usda_branded_sample.zip"   # synthetic, written by training/make_sample_data.py
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
GALLERY_MAX_ROWS = 60000                                        # keeps app start-up fast (TF-IDF index over the gallery)
NUTRIENT_COLS = ["energy_kcal_100g", "sugars_100g", "saturated_fat_100g", "salt_100g", "proteins_100g", "fiber_100g"]

# Exact ODbL attribution wording (OFF requires prominent attribution) + USDA citation.
ATTRIBUTION_MD = (
    "Contains data from [Open Food Facts](https://world.openfoodfacts.org), available under the "
    "[Open Database License](https://opendatacommons.org/licenses/odbl/1-0/). "
    "Product ingredient data also from [USDA FoodData Central](https://fdc.nal.usda.gov) (public domain)."
)


