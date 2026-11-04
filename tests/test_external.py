import pandas as pd

from nutriscan.external import canonical_barcode, exclude_overlap, external_eval_sets, normalize_usda


def test_canonical_barcode():
    assert canonical_barcode("0012345678905") == "12345678905"
    assert canonical_barcode("012345678905") == "12345678905"      # UPC-A vs EAN-13 collapse together
    assert canonical_barcode("00000") is None and canonical_barcode(None) is None


def _usda():
    return pd.DataFrame({
        "fdc_id": ["1", "2", "3", "4"],
        "gtin_upc": ["012345678905", "12345678905", "000099999999", "055555555555"],
        "brand_owner": ["A", "A", "B", "C"], "brand_name": ["x", "x2", "y", "z"],
        "ingredients": ["SUGAR, SALT", "SUGAR, SALT, FLAVOUR", "OATS", "RICE"],
        "branded_food_category": ["c"] * 4,
        "modified_date": ["2020-01-01", "2024-01-01", "2023-01-01", "2023-01-01"],
        "market_country": ["US"] * 4, "discontinued_date": [None] * 4,
    })


def _off():
    return pd.DataFrame({
        "code": ["0012345678905", "99999999", "777"], "brand": ["a", "b", "c"],
        "ingredients_text": ["sugar, salt", "oats", "water"],
        "nova_group": [4.0, 1.0, 3.0], "nutriscore_grade": ["d", "a", "b"],
        "health_tier": ["poor", "good", "good"], "category": ["snacks", "cereals", "drinks"],
    })


def test_usda_dedupe_keeps_latest():
    u = normalize_usda(_usda())
    assert len(u) == 3
    row = u[u.code_canon == "12345678905"].iloc[0]
    assert "FLAVOUR" in row.ingredients_text           # latest modified_date wins


def test_eval_sets_are_product_disjoint():
    sets = external_eval_sets(_off(), normalize_usda(_usda()))
    train_codes = set(sets["train_disjoint"]["code_canon"])
    ext_codes = set(sets["external_matched"]["code_canon"])
    assert not (train_codes & ext_codes)                # headline set has no leakage across barcodes
    assert ext_codes == {"12345678905", "99999999"}
    assert len(sets["train_full"]) == 3
    assert list(sets["usda_unlabeled"]["code_canon"]) == ["55555555555"]
    assert set(sets["external_matched"]["nova_group"]) == {4.0, 1.0}   # labels come from OFF


def test_exclude_overlap():
    out = exclude_overlap(_off(), ["12345678905"])
    assert list(out["code"]) == ["99999999", "777"]
