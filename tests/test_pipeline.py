from nutriscan import config
from nutriscan.data import load_products
from nutriscan.pipeline import NutriScan


def test_end_to_end_on_sample():
    df = load_products(config.SAMPLE_CSV)   # explicit: never train on the real parquet during tests
    assert len(df) > 20 and "ingredients_norm" in df
    ns = NutriScan.train(df, data_source="sample")
    r = ns.analyze("Sugar, wheat flour, hydrogenated vegetable oil, glucose syrup, flavouring, colour (E102)")
    assert r["nova_final"] == 4
    assert any(f.kind == "rule" for f in r["findings"])
    assert r["versions"]["kb"]
    r2 = ns.analyze("")
    assert "error" in r2
