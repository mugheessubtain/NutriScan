# Sample label photos

Rendered test labels for the **Upload label photo** button. Brands and products are fictional; ingredient lists
are typical of products sold in Pakistan. Each `.jpg` has a `.txt` with the exact ingredient list (ground truth).

| File | What it tests |
|---|---|
| `01_cream_biscuits_clear` | clean photo, long list with E-numbers |
| `02_nimko_mix_clear` | Pakistani snack, MSG |
| `03_cola_angled` | photo taken at an angle (text is straightened automatically) |
| `04_instant_noodles_blurry` | blur and noise, two-part ingredient list |
| `05_ketchup_glare` | light glare over the text |
| `06_rolled_oats_clear` | single-ingredient product (NOVA 1) |
| `07_mango_juice_small_angled` | small, low-resolution and tilted |
| `08_banaspati_ghee_noisy` | grainy photo; partially hydrogenated oil (trans fat rule) |

Score the OCR on all of them (word error rate and ingredient F1):
```
python training/eval_ocr.py --dir samples --engines rapidocr
```
For the report, also add 10 or more real photos taken with a phone, each with a typed `.txt` next to it.
