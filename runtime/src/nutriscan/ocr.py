"""Section 11 OCR. Heavy imports are lazy so the rest of the project works without OCR installed.

Default engine: RapidOCR (PaddleOCR models on ONNX Runtime). It installs with pip alone, unlike Tesseract, which
needs a separate Windows installer and a PATH entry. Tesseract and EasyOCR stay available for the comparison in
training/eval_ocr.py; change DEFAULT_ENGINE if your own photos favour another engine.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate so text lines are horizontal (minimum-area rectangle around dark pixels); skips tiny angles."""
    import cv2
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] > 0))
    if len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 30:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(image_bgr: np.ndarray, binarize: bool = False) -> np.ndarray:
    """Grayscale -> upscale small images -> denoise -> deskew -> contrast (CLAHE, softens glare) [-> threshold].

    Neural engines (RapidOCR, EasyOCR) do better on the grayscale result; Tesseract on the binarised one."""
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    if max(gray.shape) < 1500:                                   # small label text: upscale before OCR
        f = 1500 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = _deskew(gray)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if binarize:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return gray


@lru_cache(maxsize=1)
def _easyocr_reader(langs: tuple[str, ...]):
    import easyocr
    return easyocr.Reader(list(langs), gpu=False)


@lru_cache(maxsize=1)
def _rapidocr_engine():
    try:                                   # newer package
        from rapidocr import RapidOCR
        return RapidOCR(), "new"
    except ImportError:                    # older package name
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR(), "old"


def ocr_easyocr(image: np.ndarray, langs=("en",)) -> str:
    return " ".join(_easyocr_reader(tuple(langs)).readtext(image, detail=0, paragraph=True))


def ocr_tesseract(image: np.ndarray) -> str:
    import pytesseract
    return pytesseract.image_to_string(image, config="--psm 6")


def ocr_rapidocr(image: np.ndarray) -> str:
    """RapidOCR: `pip install rapidocr onnxruntime`. No Paddle/torch and no system install needed."""
    engine, kind = _rapidocr_engine()
    if kind == "new":
        res = engine(image)
        return " ".join(res.txts or [])
    result, _ = engine(image)
    return " ".join(line[1] for line in (result or []))


ENGINES = {"rapidocr": ocr_rapidocr, "tesseract": ocr_tesseract, "easyocr": ocr_easyocr}
DEFAULT_ENGINE = "rapidocr"


def read_label(image_bytes: bytes, engine: str = DEFAULT_ENGINE, do_preprocess: bool = True) -> str:
    """Decode an uploaded photo in memory (never written to disk) and return the OCR text."""
    import cv2
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    proc = preprocess(img, binarize=engine == "tesseract") if do_preprocess else img
    return ENGINES[engine](proc)
