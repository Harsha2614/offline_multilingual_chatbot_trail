import os
import pickle
from langdetect import detect, DetectorFactory

# Stabilize langdetect
DetectorFactory.seed = 0

# Universal translator model id
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Map ISO codes (langdetect output) -> NLLB codes
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "fr": "fra_Latn",
    "fi": "fin_Latn",
    "sw": "swa_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "bn": "ben_Beng",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ur": "urd_Arab",
    "id": "ind_Latn",
    "pt": "por_Latn",   # Portuguese
    # add more if needed
}

def detect_lang(text: str) -> str:
    """
    Detect language code using langdetect. Returns ISO code (like 'en').
    Falls back to 'en' on error.
    """
    try:
        return detect(text)
    except Exception:
        return "en"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
