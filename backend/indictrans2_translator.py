"""
backend/indictrans2_translator.py

Local / Offline IndicTrans2 translation service for IndieTranslate.
Scope: Indic -> Indic language pairs using local ai4bharat/indictrans2-indic-indic-dist-320M model.
"""

import os
import sys
import time
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict

import torch

# Diagnostic / Compatibility shim:
# In newer transformers (v5.x), PreTrainedTokenizerBase is in tokenization_utils_base,
# whereas IndicTransToolkit imports it from tokenization_utils.
try:
    import transformers.tokenization_utils
    import transformers.tokenization_utils_base
    if not hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase"):
        transformers.tokenization_utils.PreTrainedTokenizerBase = (
            transformers.tokenization_utils_base.PreTrainedTokenizerBase
        )
except Exception:
    pass

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

logger = logging.getLogger("indictrans2-translator")

# Centralized Language Code Mapping
# Maps standard BCP-47 / ISO codes to IndicTrans2 Flores codes
LANGUAGE_MAP: Dict[str, str] = {
    # Tamil
    "ta": "tam_Taml",
    "ta-in": "tam_Taml",
    "tam_taml": "tam_Taml",
    # Hindi
    "hi": "hin_Deva",
    "hi-in": "hin_Deva",
    "hin_deva": "hin_Deva",
    # Telugu
    "te": "tel_Telu",
    "te-in": "tel_Telu",
    "tel_telu": "tel_Telu",
    # Kannada
    "kn": "kan_Knda",
    "kn-in": "kan_Knda",
    "kan_knda": "kan_Knda",
    # Malayalam
    "ml": "mal_Mlym",
    "ml-in": "mal_Mlym",
    "mal_mlym": "mal_Mlym",
    # Marathi
    "mr": "mar_Deva",
    "mr-in": "mar_Deva",
    "mar_deva": "mar_Deva",
    # Bengali
    "bn": "ben_Beng",
    "bn-in": "ben_Beng",
    "ben_beng": "ben_Beng",
    # Gujarati
    "gu": "guj_Gujr",
    "gu-in": "guj_Gujr",
    "guj_gujr": "guj_Gujr",
    # Punjabi
    "pa": "pan_Guru",
    "pa-in": "pan_Guru",
    "pan_guru": "pan_Guru",
    # Odia
    "or": "ory_Orya",
    "or-in": "ory_Orya",
    "ory-in": "ory_Orya",
    "ory_orya": "ory_Orya",
    # Assamese
    "as": "asm_Beng",
    "as-in": "asm_Beng",
    "asm_beng": "asm_Beng",
    # Urdu
    "ur": "urd_Arab",
    "ur-in": "urd_Arab",
    "urd_arab": "urd_Arab",
    # Sanskrit
    "sa": "san_Deva",
    "sa-in": "san_Deva",
    "san_deva": "san_Deva",
    # Maithili
    "mai": "mai_Deva",
    "mai-in": "mai_Deva",
    "mai_deva": "mai_Deva",
    # Kashmiri
    "ks": "kas_Arab",
    "ks-in": "kas_Arab",
    "kas_arab": "kas_Arab",
    # Nepali
    "ne": "npi_Deva",
    "ne-in": "npi_Deva",
    "npi_deva": "npi_Deva",
    # Sindhi
    "sd": "snd_Deva",
    "sd-in": "snd_Deva",
    "snd_deva": "snd_Deva",
    # Konkani / Goan Konkani
    "gom": "gom_Deva",
    "gom-in": "gom_Deva",
    "kok": "gom_Deva",
    "kok-in": "gom_Deva",
    "gom_deva": "gom_Deva",
    # Bodo
    "brx": "brx_Deva",
    "brx-in": "brx_Deva",
    "brx_deva": "brx_Deva",
    # Dogri
    "doi": "doi_Deva",
    "doi-in": "doi_Deva",
    "doi_deva": "doi_Deva",
    # Santali
    "sat": "sat_Olck",
    "sat-in": "sat_Olck",
    "sat_olck": "sat_Olck",
    # Manipuri
    "mni": "mni_Mtei",
    "mni-in": "mni_Mtei",
    "mni_mtei": "mni_Mtei",
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "indictrans2-indic-indic-320M"


class IndicTrans2Translator:
    _instance: Optional["IndicTrans2Translator"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.getenv("INDICTRANS2_MODEL_PATH", str(DEFAULT_MODEL_DIR))

        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"IndicTrans2 model directory not found at: {self.model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading IndicTrans2 model from {self.model_path} on {self.device} ({self.dtype})...")
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            dtype=self.dtype,
            local_files_only=True,
        ).to(self.device)

        self.model.eval()
        self.ip = IndicProcessor(inference=True)

        load_sec = round(time.time() - t0, 3)
        logger.info(f"IndicTrans2 model loaded successfully in {load_sec}s on {self.device}")

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "IndicTrans2Translator":
        """Singleton accessor to ensure model is loaded only once."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    @staticmethod
    def normalize_lang_code(lang_code: str) -> str:
        """
        Convert BCP-47 / common language codes to IndicTrans2 Flores code.
        Rejects unsupported languages (such as English).
        """
        normalized = lang_code.strip().lower()
        if normalized.startswith("en") or "eng" in normalized:
            raise ValueError(
                f"IndicTrans2 Indic-Indic model does not support English ('{lang_code}'). "
                "Only Indic-to-Indic language pairs are supported by this model."
            )

        mapped = LANGUAGE_MAP.get(normalized)
        if not mapped:
            # Check prefix match (e.g. 'ta' in 'ta-in')
            prefix = normalized.split("-")[0].split("_")[0]
            mapped = LANGUAGE_MAP.get(prefix)

        if not mapped:
            raise ValueError(
                f"Unsupported language code for IndicTrans2: '{lang_code}'. "
                f"Supported Indic languages: {list(set(LANGUAGE_MAP.values()))}"
            )
        return mapped

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a single text from source_lang to target_lang.
        Logs: source language, target language, latency, translated text.
        """
        cleaned = text.strip()
        if not cleaned:
            return ""

        try:
            src_flores = self.normalize_lang_code(source_lang)
            tgt_flores = self.normalize_lang_code(target_lang)
        except Exception as e:
            logger.warning(f"IndicTrans2 language pair ({source_lang} -> {target_lang}) not supported: {e}. Returning original text.")
            return cleaned

        if src_flores == tgt_flores:
            return cleaned

        t_start = time.time()

        # Preprocessing with IndicProcessor
        batch = self.ip.preprocess_batch([cleaned], src_lang=src_flores, tgt_lang=tgt_flores)

        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            max_length=256,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=1,
                num_return_sequences=1,
            )

        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations = self.ip.postprocess_batch(decoded, lang=tgt_flores)
        translated_text = translations[0].strip() if translations else ""

        latency_ms = round((time.time() - t_start) * 1000, 2)
        logger.info(
            f"IndicTrans2 [{src_flores} -> {tgt_flores}] "
            f"Latency: {latency_ms}ms | Translated: '{translated_text}'"
        )

        return translated_text

    async def translate_async(self, text: str, source_lang: str, target_lang: str) -> str:
        """Non-blocking async wrapper running inference in worker thread."""
        return await asyncio.to_thread(self.translate, text, source_lang, target_lang)


# Convenience module-level function
def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Convenience helper to translate text using singleton translator instance."""
    translator = IndicTrans2Translator.get_instance()
    return translator.translate(text, source_lang, target_lang)
