"""
backend/whisper_asr.py

Local / Offline Streaming ASR service using faster-whisper.
Collects PCM 16kHz audio chunks, detects speech activity, and emits partial and final transcripts.
"""

import os
import time
import math
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, AsyncGenerator, Dict, Any, Tuple
import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger("whisper-asr")

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "faster-whisper-small"
    / "models--Systran--faster-whisper-small"
    / "snapshots"
    / "536b0662742c02347bc0e980a01041f333bce120"
)

# Language code normalization for Whisper
WHISPER_LANG_MAP: Dict[str, str] = {
    "ta": "ta",
    "ta-in": "ta",
    "tamil": "ta",
    "hi": "hi",
    "hi-in": "hi",
    "hindi": "hi",
    "te": "te",
    "te-in": "te",
    "telugu": "te",
    "kn": "kn",
    "kn-in": "kn",
    "kannada": "kn",
    "ml": "ml",
    "ml-in": "ml",
    "malayalam": "ml",
    "en": "en",
    "en-in": "en",
    "english": "en",
    "mr": "mr",
    "mr-in": "mr",
    "bn": "bn",
    "bn-in": "bn",
    "gu": "gu",
    "gu-in": "gu",
    "pa": "pa",
    "pa-in": "pa",
}


class LocalWhisperASR:
    _instance: Optional["LocalWhisperASR"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.getenv("WHISPER_MODEL_PATH", str(DEFAULT_MODEL_DIR))

        resolved_path = Path(model_path).resolve()
        if not resolved_path.exists():
            # Fallback to parent dir or model name
            alt_path = Path(__file__).resolve().parent.parent / "models" / "faster-whisper-small"
            if alt_path.exists():
                resolved_path = alt_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        logger.info(f"Loading Faster-Whisper model from {resolved_path} on {self.device} ({self.compute_type})...")
        t0 = time.time()
        self.model = WhisperModel(
            str(resolved_path),
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
        )
        logger.info(f"Faster-Whisper model loaded in {round(time.time() - t0, 2)}s")

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "LocalWhisperASR":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    @staticmethod
    def normalize_language(lang_code: str) -> str:
        code = lang_code.strip().lower()
        return WHISPER_LANG_MAP.get(code, code.split("-")[0].split("_")[0])

    def transcribe_pcm(self, pcm_bytes: bytes, language: str) -> Tuple[str, float]:
        """Transcribe raw 16-bit 16kHz linear PCM audio buffer."""
        if not pcm_bytes or len(pcm_bytes) < 3200:  # less than 100ms
            return "", 0.0

        whisper_lang = self.normalize_language(language)
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Run inference
        segments, info = self.model.transcribe(
            audio_np,
            language=whisper_lang,
            beam_size=1,
            condition_on_previous_text=False,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)

        full_text = " ".join(texts).strip()
        confidence = float(info.language_probability) if hasattr(info, "language_probability") else 0.95
        return full_text, confidence

    async def transcribe_pcm_async(self, pcm_bytes: bytes, language: str) -> Tuple[str, float]:
        return await asyncio.to_thread(self.transcribe_pcm, pcm_bytes, language)
