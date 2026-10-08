"""
backend/whisper_asr.py

Local faster-whisper ASR integration for HackNex EPS03.
- Model loaded once at startup.
- Runtime CUDA detection with graceful fallback to CPU.
- Decoupled streaming session for low-latency live transcription.
"""

import os
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import faster_whisper
import ctranslate2

logger = logging.getLogger("whisper-asr")

# Global singleton model instance
_MODEL: Optional[faster_whisper.WhisperModel] = None
_MODEL_INFO: Dict[str, Any] = {
    "model": "base",
    "device": "cpu",
    "compute_type": "int8",
    "loaded": False,
}

# Language code normalization for Whisper (ISO-639-1)
LANG_MAP = {
    "ta-IN": "ta",
    "ta": "ta",
    "hi-IN": "hi",
    "hi": "hi",
    "en-IN": "en",
    "en": "en",
}


def load_whisper_model(model_name: Optional[str] = None) -> faster_whisper.WhisperModel:
    """
    Load the faster-whisper model once.
    Detects CUDA availability at runtime and performs a verification test.
    Gracefully falls back to CPU (int8) if CUDA libraries (e.g. cublas) fail to load.
    """
    global _MODEL, _MODEL_INFO

    if _MODEL is not None:
        return _MODEL

    selected_model = model_name or os.getenv("WHISPER_MODEL", "base")
    logger.info(f"Initializing faster-whisper model: {selected_model}")

    device = "cpu"
    compute_type = "int8"
    cpu_threads = min(8, max(4, (os.cpu_count() or 4) // 2))

    # 1. Runtime CUDA detection
    if ctranslate2.get_cuda_device_count() > 0:
        try:
            logger.info("CUDA device detected. Testing faster-whisper on GPU (float16)...")
            gpu_model = faster_whisper.WhisperModel(
                selected_model,
                device="cuda",
                compute_type="float16",
            )
            # Run test inference on 0.1s silence to confirm cublas/CUDA DLLs work
            dummy = np.zeros(1600, dtype=np.float32)
            list(gpu_model.transcribe(dummy, beam_size=1)[0])
            _MODEL = gpu_model
            device = "cuda"
            compute_type = "float16"
            logger.info("Successfully initialized faster-whisper on CUDA GPU (float16).")
        except Exception as e:
            logger.warning(
                f"CUDA initialization or test failed ({e}). Gracefully falling back to CPU (int8)."
            )
            _MODEL = None

    # 2. CPU Fallback
    if _MODEL is None:
        logger.info(
            f"Loading faster-whisper model '{selected_model}' on CPU (int8) with {cpu_threads} threads..."
        )
        t0 = time.time()
        _MODEL = faster_whisper.WhisperModel(
            selected_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
        )
        logger.info(f"Loaded faster-whisper '{selected_model}' on CPU in {time.time()-t0:.2f}s.")
        device = "cpu"
        compute_type = "int8"

    _MODEL_INFO = {
        "model": selected_model,
        "device": device,
        "compute_type": compute_type,
        "loaded": True,
    }
    return _MODEL


def get_model() -> faster_whisper.WhisperModel:
    global _MODEL
    if _MODEL is None:
        return load_whisper_model()
    return _MODEL


def get_model_info() -> Dict[str, Any]:
    return dict(_MODEL_INFO)


def transcribe_audio_array(audio: np.ndarray, language: str) -> Tuple[str, float]:
    """
    Run Whisper model transcription on a numpy float32 audio array.
    Designed to be called inside loop.run_in_executor to avoid blocking the asyncio event loop.
    """
    if len(audio) == 0:
        return "", 0.0

    model = get_model()
    t_start = time.perf_counter()

    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    text = " ".join(seg.text for seg in segments).strip()
    latency_ms = (time.perf_counter() - t_start) * 1000
    return text, latency_ms


class WhisperStreamingSession:
    """
    Manages audio buffering and state for a live streaming session.
    """

    SAMPLE_RATE = 16000
    SILENCE_RMS_THRESHOLD = 0.0035
    # ~1.2s silence after speech triggers utterance completion
    SILENCE_CHUNKS_FOR_ENDPOINT = 14
    # Minimum audio (~0.4s) before first transcription
    MIN_SAMPLES_FOR_EVAL = 6400
    # Minimum new audio (~0.35s) before next transcription
    MIN_NEW_SAMPLES_STEP = 5600
    # Minimum interval between transcription runs (0.3s)
    MIN_UPDATE_INTERVAL_S = 0.30

    def __init__(self, language_code: str = "ta-IN", utterance_idx: int = 0):
        self.language_code = language_code
        self.whisper_lang = LANG_MAP.get(language_code, language_code.split("-")[0].lower())
        self.utterance_idx = utterance_idx

        self.buffer: List[np.ndarray] = []
        self.total_samples: int = 0

        self.speech_started: bool = False
        self.consecutive_silence_chunks: int = 0
        self.samples_at_last_transcribe: int = 0
        self.last_transcribe_time: float = 0.0
        self.last_hypothesis: str = ""
        self.window_start_time: Optional[float] = None
        self.is_transcribing: bool = False

    def feed(self, pcm_bytes: bytes) -> Tuple[bool, bool]:
        """
        Feed 16-bit 16kHz mono PCM audio chunk.
        Returns (is_silent, should_finalize).
        """
        if not pcm_bytes:
            return True, False

        chunk_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(chunk_int16) == 0:
            return True, False

        chunk_f32 = chunk_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(chunk_f32 ** 2)))
        is_silent = rms < self.SILENCE_RMS_THRESHOLD

        if self.window_start_time is None:
            self.window_start_time = time.perf_counter()

        self.buffer.append(chunk_f32)
        self.total_samples += len(chunk_f32)

        should_finalize = False
        if not is_silent:
            self.speech_started = True
            self.consecutive_silence_chunks = 0
        else:
            if self.speech_started:
                self.consecutive_silence_chunks += 1
                if self.consecutive_silence_chunks >= self.SILENCE_CHUNKS_FOR_ENDPOINT:
                    should_finalize = True

        return is_silent, should_finalize

    def should_transcribe_partial(self) -> bool:
        """Check if conditions are met to transcribe a new partial hypothesis."""
        if self.is_transcribing:
            return False

        if self.total_samples < self.MIN_SAMPLES_FOR_EVAL:
            return False

        new_samples = self.total_samples - self.samples_at_last_transcribe
        if new_samples < self.MIN_NEW_SAMPLES_STEP:
            return False

        now = time.perf_counter()
        if (now - self.last_transcribe_time) < self.MIN_UPDATE_INTERVAL_S:
            return False

        return True

    def get_audio_snapshot(self, max_seconds: float = 15.0) -> np.ndarray:
        """Return a numpy copy of current buffered audio, capped to max_seconds."""
        if not self.buffer:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(self.buffer)
        max_samples = int(max_seconds * self.SAMPLE_RATE)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        return audio

    def record_partial_done(self, hypothesis: str):
        """Update state after a partial transcription completes."""
        self.samples_at_last_transcribe = self.total_samples
        self.last_transcribe_time = time.perf_counter()
        self.last_hypothesis = hypothesis

    def reset(self, new_utterance_idx: int):
        """Reset buffer and state for the next utterance."""
        self.utterance_idx = new_utterance_idx
        self.buffer = []
        self.total_samples = 0
        self.speech_started = False
        self.consecutive_silence_chunks = 0
        self.samples_at_last_transcribe = 0
        self.last_transcribe_time = 0.0
        self.last_hypothesis = ""
        self.window_start_time = None
        self.is_transcribing = False
