"""
backend/main.py

HackNex 2026 EPS03 — IndieTranslate: Two-Device Bidirectional Live Translation for Indic Languages

Architecture:
  Device A (Participant A: Language A)
    Microphone → WebSocket A → Independent Sarvam STT
    → Stability Analyzer → Completeness Analyzer → Commit Engine
    → Context/Entity Protector → Sarvam Translation (A → B)
    → Route to Participant B ONLY → Device B Browser TTS + Captions

  Device B (Participant B: Language B)
    Microphone → WebSocket B → Independent Sarvam STT
    → Stability Analyzer → Completeness Analyzer → Commit Engine
    → Context/Entity Protector → Sarvam Translation (B → A)
    → Route to Participant A ONLY → Device A Browser TTS + Captions
"""

import os
import json
import base64
import asyncio
import logging
import secrets
from typing import Optional, Tuple, List, Dict, Any
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sarvamai.client import AsyncSarvamAI
import numpy as np
import time

from stability_analyzer import ASRStabilityAnalyzer
from commit_engine import CommitEngine
from entity_protector import EntityProtector
from session_manager import session_manager, ConversationSession, Participant
from whisper_asr import LocalWhisperASR
from indictrans2_translator import IndicTrans2Translator

# Load environment variables
env_path = backend_dir / "key.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "indictrans2").lower()
ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper").lower()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live-translator")

app = FastAPI(title="IndieTranslate — Two-Device Live Indic Translation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Generic Translation Helper ────────────────────────────────────────────────
async def translate_text(
    client: Optional[AsyncSarvamAI],
    text: str,
    source_lang: str,
    target_lang: str,
    backend: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Generic translation flow preserving contextual entities.
    translate(source_language, target_language, text)
    Routes to local IndicTrans2 when backend='indictrans2' or 'local',
    otherwise preserves default Sarvam translation pipeline.
    """
    cleaned = text.strip()
    if not cleaned or source_lang == target_lang:
        return cleaned, []

    protected_text, entity_map, detected = EntityProtector.protect(cleaned, source_lang)
    preserved_entities = [d["text"] for d in detected]

    selected_backend = (backend or TRANSLATION_BACKEND).lower()
    if selected_backend in ("indictrans2", "local") or not client:
        translator = IndicTrans2Translator.get_instance()
        translated_output = await translator.translate_async(
            protected_text, source_lang, target_lang
        )
    else:
        res = await client.text.translate(
            input=protected_text,
            source_language_code=source_lang,
            target_language_code=target_lang,
        )
        translated_output = res.translated_text

    restored_text = EntityProtector.restore(translated_output, entity_map)
    return restored_text, preserved_entities


def get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── REST Endpoints ─────────────────────────────────────────────────────────────
class CreateSessionRequest(BaseModel):
    language: Optional[str] = "en-IN"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr_backend": ASR_BACKEND,
        "translation_backend": TRANSLATION_BACKEND,
        "sarvam_configured": bool(SARVAM_API_KEY),
        "active_sessions": len(session_manager.sessions),
        "local_ip": get_local_ip(),
    }


@app.get("/api/info")
async def api_info():
    return {
        "local_ip": get_local_ip(),
        "frontend_port": 5173,
        "backend_port": 8000,
        "asr_backend": ASR_BACKEND,
        "translation_backend": TRANSLATION_BACKEND,
        "sarvam_configured": bool(SARVAM_API_KEY),
    }


@app.post("/api/session/create")
async def api_create_session(req: CreateSessionRequest):
    session = session_manager.create_session()
    return {
        "status": "created",
        "session_id": session.session_id,
        "language": req.language,
    }


@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        return {"exists": False, "session_id": session_id.upper()}
    return {
        "exists": True,
        "session_id": session.session_id,
        "participant_count": session.participant_count(),
        "is_full": session.is_full(),
        "language_a": session.participant_a.language if session.participant_a else None,
        "language_b": session.participant_b.language if session.participant_b else None,
    }


# ── Per-Participant ASR & Orchestration Worker ─────────────────────────────────
async def run_participant_asr(
    participant: Participant,
    session: ConversationSession,
    sarvam_client: AsyncSarvamAI,
):
    """
    Independent ASR and evidence-gated translation pipeline for one participant.
    """
    logger.info(
        f"ASR worker started for participant {participant.participant_id} "
        f"in session {session.session_id}"
    )
    try:
        while True:
            # Wait until participant starts listening
            await participant.listening_event.wait()
            current_lang = participant.language
            logger.info(
                f"Participant {participant.participant_id} connecting to "
                f"Sarvam Realtime STT ({current_lang})"
            )

            try:
                async with sarvam_client.speech_to_text_realtime_streaming.connect(
                    language_code=current_lang,
                    model="saaras:v3-realtime",
                    stream_type="fast",
                    encoding="linear16",
                    sample_rate="16000",
                    endpointing="vad",
                ) as sarvam_ws:
                    logger.info(
                        f"Sarvam STT connected for {participant.participant_id} ({current_lang})"
                    )

                    # Task 1: Forward PCM chunks from queue to Sarvam
                    async def tx_audio():
                        while participant.listening_event.is_set() and participant.language == current_lang:
                            try:
                                raw_pcm = await asyncio.wait_for(
                                    participant.audio_queue.get(), timeout=0.4
                                )
                                b64_audio = base64.b64encode(raw_pcm).decode("utf-8")
                                await sarvam_ws.send_realtime_audio_input(
                                    RealtimeAudioInput(audio=b64_audio)
                                )
                            except asyncio.TimeoutError:
                                continue
                            except Exception as ex:
                                logger.debug(f"Audio tx loop exited: {ex}")
                                break

                    # Task 2: Receive STT events from Sarvam and evaluate orchestration
                    async def rx_stt():
                        async for msg in sarvam_ws:
                            if (
                                not participant.listening_event.is_set()
                                or participant.language != current_lang
                            ):
                                break

                            if isinstance(msg, bytes):
                                continue

                            event_type = getattr(msg, "event", None)
                            if isinstance(msg, dict):
                                event_type = msg.get("event")

                            # Handle partial transcript
                            if (
                                isinstance(msg, RealtimeTranscriptPartial)
                                or event_type == "transcript.partial"
                            ):
                                text = (
                                    getattr(msg, "text", "")
                                    if hasattr(msg, "text")
                                    else msg.get("text", "")
                                )
                                utt_idx = (
                                    getattr(msg, "utterance_idx", 0)
                                    if hasattr(msg, "utterance_idx")
                                    else msg.get("utterance_idx", 0)
                                )
                                raw_lang = getattr(msg, "language", None) if hasattr(msg, "language") else msg.get("language")
                                lang = raw_lang or current_lang

                                # 1. Run ASR Stability Analysis
                                stability_info = participant.stability_analyzer.analyze(
                                    text=text, utterance_idx=utt_idx
                                )

                                # 2. Run Evidence-Gated Commit Engine
                                commit_eval = participant.commit_engine.evaluate_partial(
                                    current_text=text,
                                    stability_info=stability_info,
                                    utterance_idx=utt_idx,
                                    is_final_event=False,
                                )

                                decision = commit_eval["decision"]
                                newly_translated_text = ""
                                preserved_entities = []

                                peer = session.get_other_participant(participant.participant_id)

                                # Determine target language (peer's language or sensible default if waiting for peer)
                                if peer and peer.language:
                                    target_lang = peer.language
                                else:
                                    src_prefix = current_lang.split("-")[0].lower()
                                    target_lang = "ta-IN" if src_prefix in ("en", "hi") else "en-IN"

                                # 3. Incremental translation trigger: COMMIT is executed and routed
                                if decision == "COMMIT" and commit_eval.get(
                                    "segment_to_translate"
                                ):
                                    seg = commit_eval["segment_to_translate"]
                                    try:
                                        newly_translated_text, preserved_entities = await translate_text(
                                            sarvam_client, seg, current_lang, target_lang
                                        )
                                        participant.commit_engine.record_commit(
                                            seg, newly_translated_text
                                        )

                                        # Route translation commit to peer if connected
                                        if peer and newly_translated_text.strip():
                                            await peer.send_json({
                                                "type": "translation.commit",
                                                "speaker_id": participant.participant_id,
                                                "source_language": current_lang,
                                                "target_language": target_lang,
                                                "original": seg,
                                                "translation": newly_translated_text.strip(),
                                                "utterance_index": utt_idx,
                                                "decision": "COMMIT",
                                            })
                                    except Exception as e:
                                        logger.error(
                                            f"Commit translation error ({participant.participant_id}): {e}"
                                        )

                                # Send partial transcript to the SPEAKER
                                await participant.send_json({
                                    "type": "transcript.partial",
                                    "text": text,
                                    "utterance_idx": utt_idx,
                                    "language": lang,
                                    "stability": stability_info,
                                    "evidence_gate": {
                                        "decision": decision,
                                        "commit_score": commit_eval["commit_score"],
                                        "stability_score": commit_eval["stability_score"],
                                        "completeness_score": commit_eval["completeness_score"],
                                        "completeness_state": commit_eval.get("completeness_state"),
                                        "completeness_reason": commit_eval.get("completeness_reason"),
                                        "committed_source": participant.commit_engine.committed_source_text,
                                        "committed_translation": participant.commit_engine.committed_translation_text,
                                        "uncommitted_source": commit_eval.get("uncommitted_source", ""),
                                        "preserved_entities": preserved_entities,
                                    },
                                })

                            # Handle final transcript
                            elif (
                                isinstance(msg, RealtimeTranscriptFinal)
                                or event_type == "transcript.final"
                            ):
                                text = (
                                    getattr(msg, "text", "")
                                    if hasattr(msg, "text")
                                    else msg.get("text", "")
                                )
                                utt_idx = (
                                    getattr(msg, "utterance_idx", 0)
                                    if hasattr(msg, "utterance_idx")
                                    else msg.get("utterance_idx", 0)
                                )
                                raw_lang = getattr(msg, "language", None) if hasattr(msg, "language") else msg.get("language")
                                lang = raw_lang or current_lang
                                clean_text = text.strip() if text else ""

                                stability_info = participant.stability_analyzer.analyze(
                                    text=clean_text, utterance_idx=utt_idx
                                )
                                commit_eval = participant.commit_engine.evaluate_partial(
                                    current_text=clean_text,
                                    stability_info=stability_info,
                                    utterance_idx=utt_idx,
                                    is_final_event=True,
                                )

                                peer = session.get_other_participant(participant.participant_id)
                                translated_text = ""
                                preserved_entities = []
                                already_committed = False

                                # Determine target language
                                if peer and peer.language:
                                    target_lang = peer.language
                                else:
                                    src_prefix = current_lang.split("-")[0].lower()
                                    target_lang = "ta-IN" if src_prefix in ("en", "hi") else "en-IN"

                                if clean_text:
                                    if (
                                        participant.commit_engine.committed_source_text == clean_text
                                        and participant.commit_engine.committed_translation_text
                                    ):
                                        translated_text = (
                                            participant.commit_engine.committed_translation_text
                                        )
                                        already_committed = True
                                    else:
                                        try:
                                            translated_text, preserved_entities = await translate_text(
                                                sarvam_client, clean_text, current_lang, target_lang
                                            )
                                        except Exception as trans_err:
                                            logger.error(
                                                f"Final translation error ({participant.participant_id}): {trans_err}"
                                            )
                                            translated_text = f"[Translation failed: {trans_err}]"

                                if translated_text.strip() and not already_committed and peer:
                                    await peer.send_json({
                                        "type": "translation.commit",
                                        "speaker_id": participant.participant_id,
                                        "source_language": current_lang,
                                        "target_language": target_lang,
                                        "original": clean_text,
                                        "translation": translated_text.strip(),
                                        "utterance_index": utt_idx,
                                        "decision": "COMMIT",
                                    })

                                # Reset analyzers for next utterance
                                participant.stability_analyzer.reset(utt_idx + 1)
                                participant.commit_engine.reset_utterance(utt_idx + 1)

                                # Send final transcript to SPEAKER
                                await participant.send_json({
                                    "type": "transcript.final",
                                    "text": clean_text,
                                    "translated_text": translated_text,
                                    "source_language": current_lang,
                                    "target_language": target_lang,
                                    "utterance_idx": utt_idx,
                                    "language": lang,
                                    "preserved_entities": preserved_entities,
                                })

                    task_tx = asyncio.create_task(tx_audio())
                    task_rx = asyncio.create_task(rx_stt())

                    done, pending = await asyncio.wait(
                        [task_tx, task_rx], return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        if not t.cancelled() and t.exception():
                            logger.error(f"ASR task exception: {t.exception()}", exc_info=t.exception())

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Sarvam STT connection loop error: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        logger.info(f"ASR worker cancelled for participant {participant.participant_id}")
    except Exception as e:
        logger.error(f"Fatal error in ASR worker for {participant.participant_id}: {e}", exc_info=True)


async def run_participant_local_whisper_asr(
    participant: Participant,
    session: ConversationSession,
):
    """
    100% Offline Local ASR & Translation Worker using Faster-Whisper and IndicTrans2.
    Streams 16kHz PCM chunks, performs energy VAD, generates partial transcripts,
    runs Stability/Completeness analysis, and commits translations locally.
    """
    whisper_asr = LocalWhisperASR.get_instance()
    logger.info(
        f"Local Whisper ASR worker started for participant {participant.participant_id} "
        f"in session {session.session_id}"
    )

    try:
        while True:
            await participant.listening_event.wait()
            current_lang = participant.language
            utt_idx = 0
            audio_buffer = bytearray()
            last_partial_time = time.time()
            last_speech_time = time.time()
            is_speaking = False
            last_transcribed_text = ""

            logger.info(
                f"Participant {participant.participant_id} listening with Local Whisper ASR ({current_lang})"
            )

            while participant.listening_event.is_set() and participant.language == current_lang:
                try:
                    raw_pcm = await asyncio.wait_for(
                        participant.audio_queue.get(), timeout=0.1
                    )
                    audio_buffer.extend(raw_pcm)

                    # Compute RMS to detect voice activity
                    if len(raw_pcm) >= 64:
                        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(samples ** 2)))
                    else:
                        rms = 0.0

                    now = time.time()
                    if rms > 200.0:  # Speech detected
                        last_speech_time = now
                        is_speaking = True

                    # Partial transcription trigger: if speaking and >=0.4s audio buffer
                    if is_speaking and len(audio_buffer) >= 16000 * 2 * 0.4 and (now - last_partial_time >= 0.45):
                        last_partial_time = now
                        text, conf = await whisper_asr.transcribe_pcm_async(
                            bytes(audio_buffer), current_lang
                        )
                        clean_text = text.strip()
                        if clean_text and clean_text != last_transcribed_text:
                            last_transcribed_text = clean_text

                            stability_info = participant.stability_analyzer.analyze(
                                text=clean_text, utterance_idx=utt_idx
                            )
                            commit_eval = participant.commit_engine.evaluate_partial(
                                current_text=clean_text,
                                stability_info=stability_info,
                                utterance_idx=utt_idx,
                                is_final_event=False,
                            )
                            decision = commit_eval["decision"]
                            peer = session.get_other_participant(participant.participant_id)
                            target_lang = peer.language if (peer and peer.language) else ("hi-IN" if current_lang.startswith("ta") else "ta-IN")

                            preview_trans, _ = await translate_text(
                                None, clean_text, current_lang, target_lang, backend="indictrans2"
                            )

                            if decision == "COMMIT" and commit_eval.get("segment_to_translate"):
                                seg = commit_eval["segment_to_translate"]
                                newly_trans, _ = await translate_text(
                                    None, seg, current_lang, target_lang, backend="indictrans2"
                                )
                                participant.commit_engine.record_commit(seg, newly_trans)
                                if peer and newly_trans.strip():
                                    await peer.send_json({
                                        "type": "translation.commit",
                                        "speaker_id": participant.participant_id,
                                        "source_language": current_lang,
                                        "target_language": target_lang,
                                        "original": seg,
                                        "translation": newly_trans.strip(),
                                        "utterance_index": utt_idx,
                                        "decision": "COMMIT",
                                    })

                            # Send partial transcript to SPEAKER
                            await participant.send_json({
                                "type": "transcript.partial",
                                "text": clean_text,
                                "preview_translation": preview_trans,
                                "utterance_idx": utt_idx,
                                "language": current_lang,
                                "stability": stability_info,
                                "evidence_gate": {
                                    "decision": decision,
                                    "commit_score": commit_eval["commit_score"],
                                    "stability_score": commit_eval["stability_score"],
                                    "completeness_score": commit_eval["completeness_score"],
                                    "completeness_state": commit_eval.get("completeness_state"),
                                    "completeness_reason": commit_eval.get("completeness_reason"),
                                    "committed_source": participant.commit_engine.committed_source_text,
                                    "committed_translation": participant.commit_engine.committed_translation_text,
                                    "uncommitted_source": commit_eval.get("uncommitted_source", ""),
                                    "preserved_entities": [],
                                },
                            })

                    # Silence / Endpointing: if speaker was speaking and has paused for >=0.75s
                    if is_speaking and (now - last_speech_time >= 0.75) and len(audio_buffer) >= 16000 * 2 * 0.4:
                        final_text, _ = await whisper_asr.transcribe_pcm_async(
                            bytes(audio_buffer), current_lang
                        )
                        clean_final = final_text.strip()
                        if clean_final:
                            peer = session.get_other_participant(participant.participant_id)
                            target_lang = peer.language if (peer and peer.language) else ("hi-IN" if current_lang.startswith("ta") else "ta-IN")

                            stability_info = participant.stability_analyzer.analyze(
                                text=clean_final, utterance_idx=utt_idx
                            )
                            commit_eval = participant.commit_engine.evaluate_partial(
                                current_text=clean_final,
                                stability_info=stability_info,
                                utterance_idx=utt_idx,
                                is_final_event=True,
                            )

                            translated_final, _ = await translate_text(
                                None, clean_final, current_lang, target_lang, backend="indictrans2"
                            )

                            if translated_final.strip() and peer:
                                await peer.send_json({
                                    "type": "translation.commit",
                                    "speaker_id": participant.participant_id,
                                    "source_language": current_lang,
                                    "target_language": target_lang,
                                    "original": clean_final,
                                    "translation": translated_final.strip(),
                                    "utterance_index": utt_idx,
                                    "decision": "COMMIT",
                                })

                            participant.stability_analyzer.reset(utt_idx + 1)
                            participant.commit_engine.reset_utterance(utt_idx + 1)

                            await participant.send_json({
                                "type": "transcript.final",
                                "text": clean_final,
                                "translated_text": translated_final,
                                "source_language": current_lang,
                                "target_language": target_lang,
                                "utterance_idx": utt_idx,
                                "language": current_lang,
                                "preserved_entities": [],
                            })
                            utt_idx += 1

                        audio_buffer.clear()
                        is_speaking = False
                        last_transcribed_text = ""

                except asyncio.TimeoutError:
                    now = time.time()
                    if is_speaking and (now - last_speech_time >= 0.75) and len(audio_buffer) >= 16000 * 2 * 0.4:
                        final_text, _ = await whisper_asr.transcribe_pcm_async(
                            bytes(audio_buffer), current_lang
                        )
                        clean_final = final_text.strip()
                        if clean_final:
                            peer = session.get_other_participant(participant.participant_id)
                            target_lang = peer.language if (peer and peer.language) else ("hi-IN" if current_lang.startswith("ta") else "ta-IN")

                            translated_final, _ = await translate_text(
                                None, clean_final, current_lang, target_lang, backend="indictrans2"
                            )

                            if translated_final.strip() and peer:
                                await peer.send_json({
                                    "type": "translation.commit",
                                    "speaker_id": participant.participant_id,
                                    "source_language": current_lang,
                                    "target_language": target_lang,
                                    "original": clean_final,
                                    "translation": translated_final.strip(),
                                    "utterance_index": utt_idx,
                                    "decision": "COMMIT",
                                })

                            participant.stability_analyzer.reset(utt_idx + 1)
                            participant.commit_engine.reset_utterance(utt_idx + 1)

                            await participant.send_json({
                                "type": "transcript.final",
                                "text": clean_final,
                                "translated_text": translated_final,
                                "source_language": current_lang,
                                "target_language": target_lang,
                                "utterance_idx": utt_idx,
                                "language": current_lang,
                                "preserved_entities": [],
                            })
                            utt_idx += 1

                        audio_buffer.clear()
                        is_speaking = False
                        last_transcribed_text = ""
                    continue
                except Exception as ex:
                    logger.error(f"Local Whisper ASR loop error: {ex}", exc_info=True)
                    await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logger.info(f"Local Whisper ASR worker cancelled for participant {participant.participant_id}")
    except Exception as e:
        logger.error(f"Fatal error in Local Whisper ASR worker for {participant.participant_id}: {e}", exc_info=True)


# ── Legacy Single-Device Flow Helper (Backward Compatibility) ──────────────────
async def handle_legacy_single_device(
    websocket: WebSocket,
    init_data: dict,
    sarvam_client: AsyncSarvamAI,
):
    source_lang = init_data.get("source_language_code") or init_data.get("language_code", "ta-IN")
    target_lang = init_data.get("target_language_code")
    if not target_lang:
        target_lang = "ta-IN" if source_lang == "en-IN" else "en-IN"

    stability_analyzer = ASRStabilityAnalyzer()
    commit_engine = CommitEngine(language_code=source_lang)

    async with sarvam_client.speech_to_text_realtime_streaming.connect(
        language_code=source_lang,
        model="saaras:v3-realtime",
        stream_type="fast",
        encoding="linear16",
        sample_rate="16000",
        endpointing="vad",
    ) as sarvam_ws:
        await websocket.send_json({
            "type": "ready",
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "language_code": source_lang,
        })

        async def fwd_sarvam():
            async for msg in sarvam_ws:
                if isinstance(msg, bytes):
                    continue
                event_type = getattr(msg, "event", None)
                if isinstance(msg, dict):
                    event_type = msg.get("event")

                if isinstance(msg, RealtimeTranscriptPartial) or event_type == "transcript.partial":
                    text = getattr(msg, "text", "") if hasattr(msg, "text") else msg.get("text", "")
                    utterance_idx = getattr(msg, "utterance_idx", 0) if hasattr(msg, "utterance_idx") else msg.get("utterance_idx", 0)
                    lang = getattr(msg, "language", source_lang) if hasattr(msg, "language") else msg.get("language", source_lang)

                    stability_info = stability_analyzer.analyze(text=text, utterance_idx=utterance_idx)
                    commit_eval = commit_engine.evaluate_partial(
                        current_text=text,
                        stability_info=stability_info,
                        utterance_idx=utterance_idx,
                        is_final_event=False,
                    )
                    decision = commit_eval["decision"]
                    newly_translated_text = ""
                    preserved_entities = []

                    if decision in ("COMMIT", "REVISE") and commit_eval.get("segment_to_translate"):
                        seg = commit_eval["segment_to_translate"]
                        try:
                            newly_translated_text, preserved_entities = await translate_text(
                                sarvam_client, seg, source_lang, target_lang
                            )
                            commit_engine.record_commit(seg, newly_translated_text)
                            if newly_translated_text.strip():
                                await websocket.send_json({
                                    "type": "translation.commit",
                                    "source": seg,
                                    "translation": newly_translated_text.strip(),
                                    "target_language": target_lang,
                                    "source_language": source_lang,
                                    "decision": decision,
                                    "utterance_idx": utterance_idx,
                                })
                        except Exception as e:
                            logger.error(f"Single device commit error: {e}")

                    await websocket.send_json({
                        "type": "transcript.partial",
                        "text": text,
                        "utterance_idx": utterance_idx,
                        "language": lang,
                        "stability": stability_info,
                        "evidence_gate": {
                            "decision": decision,
                            "commit_score": commit_eval["commit_score"],
                            "stability_score": commit_eval["stability_score"],
                            "completeness_score": commit_eval["completeness_score"],
                            "completeness_state": commit_eval.get("completeness_state"),
                            "completeness_reason": commit_eval.get("completeness_reason"),
                            "committed_source": commit_engine.committed_source_text,
                            "committed_translation": commit_engine.committed_translation_text,
                            "uncommitted_source": commit_eval.get("uncommitted_source", ""),
                            "preserved_entities": preserved_entities,
                        },
                    })

                elif isinstance(msg, RealtimeTranscriptFinal) or event_type == "transcript.final":
                    text = getattr(msg, "text", "") if hasattr(msg, "text") else msg.get("text", "")
                    utterance_idx = getattr(msg, "utterance_idx", 0) if hasattr(msg, "utterance_idx") else msg.get("utterance_idx", 0)
                    lang = getattr(msg, "language", source_lang) if hasattr(msg, "language") else msg.get("language", source_lang)
                    clean_text = text.strip() if text else ""

                    stability_info = stability_analyzer.analyze(text=clean_text, utterance_idx=utterance_idx)
                    commit_eval = commit_engine.evaluate_partial(
                        current_text=clean_text,
                        stability_info=stability_info,
                        utterance_idx=utterance_idx,
                        is_final_event=True,
                    )
                    translated_text = ""
                    preserved_entities = []
                    already_committed = False

                    if clean_text and source_lang != target_lang:
                        if (
                            commit_engine.committed_source_text == clean_text
                            and commit_engine.committed_translation_text
                        ):
                            translated_text = commit_engine.committed_translation_text
                            already_committed = True
                        else:
                            try:
                                translated_text, preserved_entities = await translate_text(
                                    sarvam_client, clean_text, source_lang, target_lang
                                )
                            except Exception as trans_err:
                                logger.error(f"Single device final translation error: {trans_err}")
                                translated_text = f"[Translation failed: {trans_err}]"

                    if translated_text.strip() and not already_committed:
                        await websocket.send_json({
                            "type": "translation.commit",
                            "source": clean_text,
                            "translation": translated_text.strip(),
                            "target_language": target_lang,
                            "source_language": source_lang,
                            "decision": "FINAL",
                            "utterance_idx": utterance_idx,
                        })

                    stability_analyzer.reset(utterance_idx + 1)
                    commit_engine.reset_utterance(utterance_idx + 1)

                    await websocket.send_json({
                        "type": "transcript.final",
                        "text": clean_text,
                        "translated_text": translated_text,
                        "source_language": source_lang,
                        "target_language": target_lang,
                        "utterance_idx": utterance_idx,
                        "language": lang,
                        "preserved_entities": preserved_entities,
                    })

        async def fwd_client():
            while True:
                data = await websocket.receive()
                if "bytes" in data and data["bytes"]:
                    b64_audio = base64.b64encode(data["bytes"]).decode("utf-8")
                    await sarvam_ws.send_realtime_audio_input(RealtimeAudioInput(audio=b64_audio))
                elif "text" in data and data["text"]:
                    msg_json = json.loads(data["text"])
                    if msg_json.get("type") == "stop" or msg_json.get("action") == "stop":
                        break

        t_rx = asyncio.create_task(fwd_sarvam())
        t_tx = asyncio.create_task(fwd_client())
        done, pending = await asyncio.wait([t_rx, t_tx], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()


# ── Main WebSocket Endpoint ────────────────────────────────────────────────────
@app.websocket("/ws/transcribe")
@app.websocket("/ws/session")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")

    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY) if SARVAM_API_KEY else None
    session: Optional[ConversationSession] = None
    participant: Optional[Participant] = None

    try:
        # First message determines mode (Session-based or Legacy Single-Device)
        init_raw = await websocket.receive_text()
        init_data = json.loads(init_raw)

        action = init_data.get("action") or init_data.get("type")

        # ── Backward Compatibility: Legacy Single-Device Mode ──
        if action not in ("create_session", "join_session"):
            if "source_language_code" in init_data or "language_code" in init_data:
                logger.info("Running in legacy single-device mode")
                await handle_legacy_single_device(websocket, init_data, sarvam_client)
                return
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid initial message. Expected create_session or join_session.",
                })
                await websocket.close()
                return

        # ── Two-Device Session Mode ──
        participant_id = f"p_{secrets.token_hex(4)}"
        selected_language = init_data.get("language") or init_data.get("language_code", "en-IN")

        if action == "create_session":
            session = session_manager.create_session()
            participant = session.add_participant(
                participant_id=participant_id,
                websocket=websocket,
                language=selected_language,
            )
            logger.info(
                f"Session {session.session_id} created by {participant_id} ({selected_language})"
            )
            await participant.send_json({
                "type": "session.created",
                "session_id": session.session_id,
                "participant_id": participant_id,
                "language": selected_language,
            })

        elif action == "join_session":
            code = init_data.get("session_id", "").strip().upper()
            target_session = session_manager.get_session(code)
            if not target_session:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Session '{code}' not found.",
                })
                await websocket.close()
                return

            if target_session.is_full():
                await websocket.send_json({
                    "type": "error",
                    "message": f"Session '{code}' already has two participants.",
                })
                await websocket.close()
                return

            session = target_session
            participant = session.add_participant(
                participant_id=participant_id,
                websocket=websocket,
                language=selected_language,
            )
            peer = session.get_other_participant(participant_id)

            logger.info(
                f"Participant {participant_id} ({selected_language}) joined session {session.session_id}"
            )
            await participant.send_json({
                "type": "session.joined",
                "session_id": session.session_id,
                "participant_id": participant_id,
                "own_language": selected_language,
                "other_language": peer.language if peer else None,
                "peer_connected": peer is not None,
            })

            # Notify the other participant that peer has connected
            if peer:
                await peer.send_json({
                    "type": "peer.connected",
                    "participant_id": participant_id,
                    "language": selected_language,
                })

        # Start background independent ASR worker for this participant
        if ASR_BACKEND in ("whisper", "local") or not sarvam_client:
            participant.asr_task = asyncio.create_task(
                run_participant_local_whisper_asr(participant, session)
            )
        else:
            participant.asr_task = asyncio.create_task(
                run_participant_asr(participant, session, sarvam_client)
            )

        # ── Main Event & Audio Loop for this Participant ──
        while True:
            data = await websocket.receive()

            # Binary audio PCM frame from microphone
            if "bytes" in data and data["bytes"]:
                if participant.is_listening:
                    await participant.audio_queue.put(data["bytes"])

            # JSON Control message
            elif "text" in data and data["text"]:
                msg_json = json.loads(data["text"])
                msg_action = msg_json.get("action") or msg_json.get("type")

                if msg_action == "start_listening":
                    participant.listening_event.set()
                    participant.is_listening = True
                    await participant.send_json({
                        "type": "listening.started",
                        "language": participant.language,
                    })

                elif msg_action in ("stop_listening", "stop"):
                    participant.listening_event.clear()
                    participant.is_listening = False
                    # Drain queue
                    while not participant.audio_queue.empty():
                        try:
                            participant.audio_queue.get_nowait()
                        except Exception:
                            break
                    await participant.send_json({"type": "listening.stopped"})

                elif msg_action == "change_language":
                    new_lang = msg_json.get("language")
                    if new_lang and new_lang != participant.language:
                        logger.info(
                            f"Participant {participant.participant_id} changed language "
                            f"from {participant.language} to {new_lang}"
                        )
                        participant.update_language(new_lang)

                        # Notify peer of the language change
                        peer = session.get_other_participant(participant.participant_id)
                        if peer:
                            await peer.send_json({
                                "type": "peer.language_changed",
                                "participant_id": participant.participant_id,
                                "language": new_lang,
                            })

                        await participant.send_json({
                            "type": "language.updated",
                            "language": new_lang,
                        })

                elif msg_action == "leave_session":
                    logger.info(
                        f"Participant {participant.participant_id} requested to leave session"
                    )
                    break

    except (WebSocketDisconnect, Exception) as e:
        if isinstance(e, WebSocketDisconnect) or "disconnect" in str(e).lower():
            logger.info("Participant WebSocket disconnected cleanly")
        else:
            logger.error(f"Error in WebSocket session handler: {e}", exc_info=True)
    finally:
        # ── Cleanup on Disconnect ──
        if participant and session:
            if participant.asr_task and not participant.asr_task.done():
                participant.asr_task.cancel()

            session.remove_participant(participant.participant_id)
            logger.info(
                f"Removed participant {participant.participant_id} from session {session.session_id}"
            )

            # Notify remaining peer
            peer = session.get_other_participant(participant.participant_id)
            if peer:
                await peer.send_json({
                    "type": "peer.disconnected",
                    "participant_id": participant.participant_id,
                })

            # If both left, prune session from memory
            if session.is_empty():
                session_manager.remove_session(session.session_id)
