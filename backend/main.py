"""
backend/main.py

HackNex 2026 EPS03 — Live Translation for Indic Languages

Pipeline:
  Browser PCM Audio (16-bit, 16kHz, mono)
  → WebSocket
  → Local faster-whisper ASR (loaded once, CUDA runtime detection / CPU fallback)
  → ASR Stability Analyzer
  → Evidence-Gated Commit Engine
  → Context-Aware Entity Protector
  → Sarvam Text Translation (client.text.translate)
  → WebSocket: translation.commit / transcript.partial / transcript.final
  → Browser Web Speech API (TTS)
"""

import os
import json
import asyncio
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sarvamai.client import AsyncSarvamAI

from stability_analyzer import ASRStabilityAnalyzer
from commit_engine import CommitEngine
from entity_protector import EntityProtector
import whisper_asr

# Load environment variables
env_path = Path(__file__).resolve().parent / "key.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("live-translator")

app = FastAPI(title="Live Translation for Indic Languages")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: load faster-whisper model ONCE ───────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Loading local faster-whisper model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, whisper_asr.load_whisper_model)
    info = whisper_asr.get_model_info()
    logger.info(
        f"Whisper initialized: model={info['model']}, device={info['device']}, compute_type={info['compute_type']}"
    )


# ── Health endpoint ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    model_info = whisper_asr.get_model_info()
    return {
        "status": "ok",
        "sarvam_configured": bool(SARVAM_API_KEY),
        "asr_engine": "local-whisper",
        "whisper_model": model_info["model"],
        "whisper_device": model_info["device"],
        "compute_type": model_info["compute_type"],
    }


# ── WebSocket streaming endpoint ──────────────────────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /ws/transcribe")

    if not SARVAM_API_KEY:
        await websocket.send_json({
            "type": "error",
            "message": "SARVAM_API_KEY not configured on backend for translation.",
        })
        await websocket.close()
        return

    sarvam_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    stability_analyzer = ASRStabilityAnalyzer()

    try:
        # 1. Receive language configuration message
        init_msg = await websocket.receive_text()
        init_data = json.loads(init_msg)
        source_lang = (
            init_data.get("source_language_code")
            or init_data.get("language_code", "ta-IN")
        )
        target_lang = init_data.get("target_language_code")
        if not target_lang:
            target_lang = "ta-IN" if source_lang == "en-IN" else "en-IN"

        commit_engine = CommitEngine(language_code=source_lang)
        utterance_idx = 0

        # Initialize local Whisper streaming session
        asr_session = whisper_asr.WhisperStreamingSession(
            language_code=source_lang,
            utterance_idx=utterance_idx,
        )

        model_info = whisper_asr.get_model_info()
        logger.info(
            f"Session started: {source_lang} -> {target_lang} (Whisper lang={asr_session.whisper_lang}, model={model_info['model']} on {model_info['device']})"
        )

        await websocket.send_json({
            "type": "ready",
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "language_code": source_lang,
            "asr_engine": "local-whisper",
            "whisper_model": model_info["model"],
            "whisper_device": model_info["device"],
        })

        finalize_flag = asyncio.Event()
        loop = asyncio.get_event_loop()

        # ── Pipeline Handlers ──────────────────────────────────────────────────
        async def handle_partial(text: str, utt_idx: int, t_window_start: Optional[float]):
            """
            Process Whisper partial hypothesis:
            1. Stability Analyzer
            2. Evidence-Gated Commit Engine
            3. Context-Aware Entity Protection & Sarvam Translation (if COMMIT/REVISE)
            4. Dispatch websocket events
            """
            if not text:
                return

            t_whisper_done = time.perf_counter()
            asr_lat_ms = (t_whisper_done - t_window_start) * 1000 if t_window_start else 0.0

            # 1. Stability Analyzer
            stability_info = stability_analyzer.analyze(text=text, utterance_idx=utt_idx)

            # 2. Evidence-Gated Commit Engine
            commit_eval = commit_engine.evaluate_partial(
                current_text=text,
                stability_info=stability_info,
                utterance_idx=utt_idx,
                is_final_event=False,
            )

            decision = commit_eval["decision"]
            newly_translated_text = ""
            preserved_entities = []

            # 3. Only COMMIT or REVISE triggers incremental translation
            if decision in ("COMMIT", "REVISE") and commit_eval.get("segment_to_translate"):
                seg = commit_eval["segment_to_translate"]
                t_trans_start = time.perf_counter()
                try:
                    # Context-Aware Semantic Protection
                    protected_seg, entity_map, detected = EntityProtector.protect(seg, source_lang)
                    preserved_entities = [d["text"] for d in detected]

                    # Sarvam Text Translation
                    res = await sarvam_client.text.translate(
                        input=protected_seg,
                        source_language_code=source_lang,
                        target_language_code=target_lang,
                    )
                    # Restore protected entities
                    newly_translated_text = EntityProtector.restore(res.translated_text, entity_map)
                    commit_engine.record_commit(seg, newly_translated_text)

                    t_trans_end = time.perf_counter()
                    trans_lat_ms = (t_trans_end - t_trans_start) * 1000
                    total_lat_ms = (t_trans_end - t_window_start) * 1000 if t_window_start else trans_lat_ms

                    logger.info(
                        f"[LATENCY] [{decision}] ASR={asr_lat_ms:.0f}ms | Translate={trans_lat_ms:.0f}ms | Total={total_lat_ms:.0f}ms | '{seg}' -> '{newly_translated_text}'"
                    )

                    if newly_translated_text.strip():
                        await websocket.send_json({
                            "type": "translation.commit",
                            "source": seg,
                            "translation": newly_translated_text.strip(),
                            "target_language": target_lang,
                            "source_language": source_lang,
                            "decision": decision,
                            "utterance_idx": utt_idx,
                        })
                except Exception as e:
                    logger.error(f"Early commit/revision translation error: {e}")

            await websocket.send_json({
                "type": "transcript.partial",
                "text": text,
                "utterance_idx": utt_idx,
                "language": source_lang,
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

        async def handle_final(text: str, utt_idx: int, t_window_start: Optional[float]):
            """
            Process final utterance text, translate any uncommitted text, and emit transcript.final.
            """
            clean_text = text.strip()
            t_whisper_done = time.perf_counter()
            asr_lat_ms = (t_whisper_done - t_window_start) * 1000 if t_window_start else 0.0

            stability_info = stability_analyzer.analyze(text=clean_text, utterance_idx=utt_idx)
            commit_eval = commit_engine.evaluate_partial(
                current_text=clean_text,
                stability_info=stability_info,
                utterance_idx=utt_idx,
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
                    t_trans_start = time.perf_counter()
                    try:
                        protected_clean, entity_map, detected = EntityProtector.protect(
                            clean_text, source_lang
                        )
                        preserved_entities = [d["text"] for d in detected]
                        translation_res = await sarvam_client.text.translate(
                            input=protected_clean,
                            source_language_code=source_lang,
                            target_language_code=target_lang,
                        )
                        translated_text = EntityProtector.restore(
                            translation_res.translated_text, entity_map
                        )
                        t_trans_end = time.perf_counter()
                        trans_lat_ms = (t_trans_end - t_trans_start) * 1000
                        total_lat_ms = (t_trans_end - t_window_start) * 1000 if t_window_start else trans_lat_ms
                        logger.info(
                            f"[LATENCY] [FINAL] ASR={asr_lat_ms:.0f}ms | Translate={trans_lat_ms:.0f}ms | Total={total_lat_ms:.0f}ms | '{clean_text}' -> '{translated_text}'"
                        )
                    except Exception as trans_err:
                        logger.error(f"Final translation error: {trans_err}")
                        translated_text = f"[Translation failed: {trans_err}]"

            if translated_text.strip() and not already_committed:
                await websocket.send_json({
                    "type": "translation.commit",
                    "source": clean_text,
                    "translation": translated_text.strip(),
                    "target_language": target_lang,
                    "source_language": source_lang,
                    "decision": "FINAL",
                    "utterance_idx": utt_idx,
                })

            stability_analyzer.reset(utt_idx + 1)
            commit_engine.reset_utterance(utt_idx + 1)

            await websocket.send_json({
                "type": "transcript.final",
                "text": clean_text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "utterance_idx": utt_idx,
                "language": source_lang,
                "preserved_entities": preserved_entities,
            })

        # ── Concurrent Tasks: Audio Receiver & ASR Inference Worker ───────────
        stop_received = False

        async def rx_audio_task():
            nonlocal stop_received
            try:
                while True:
                    data = await websocket.receive()
                    if "bytes" in data and data["bytes"]:
                        _, should_finalize = asr_session.feed(data["bytes"])
                        if should_finalize:
                            finalize_flag.set()
                    elif "text" in data and data["text"]:
                        msg_json = json.loads(data["text"])
                        if msg_json.get("type") == "stop":
                            logger.info("Client requested stop")
                            stop_received = True
                            finalize_flag.set()
                            break
            except WebSocketDisconnect:
                logger.info("Client WebSocket disconnected in rx task")
                stop_received = True
                finalize_flag.set()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in rx_audio_task: {e}")

        async def asr_worker_task():
            nonlocal utterance_idx
            try:
                while not stop_received:
                    # Check for endpoint finalization
                    if finalize_flag.is_set():
                        finalize_flag.clear()
                        if asr_session.total_samples > 0 and asr_session.speech_started:
                            t_start = asr_session.window_start_time
                            audio_snap = asr_session.get_audio_snapshot(max_seconds=20.0)
                            final_text, _ = await loop.run_in_executor(
                                None,
                                whisper_asr.transcribe_audio_array,
                                audio_snap,
                                asr_session.whisper_lang,
                            )
                            if final_text.strip():
                                await handle_final(final_text, utterance_idx, t_start)
                                utterance_idx += 1
                                asr_session.reset(utterance_idx)
                        continue

                    # Check for partial update
                    if asr_session.should_transcribe_partial():
                        asr_session.is_transcribing = True
                        t_start = asr_session.window_start_time
                        audio_snap = asr_session.get_audio_snapshot(max_seconds=15.0)

                        hypothesis, _ = await loop.run_in_executor(
                            None,
                            whisper_asr.transcribe_audio_array,
                            audio_snap,
                            asr_session.whisper_lang,
                        )
                        asr_session.is_transcribing = False
                        is_new = (hypothesis != asr_session.last_hypothesis) and (hypothesis != "")
                        asr_session.record_partial_done(hypothesis)

                        if is_new:
                            await handle_partial(hypothesis, utterance_idx, t_start)

                    await asyncio.sleep(0.06)

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in asr_worker_task: {e}")

        task_rx = asyncio.create_task(rx_audio_task())
        task_asr = asyncio.create_task(asr_worker_task())

        done, pending = await asyncio.wait(
            [task_rx, task_asr],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for t in pending:
            t.cancel()

        # Teardown: flush remaining audio if needed
        if asr_session.total_samples > 0 and asr_session.speech_started:
            t_start = asr_session.window_start_time
            audio_snap = asr_session.get_audio_snapshot(max_seconds=20.0)
            final_text, _ = await loop.run_in_executor(
                None,
                whisper_asr.transcribe_audio_array,
                audio_snap,
                asr_session.whisper_lang,
            )
            if final_text.strip():
                await handle_final(final_text, utterance_idx, t_start)

    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected cleanly")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.info("Transcribe session closed")
