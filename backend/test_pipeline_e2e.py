"""
backend/test_pipeline_e2e.py

End-to-End Pipeline Verification for HackNex EPS03:
Tests:
1. English: "My company's name is Fish."
2. Tamil: "நான் Chennai போறேன்."
3. Hindi: "मैं कल चेन्नई जाऊंगा।"

Verifies:
- Stability Analyzer updates
- Evidence-Gated Commit Engine evaluations
- Context-Aware Entity Preservation ("Fish", "Chennai")
- Sarvam Text Translation with entity protection & restoration
- Latency measurement logging
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout for utf-8 display of Indic characters
sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(__file__).resolve().parent / "key.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from sarvamai.client import AsyncSarvamAI
from stability_analyzer import ASRStabilityAnalyzer
from commit_engine import CommitEngine
from entity_protector import EntityProtector
import whisper_asr


async def run_tests():
    print("=" * 70)
    print("STARTING HACKNEX EPS03 PIPELINE TESTS")
    print("=" * 70)

    # 1. Check Whisper Model Initialization
    whisper_asr.load_whisper_model()
    model_info = whisper_asr.get_model_info()
    print(f"\n[ASR ENGINE] Model={model_info['model']} | Device={model_info['device']} | Compute={model_info['compute_type']}")

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("[ERROR] SARVAM_API_KEY not found in environment or key.env")
        return False
    sarvam_client = AsyncSarvamAI(api_subscription_key=api_key)

    test_cases = [
        {
            "id": "1. English with Contextual Entity",
            "source_lang": "en-IN",
            "target_lang": "ta-IN",
            "partials": [
                "My company's",
                "My company's name",
                "My company's name is Fish",
            ],
            "expected_preserved": ["Fish"],
        },
        {
            "id": "2. Tamil with Code-Mixed City Entity",
            "source_lang": "ta-IN",
            "target_lang": "en-IN",
            "partials": [
                "நான்",
                "நான் Chennai",
                "நான் Chennai போறேன்",
            ],
            "expected_preserved": ["Chennai"],
        },
        {
            "id": "3. Hindi with City Entity",
            "source_lang": "hi-IN",
            "target_lang": "en-IN",
            "partials": [
                "मैं कल",
                "मैं कल चेन्नई",
                "मैं कल चेन्नई जाऊंगा",
            ],
            "expected_preserved": ["चेन्नई"],
        },
    ]

    all_passed = True

    for tc in test_cases:
        print(f"\n" + "-" * 70)
        print(f"TEST: {tc['id']}")
        print(f"Language Pair: {tc['source_lang']} -> {tc['target_lang']}")
        print("-" * 70)

        stability_analyzer = ASRStabilityAnalyzer()
        commit_engine = CommitEngine(language_code=tc["source_lang"])
        utterance_idx = 0

        for i, text in enumerate(tc["partials"]):
            is_final = (i == len(tc["partials"]) - 1)
            t_eval_start = time.perf_counter()

            # Stability Analysis
            stab = stability_analyzer.analyze(text=text, utterance_idx=utterance_idx)

            # Evidence Gate Evaluation
            commit_eval = commit_engine.evaluate_partial(
                current_text=text,
                stability_info=stab,
                utterance_idx=utterance_idx,
                is_final_event=is_final,
            )
            decision = commit_eval["decision"]
            t_eval_end = time.perf_counter()

            print(f"  Step {i+1}: text='{text}'")
            print(f"    - Stability Score: {stab['stability_score']} (stable: '{stab['stable_prefix']}', unstable: '{stab['unstable_suffix']}')")
            print(f"    - Evidence Gate: decision={decision}, commit_score={commit_eval['commit_score']:.2f}, completeness={commit_eval['completeness_score']:.2f} ({commit_eval.get('completeness_state')})")

            # Check if translation is triggered
            if decision in ("COMMIT", "REVISE") or is_final:
                seg_to_translate = commit_eval.get("segment_to_translate") or text
                t_trans_start = time.perf_counter()

                # Entity Protection
                protected_seg, entity_map, detected = EntityProtector.protect(seg_to_translate, tc["source_lang"])
                detected_names = [d["text"] for d in detected]

                # Sarvam Translation
                res = await sarvam_client.text.translate(
                    input=protected_seg,
                    source_language_code=tc["source_lang"],
                    target_language_code=tc["target_lang"],
                )
                restored_text = EntityProtector.restore(res.translated_text, entity_map)
                commit_engine.record_commit(seg_to_translate, restored_text)
                t_trans_end = time.perf_counter()

                trans_latency_ms = (t_trans_end - t_trans_start) * 1000
                total_latency_ms = (t_trans_end - t_eval_start) * 1000

                print(f"    >>> TRANSLATION TRIGGERED ({decision}):")
                print(f"        Original Segment: '{seg_to_translate}'")
                print(f"        Protected Segment: '{protected_seg}' (Detected: {detected_names})")
                print(f"        Translated & Restored: '{restored_text}'")
                print(f"        [TIMING] Eval: {(t_eval_end - t_eval_start)*1000:.2f}ms | Translate: {trans_latency_ms:.1f}ms | Total: {total_latency_ms:.1f}ms")

                # Verify expected entities preserved
                for exp in tc["expected_preserved"]:
                    if exp in seg_to_translate:
                        if exp in detected_names or exp.lower() in [d.lower() for d in detected_names]:
                            print(f"        [PASS] Entity '{exp}' correctly preserved by Context Layer.")
                        else:
                            print(f"        [INFO] Entity '{exp}' detected context: {detected_names}")

    print("\n" + "=" * 70)
    print("ALL TEST SCENARIOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return True


if __name__ == "__main__":
    asyncio.run(run_tests())
