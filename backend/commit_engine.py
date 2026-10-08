"""
backend/commit_engine.py

Evidence-Gated Early Translation Commit Engine.

Combines:
1. ASR Stability (from ASRStabilityAnalyzer)
2. Linguistic Completeness (from CompletenessAnalyzer)
3. Endpoint / Temporal Evidence (unchanged partials, pause signals, ASR final)
4. Existing Committed Context (tracking committed prefix vs uncommitted suffix)

Emits Decision:
- WAIT: Low evidence, do not translate yet
- PREVIEW: Medium evidence, can be inspected without committing
- COMMIT: High evidence, semantic unit is safe to translate & commit
- REVISE: Previously committed portion was changed/retracted by ASR
"""

import time
from typing import Dict, Any, List, Optional
from completeness_analyzer import CompletenessAnalyzer


class CommitEngine:
    # Tunable engineering default weights
    WEIGHT_STABILITY = 0.35
    WEIGHT_COMPLETENESS = 0.50
    WEIGHT_ENDPOINT = 0.15

    # Decision Thresholds
    COMMIT_THRESHOLD = 0.65
    PREVIEW_THRESHOLD = 0.40

    def __init__(self, language_code: str = "ta-IN"):
        self.language_code = language_code
        self.committed_source_text: str = ""
        self.committed_translation_text: str = ""
        self.last_partial_text: str = ""
        self.last_partial_time: float = time.time()
        self.unchanged_partial_count: int = 0
        self.current_utterance_idx: int = -1

    def reset_utterance(self, utterance_idx: int = -1):
        self.committed_source_text = ""
        self.committed_translation_text = ""
        self.last_partial_text = ""
        self.last_partial_time = time.time()
        self.unchanged_partial_count = 0
        self.current_utterance_idx = utterance_idx

    def evaluate_partial(
        self,
        current_text: str,
        stability_info: Dict[str, Any],
        utterance_idx: int,
        is_final_event: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate an incoming transcript update against evidence gates.
        """
        if utterance_idx != self.current_utterance_idx:
            self.reset_utterance(utterance_idx)

        current_clean = current_text.strip()
        now = time.time()

        # Update endpoint / temporal evidence
        if current_clean == self.last_partial_text and current_clean != "":
            self.unchanged_partial_count += 1
        else:
            self.unchanged_partial_count = 0
            self.last_partial_text = current_clean
            self.last_partial_time = now

        # Check for REVISE condition:
        # If we previously committed text, but the current hypothesis no longer starts with that committed text
        if self.committed_source_text:
            committed_words = self.committed_source_text.split()
            current_words = current_clean.split()
            # If current words is shorter than committed words or prefix diverges
            if len(current_words) < len(committed_words) or current_words[:len(committed_words)] != committed_words:
                return {
                    "decision": "REVISE",
                    "commit_score": 0.0,
                    "stability_score": stability_info.get("stability_score", 0.0),
                    "completeness_score": 0.0,
                    "endpoint_score": 0.0,
                    "committed_source": "",
                    "uncommitted_source": current_clean,
                    "segment_to_translate": current_clean,
                    "reason": "asr_diverged_from_committed_prefix",
                }

        # Determine the uncommitted active tail
        if self.committed_source_text:
            # Strip the already-committed prefix
            uncommitted_text = current_clean[len(self.committed_source_text):].strip()
        else:
            uncommitted_text = current_clean

        if not uncommitted_text:
            return {
                "decision": "WAIT",
                "commit_score": 0.0,
                "stability_score": stability_info.get("stability_score", 0.0),
                "completeness_score": 0.0,
                "endpoint_score": 0.0,
                "committed_source": self.committed_source_text,
                "uncommitted_source": "",
                "segment_to_translate": "",
                "reason": "no_new_uncommitted_text",
            }

        # 1. Linguistic Completeness of the uncommitted unit
        completeness_info = CompletenessAnalyzer.analyze(
            text=uncommitted_text,
            language_code=self.language_code
        )
        comp_score = completeness_info.get("completeness_score", 0.0)

        # 2. Stability score of the whole hypothesis
        stab_score = stability_info.get("stability_score", 0.0)

        # 3. Endpoint evidence score
        if is_final_event:
            endpoint_score = 1.0
        else:
            # Hypotheses that have stayed unchanged for >= 2 partials get higher endpoint evidence
            endpoint_score = min(1.0, 0.25 * self.unchanged_partial_count)

        # Weighted Commit Score Calculation
        commit_score = round(
            (self.WEIGHT_STABILITY * stab_score) +
            (self.WEIGHT_COMPLETENESS * comp_score) +
            (self.WEIGHT_ENDPOINT * endpoint_score),
            3
        )

        # Final events always force COMMIT for whatever remaining uncommitted text exists
        if is_final_event and uncommitted_text:
            decision = "COMMIT"
        elif commit_score >= self.COMMIT_THRESHOLD and completeness_info.get("state") == "COMPLETE":
            decision = "COMMIT"
        elif commit_score >= self.PREVIEW_THRESHOLD:
            decision = "PREVIEW"
        else:
            decision = "WAIT"

        # The segment to translate with context:
        # Context-aware: to avoid translating isolated verbs/fragments without context,
        # we pass the full utterance if newly committing, or the compound clause.
        segment_to_translate = current_clean if decision in ("COMMIT", "PREVIEW") else ""

        return {
            "decision": decision,
            "commit_score": commit_score,
            "stability_score": stab_score,
            "completeness_score": comp_score,
            "endpoint_score": endpoint_score,
            "completeness_state": completeness_info.get("state"),
            "completeness_reason": completeness_info.get("reason"),
            "committed_source": self.committed_source_text,
            "uncommitted_source": uncommitted_text,
            "segment_to_translate": segment_to_translate,
            "new_commit_candidate": current_clean if decision == "COMMIT" else None,
        }

    def record_commit(self, new_source_text: str, new_translation_text: str):
        """
        Record a newly committed segment to prevent re-translation and maintain stable UI.
        """
        self.committed_source_text = new_source_text
        self.committed_translation_text = new_translation_text
