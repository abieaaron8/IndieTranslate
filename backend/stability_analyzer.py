"""
ASR Stability Analyzer
Compares successive partial transcripts to compute:
- stable prefix (matched tokens across updates)
- unstable suffix (new or fluctuating tokens)
- stability score (ratio of stable tokens to total current tokens)
"""
from typing import Tuple, Dict, Any, List


class ASRStabilityAnalyzer:
    def __init__(self):
        self.prev_words: List[str] = []
        self.stable_prefix_words: List[str] = []
        self.stable_prefix_count: int = 0
        self.current_utterance_idx: int = -1

    def reset(self, utterance_idx: int = -1):
        self.prev_words = []
        self.stable_prefix_words = []
        self.stable_prefix_count = 0
        self.current_utterance_idx = utterance_idx

    def analyze(self, text: str, utterance_idx: int) -> Dict[str, Any]:
        """
        Analyze a new partial transcript and compare with previous partial.
        """
        if utterance_idx != self.current_utterance_idx:
            self.reset(utterance_idx)

        current_words = text.strip().split() if text.strip() else []

        if not current_words:
            self.prev_words = []
            return {
                "stable_prefix": "",
                "unstable_suffix": "",
                "stability_score": 0.0,
                "stable_token_count": 0,
                "total_token_count": 0,
            }

        if not self.prev_words:
            # First partial in utterance: suffix is everything, 0 stability score
            self.prev_words = current_words
            self.stable_prefix_words = []
            return {
                "stable_prefix": "",
                "unstable_suffix": text,
                "stability_score": 0.0,
                "stable_token_count": 0,
                "total_token_count": len(current_words),
            }

        # Calculate longest common prefix of words between current and previous partials
        match_len = 0
        max_possible = min(len(self.prev_words), len(current_words))
        while match_len < max_possible and self.prev_words[match_len] == current_words[match_len]:
            match_len += 1

        # Words that matched in consecutive partials are considered stable
        self.stable_prefix_words = current_words[:match_len]
        unstable_suffix_words = current_words[match_len:]

        stable_prefix = " ".join(self.stable_prefix_words)
        unstable_suffix = " ".join(unstable_suffix_words)

        total_words = len(current_words)
        # Stability score = ratio of stable tokens to total tokens (0.0 to 1.0)
        stability_score = round(match_len / total_words, 2) if total_words > 0 else 0.0

        self.prev_words = current_words

        return {
            "stable_prefix": stable_prefix,
            "unstable_suffix": unstable_suffix,
            "stability_score": stability_score,
            "stable_token_count": match_len,
            "total_token_count": total_words,
        }
