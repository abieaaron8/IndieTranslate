"""
backend/completeness_analyzer.py

Lightweight deterministic Indic-aware completeness analyzer.
Evaluates partial transcripts for Tamil (ta-IN), Hindi (hi-IN), and English (en-IN).
Returns structured metadata:
- state: INCOMPLETE | AMBIGUOUS | COMPLETE
- completeness_score: float (0.0 to 1.0)
- reason: descriptive rule tag
"""

import re
from typing import Dict, Any, List


class CompletenessAnalyzer:
    # Common connectors / prepositions / trailing particles indicating continuation
    TAMIL_CONTINUATION_PATTERNS = [
        r"மற்றும்$", r"ஆனால்$", r"ஏனென்றால்$", r"அப்புறம்$", r"பிறகு$", r"என்று$", r"என$",
        r"ஆகையால்$", r"எனவே$", r"அல்லது$", r"கூட$", r"உம்$", r"போது$", r"இருந்து$",
        r"[கசதப]்கு$", r"க்கு$", r"இல்$", r"ஆல்$", r"உடன்$", r"ஓடு$",  # Dative/Locative/Instrumental case suffixes at word end
    ]

    # Tamil common finite verb suffixes & copulas indicating sentence / clause closure
    TAMIL_FINITE_VERB_PATTERNS = [
        r"போறேன்$", r"வருகிறேன்$", r"போகிறேன்$", r"செய்கிறேன்$", r"இருக்கிறேன்$", r"இருக்கிறார்$",
        r"வந்தேன்$", r"சென்றேன்$", r"பண்ணினேன்$", r"பண்ணுகிறேன்$", r"செய்தேன்$", r"இருந்தது$",
        r"செய்வேன்$", r"வருவேன்$", r"போவேன்$", r"இருப்பேன்$", r"செல்கிறார்$", r"செய்கிறார்கள்$",
        r"வேண்டும்$", r"வேண்டாம்$", r"முடியும்$", r"முடியாது$", r"ஆயிற்று$", r"இல்லை$", r"உண்டு$",
        r"றேன்$", r"றான்$", r"றாள்$", r"றார்$", r"றார்கள்$", r"றது$", r"னேன்$", r"னான்$", r"னாள்$", r"னார்$", r"னார்கள்$",
        r"கிறேன்$", r"கிறான்$", r"கிறாள்$", r"கிறார்$", r"கிறார்கள்$", r"கிறது$",
    ]

    HINDI_CONTINUATION_PATTERNS = [
        r"और$", r"लेकिन$", r"किंतु$", r"परंतु$", r"क्योंकि$", r"अगर$", r"यदि$", r"या$",
        r"तो$", r"भी$", r"को$", r"से$", r"में$", r"पर$", r"का$", r"के$", r"की$", r"ने$",
    ]

    HINDI_FINITE_VERB_PATTERNS = [
        r"हूं$", r"हूँ$", r"है$", r"हैं$", r"हो$", r"था$", r"थी$", r"थे$",
        r"जाऊंगा$", r"जाऊंगी$", r"जाएंगे$", r"करूंगा$", r"करूंगी$", r"करेंगे$",
        r"आऊंगा$", r"आऊंगी$", r"आएंगे$", r"सकता$", r"सकती$", r"सकते$",
        r"गया$", r"गई$", r"गए$", r"लिया$", r"ली$", r"लिए$", r"दिया$", r"दी$", r"दिए$",
        r"रहा$", r"रही$", r"रहे$",
    ]

    ENGLISH_CONTINUATION_PATTERNS = [
        r"\band$", r"\bbut$", r"\bbecause$", r"\bor$", r"\bif$", r"\bthat$", r"\bwhich$",
        r"\bto$", r"\bfor$", r"\bwith$", r"\bin$", r"\bon$", r"\bat$", r"\bfrom$", r"\bof$",
        r"\bthe$", r"\ba$", r"\ban$", r"\bmy$", r"\byour$", r"\bhis$", r"\bher$", r"\bour$", r"\btheir$",
        r"\bis$", r"\bam$", r"\bare$", r"\bwas$", r"\bwere$", r"\bwill$", r"\bgoing$",
    ]

    @classmethod
    def analyze(cls, text: str, language_code: str = "ta-IN") -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {
                "state": "INCOMPLETE",
                "completeness_score": 0.0,
                "reason": "empty_input",
            }

        words = cleaned.split()
        word_count = len(words)

        # 1. Very short fragments (< 3 words) are rarely a full semantic clause
        if word_count < 2:
            return {
                "state": "INCOMPLETE",
                "completeness_score": 0.1,
                "reason": "single_word_fragment",
            }

        lang_prefix = language_code.split("-")[0].lower()

        if lang_prefix == "ta":
            return cls._analyze_tamil(words, cleaned)
        elif lang_prefix == "hi":
            return cls._analyze_hindi(words, cleaned)
        elif lang_prefix == "en":
            return cls._analyze_english(words, cleaned)
        else:
            # Generic fallback
            if word_count >= 4:
                return {"state": "AMBIGUOUS", "completeness_score": 0.5, "reason": "generic_length"}
            return {"state": "INCOMPLETE", "completeness_score": 0.3, "reason": "generic_short"}

    @classmethod
    def _analyze_tamil(cls, words: List[str], full_text: str) -> Dict[str, Any]:
        last_word = words[-1]
        word_count = len(words)

        # Check for obvious continuation connectors or dative endings on the last word
        for pat in cls.TAMIL_CONTINUATION_PATTERNS:
            if re.search(pat, last_word):
                return {
                    "state": "INCOMPLETE",
                    "completeness_score": 0.25,
                    "reason": "trailing_connector_or_case_marker",
                }

        # Check for finite verb / predicate ending
        has_finite_verb = False
        for pat in cls.TAMIL_FINITE_VERB_PATTERNS:
            if re.search(pat, last_word):
                has_finite_verb = True
                break

        if has_finite_verb:
            if word_count >= 3:
                return {
                    "state": "COMPLETE",
                    "completeness_score": 0.90,
                    "reason": "finite_predicate_observed",
                }
            else:
                return {
                    "state": "AMBIGUOUS",
                    "completeness_score": 0.65,
                    "reason": "short_finite_verb_phrase",
                }

        # In SOV Tamil, missing verb at the end usually means predicate is not yet spoken
        if word_count >= 4:
            return {
                "state": "AMBIGUOUS",
                "completeness_score": 0.45,
                "reason": "unconfirmed_predicate_sov",
            }

        return {
            "state": "INCOMPLETE",
            "completeness_score": 0.30,
            "reason": "missing_predicate",
        }

    @classmethod
    def _analyze_hindi(cls, words: List[str], full_text: str) -> Dict[str, Any]:
        last_word = words[-1]
        word_count = len(words)

        # Trailing prepositions/connectors
        for pat in cls.HINDI_CONTINUATION_PATTERNS:
            if re.search(pat, last_word):
                return {
                    "state": "INCOMPLETE",
                    "completeness_score": 0.25,
                    "reason": "trailing_connector_or_postposition",
                }

        # Check for Hindi finite auxiliary or verb ending (जाऊंगा, है, था, etc.)
        has_finite_verb = False
        for pat in cls.HINDI_FINITE_VERB_PATTERNS:
            if re.search(pat, last_word) or (word_count >= 2 and re.search(pat, words[-2])):
                has_finite_verb = True
                break

        if has_finite_verb:
            if word_count >= 3:
                return {
                    "state": "COMPLETE",
                    "completeness_score": 0.90,
                    "reason": "finite_predicate_observed",
                }
            else:
                return {
                    "state": "AMBIGUOUS",
                    "completeness_score": 0.65,
                    "reason": "short_finite_verb_phrase",
                }

        if word_count >= 4:
            return {
                "state": "AMBIGUOUS",
                "completeness_score": 0.45,
                "reason": "unconfirmed_predicate_sov",
            }

        return {
            "state": "INCOMPLETE",
            "completeness_score": 0.30,
            "reason": "missing_predicate",
        }

    @classmethod
    def _analyze_english(cls, words: List[str], full_text: str) -> Dict[str, Any]:
        last_word = words[-1].lower()
        word_count = len(words)

        # Check continuation prepositions / conjunctions / auxiliary verbs at end
        for pat in cls.ENGLISH_CONTINUATION_PATTERNS:
            if re.search(pat, last_word):
                return {
                    "state": "INCOMPLETE",
                    "completeness_score": 0.20,
                    "reason": "trailing_preposition_or_auxiliary",
                }

        # English SVO: if >= 4 words and doesn't end in connector/preposition, likely a solid clause
        if word_count >= 4:
            return {
                "state": "COMPLETE",
                "completeness_score": 0.85,
                "reason": "svo_clause_candidate",
            }
        elif word_count >= 2:
            return {
                "state": "AMBIGUOUS",
                "completeness_score": 0.50,
                "reason": "short_phrase",
            }

        return {
            "state": "INCOMPLETE",
            "completeness_score": 0.20,
            "reason": "incomplete_fragment",
        }
