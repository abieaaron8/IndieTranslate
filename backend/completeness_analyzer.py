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

    TELUGU_CONTINUATION_PATTERNS = [
        r"మరియు$", r"కానీ$", r"ఎందుకంటే$", r"అయితే$", r"లేదా$", r"కూడా$", r"తో$", r"లో$", r"కి$", r"కు$", r"నుండి$",
    ]

    TELUGU_FINITE_VERB_PATTERNS = [
        r"తున్నాను$", r"తున్నాము$", r"తున్నాడు$", r"తున్నది$", r"తున్నారు$",
        r"శాను$", r"శాము$", r"శాడు$", r"శారు$", r"చేస్తాను$", r"చేస్తాము$",
        r"వెళ్తున్నాను$", r"చేస్తున్నాను$", r"ఉన్నాను$", r"ఉంది$", r"ఉన్నారు$",
        r"కావాలి$", r"వద్దు$", r"వచ్చు$", r"లేదు$", r"అయింది$",
    ]

    KANNADA_CONTINUATION_PATTERNS = [
        r"ಮತ್ತು$", r"ಆದರೆ$", r"ಯಾಕೆಂದರೆ$", r"ಅಥವಾ$", r"ಕೂಡ$", r"ಜೊತೆ$", r"ಯಲ್ಲಿ$", r"ಗೆ$", r"ಕೆ$", r"ಇಂದ$",
    ]

    KANNADA_FINITE_VERB_PATTERNS = [
        r"ತ್ತಿದ್ದೇನೆ$", r"ತ್ತಿದ್ದೇವೆ$", r"ತ್ತಾನೆ$", r"ತ್ತಾಳೆ$", r"ತ್ತಾರೆ$",
        r"ದ್ದೇನೆ$", r"ದ್ದೇವೆ$", r"ದ್ದಾರೆ$", r"ದೆನು$", r"ದರು$",
        r"ಹೋಗುತ್ತಿದ್ದೇನೆ$", r"ಮಾಡುತ್ತಿದ್ದೇನೆ$", r"ಇದ್ದೇನೆ$", r"ಆಗಿದೆ$",
        r"ಬೇಕು$", r"ಬೇಡ$", r"ಬಹುದು$", r"ಇಲ್ಲ$",
    ]

    MALAYALAM_CONTINUATION_PATTERNS = [
        r"കൂടാതെ$", r"എന്നാൽ$", r"കാരണം$", r"അല്ലെങ്കിൽ$", r"ഉം$", r"ലേക്ക്$", r"ൽ$", r"ഒപ്പം$", r"നിന്ന്$",
    ]

    MALAYALAM_FINITE_VERB_PATTERNS = [
        r"ന്നു$", r"ുന്നു$", r"ുകയാണു്$", r"ുകയാണു$", r"ുണ്ട്$", r"ഉണ്ട്$",
        r"പോകുന്നു$", r"ചെയ്യുന്നു$", r"വന്നു$", r"പോയി$", r"ചെയ്തു$",
        r"വേണം$", r"വേണ്ട$", r"ചെയ്യാം$", r"ഇല്ല$", r"ആയി$",
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

        raw_words = cleaned.split()
        # Clean punctuation from words for robust predicate pattern matching
        words = [re.sub(r'^[^\w\s\u0900-\u0DFF]+|[^\w\s\u0900-\u0DFF]+$', '', w).strip() for w in raw_words]
        words = [w for w in words if w]
        if not words:
            words = raw_words
        word_count = len(words)

        # 1. Very short fragments (< 2 words)
        if word_count < 2:
            return {
                "state": "INCOMPLETE",
                "completeness_score": 0.15,
                "reason": "single_word_fragment",
            }

        lang_prefix = language_code.split("-")[0].lower()

        if lang_prefix == "ta":
            return cls._analyze_tamil(words, cleaned)
        elif lang_prefix == "hi":
            return cls._analyze_hindi(words, cleaned)
        elif lang_prefix == "te":
            return cls._analyze_telugu(words, cleaned)
        elif lang_prefix == "kn":
            return cls._analyze_kannada(words, cleaned)
        elif lang_prefix == "ml":
            return cls._analyze_malayalam(words, cleaned)
        elif lang_prefix == "en":
            return cls._analyze_english(words, cleaned)
        else:
            # Generic fallback for other Indic languages
            if word_count >= 4:
                return {"state": "COMPLETE", "completeness_score": 0.75, "reason": "generic_length"}
            elif word_count >= 2:
                return {"state": "AMBIGUOUS", "completeness_score": 0.5, "reason": "generic_short"}
            return {"state": "INCOMPLETE", "completeness_score": 0.3, "reason": "generic_fragment"}

    @classmethod
    def _analyze_telugu(cls, words: List[str], full_text: str) -> Dict[str, Any]:
        last_word = words[-1]
        word_count = len(words)
        for pat in cls.TELUGU_CONTINUATION_PATTERNS:
            if re.search(pat, last_word):
                return {"state": "INCOMPLETE", "completeness_score": 0.25, "reason": "trailing_connector_te"}
        for pat in cls.TELUGU_FINITE_VERB_PATTERNS:
            if re.search(pat, last_word) or (word_count >= 2 and re.search(pat, words[-2])):
                if word_count >= 3:
                    return {"state": "COMPLETE", "completeness_score": 0.88, "reason": "finite_predicate_te"}
                return {"state": "AMBIGUOUS", "completeness_score": 0.65, "reason": "short_finite_verb_te"}
        if word_count >= 4:
            return {"state": "AMBIGUOUS", "completeness_score": 0.50, "reason": "sov_clause_te"}
        return {"state": "INCOMPLETE", "completeness_score": 0.30, "reason": "incomplete_te"}

    @classmethod
    def _analyze_kannada(cls, words: List[str], full_text: str) -> Dict[str, Any]:
        last_word = words[-1]
        word_count = len(words)
        for pat in cls.KANNADA_CONTINUATION_PATTERNS:
            if re.search(pat, last_word):
                return {"state": "INCOMPLETE", "completeness_score": 0.25, "reason": "trailing_connector_kn"}
        for pat in cls.KANNADA_FINITE_VERB_PATTERNS:
            if re.search(pat, last_word) or (word_count >= 2 and re.search(pat, words[-2])):
                if word_count >= 3:
                    return {"state": "COMPLETE", "completeness_score": 0.88, "reason": "finite_predicate_kn"}
                return {"state": "AMBIGUOUS", "completeness_score": 0.65, "reason": "short_finite_verb_kn"}
        if word_count >= 4:
            return {"state": "AMBIGUOUS", "completeness_score": 0.50, "reason": "sov_clause_kn"}
        return {"state": "INCOMPLETE", "completeness_score": 0.30, "reason": "incomplete_kn"}

    @classmethod
    def _analyze_malayalam(cls, words: List[str], full_text: str) -> Dict[str, Any]:
        last_word = words[-1]
        word_count = len(words)
        for pat in cls.MALAYALAM_CONTINUATION_PATTERNS:
            if re.search(pat, last_word):
                return {"state": "INCOMPLETE", "completeness_score": 0.25, "reason": "trailing_connector_ml"}
        for pat in cls.MALAYALAM_FINITE_VERB_PATTERNS:
            if re.search(pat, last_word) or (word_count >= 2 and re.search(pat, words[-2])):
                if word_count >= 3:
                    return {"state": "COMPLETE", "completeness_score": 0.88, "reason": "finite_predicate_ml"}
                return {"state": "AMBIGUOUS", "completeness_score": 0.65, "reason": "short_finite_verb_ml"}
        if word_count >= 4:
            return {"state": "AMBIGUOUS", "completeness_score": 0.50, "reason": "sov_clause_ml"}
        return {"state": "INCOMPLETE", "completeness_score": 0.30, "reason": "incomplete_ml"}

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
