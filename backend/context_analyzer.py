"""
backend/context_analyzer.py

Context-Aware Semantic Preservation Analyzer.
Detects entities that should be preserved based on context patterns:
- Company / Organization names triggered by contextual cues ("company is Fish", "joined Apple", "works at X")
- Proper names / Person names ("to Rahul", "with Priya", "by John")
- Currencies and Numbers ("₹5000", "$100", "Rs 500")
- URLs and Emails ("test@example.com", "https://sarvam.ai")
- Phone Numbers / Numeric IDs ("+91 9876543210")
- Proper nouns / Romanized places in code-mixed Indic text ("Chennai", "Mumbai", "Delhi")

Avoids protecting common words in ordinary contexts ("I ate a fish", "I ate an apple").
"""

import re
from typing import List, Dict, Any, Tuple


class ContextAnalyzer:
    # Context cues that precede organization / company names
    ORG_PRE_CUES = [
        r"(?:my|our|his|her|their)\s+company(?:'s)?\s+(?:name\s+)?(?:is|was)\s+([A-Za-z0-9_\-]+)",
        r"(?:joined|join|joining|founded|starts?|worked?\s+at|working\s+at)\s+([A-Z][A-Za-z0-9_\-]*)",
        r"(?:company|organization|startup|corp|inc|ltd)\s+named?\s+([A-Za-z0-9_\-]+)",
    ]

    # Context cues that precede person names (e.g., "to Rahul", "with Priya", "call John")
    PERSON_PRE_CUES = [
        r"\b(?:to|with|by|for|from|contact|call|meet)\s+([A-Z][a-z]+)\b",
        r"\b(?:mr|ms|mrs|dr|prof)\.?\s+([A-Z][a-z]+)\b",
    ]

    # Currency patterns (e.g. ₹5000, Rs. 500, $100, 5000 rupees)
    CURRENCY_PATTERNS = [
        r"(?:₹|Rs\.?|INR|\$|€|£)\s*\d+(?:,\d+)*(?:\.\d+)?",
        r"\d+(?:,\d+)*(?:\.\d+)?\s*(?:rupees|dollars|rs|inr|crore|lakh)",
    ]

    # URLs and Emails
    URL_PATTERN = r"https?://[^\s]+"
    EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    # Known Indian cities / places and standard proper nouns often appearing in code-mixing
    KNOWN_CODE_MIXED_PLACES = {
        "chennai", "mumbai", "delhi", "bengaluru", "bangalore", "hyderabad",
        "kolkata", "pune", "ahmedabad", "jaipur", "kochi", "madurai", "coimbatore",
        "salem", "trichy", "tirunelveli", "noida", "gurugram", "chandigarh"
    }

    @classmethod
    def detect_entities(cls, text: str, language_code: str = "ta-IN") -> List[Dict[str, Any]]:
        """
        Scan text for contextual entities and return list of matches with start/end spans and entity type.
        """
        entities = []
        covered_spans = []

        def is_overlapping(start: int, end: int) -> bool:
            for s, e in covered_spans:
                if max(start, s) < min(end, e):
                    return True
            return False

        def add_entity(start: int, end: int, val: str, ent_type: str):
            if not is_overlapping(start, end) and val.strip():
                entities.append({
                    "start": start,
                    "end": end,
                    "text": val.strip(),
                    "type": ent_type,
                })
                covered_spans.append((start, end))

        # 1. URLs and Emails
        for m in re.finditer(cls.URL_PATTERN, text):
            add_entity(m.start(), m.end(), m.group(0), "URL")

        for m in re.finditer(cls.EMAIL_PATTERN, text):
            add_entity(m.start(), m.end(), m.group(0), "EMAIL")

        # 2. Currency and Numeric units
        for pat in cls.CURRENCY_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                add_entity(m.start(), m.end(), m.group(0), "CURRENCY")

        # 3. Contextual Organization Names ("My company's name is Fish", "joined Apple")
        for pat in cls.ORG_PRE_CUES:
            for m in re.finditer(pat, text, re.IGNORECASE):
                ent_text = m.group(1)
                start = m.start(1)
                end = m.end(1)
                add_entity(start, end, ent_text, "ORGANIZATION")

        # 4. Contextual Person Names ("Send ₹5000 to Rahul", "call Priya")
        for pat in cls.PERSON_PRE_CUES:
            for m in re.finditer(pat, text):
                ent_text = m.group(1)
                start = m.start(1)
                end = m.end(1)
                # Verify not a common lowercased stopword
                if ent_text.lower() not in {"the", "a", "an", "me", "him", "her", "us", "them", "chennai", "delhi"}:
                    add_entity(start, end, ent_text, "PERSON")

        # 5. Code-Mixed Place Names & Proper Nouns in Indic Text ("நான் Chennai போறேன்")
        words = list(re.finditer(r"[A-Za-z0-9_\-]+", text))
        for m in words:
            word = m.group(0)
            if word.lower() in cls.KNOWN_CODE_MIXED_PLACES:
                add_entity(m.start(), m.end(), word, "LOCATION")
            elif word.istitle() and len(word) >= 3:
                # If inside non-English text, standalone capitalized Latin words are almost always names/places/brands
                is_indic_context = any(ord(c) > 127 for c in text)
                if is_indic_context and word.lower() not in {"tomorrow", "today", "yesterday", "daily"}:
                    add_entity(m.start(), m.end(), word, "PROPER_NOUN")

        # Sort entities by start position
        entities.sort(key=lambda x: x["start"])
        return entities
