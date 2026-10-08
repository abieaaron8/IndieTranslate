"""
backend/entity_protector.py

Protects detected entities with placeholders before machine translation
and restores them afterwards.

Example:
Input: "My company's name is Fish"
Protected: "My company's name is __ENTITY_0__"
Placeholders: {"__ENTITY_0__": "Fish"}

After Sarvam Translation:
Translated: "என்னுடைய நிறுவனத்தின் பெயர் __ENTITY_0__"
Restored: "என்னுடைய நிறுவனத்தின் பெயர் Fish"
"""

import re
from typing import Dict, Any, List, Tuple
from context_analyzer import ContextAnalyzer


class EntityProtector:
    @classmethod
    def protect(cls, text: str, language_code: str = "ta-IN") -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """
        Detect entities in text, substitute with placeholders, and return (protected_text, entity_map, detected_entities).
        """
        detected_entities = ContextAnalyzer.detect_entities(text, language_code)
        if not detected_entities:
            return text, {}, []

        # Sort spans in reverse order to substitute without offsetting earlier indices
        sorted_spans = sorted(detected_entities, key=lambda x: x["start"], reverse=True)

        entity_map: Dict[str, str] = {}
        protected_chars = list(text)

        for i, ent in enumerate(reversed(sorted_spans)):
            placeholder = f"__ENTITY_{i}__"
            entity_map[placeholder] = ent["text"]

        for i, ent in enumerate(sorted_spans):
            # Index matching the forward loop
            actual_idx = len(sorted_spans) - 1 - i
            placeholder = f"__ENTITY_{actual_idx}__"
            start = ent["start"]
            end = ent["end"]
            protected_chars[start:end] = list(placeholder)

        protected_text = "".join(protected_chars)
        return protected_text, entity_map, detected_entities

    @classmethod
    def restore(cls, translated_text: str, entity_map: Dict[str, str]) -> str:
        """
        Restore original entity text into translated output.
        Handles variations in placeholder spacing or casing introduced by translation models.
        """
        if not entity_map or not translated_text:
            return translated_text

        result = translated_text
        for placeholder, original_val in entity_map.items():
            # Standard exact match
            if placeholder in result:
                result = result.replace(placeholder, original_val)
            else:
                # Regex match for potential spaces or bracket variations e.g. <ENTITY_0>, __ ENTITY_0 __
                # Extract index from placeholder (e.g. 0 from __ENTITY_0__)
                idx_match = re.search(r"\d+", placeholder)
                if idx_match:
                    idx = idx_match.group(0)
                    pattern = rf"(?:__|__\s*|<|\[)\s*ENTITY_{idx}\s*(?:__|__\s*|>|\])"
                    result = re.sub(pattern, original_val, result, flags=re.IGNORECASE)

        return result
