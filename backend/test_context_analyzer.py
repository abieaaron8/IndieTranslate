"""
backend/test_context_analyzer.py

Tests for Context-Aware Semantic Preservation & Entity Protection:
1. "My company's name is Fish" -> Fish protected.
2. "I ate a fish" -> fish NOT protected.
3. "நான் Chennai போறேன்" -> Chennai preserved.
4. "Send ₹5000 to Rahul" -> ₹5000 and Rahul preserved.
5. Placeholder round-trip: original -> protected -> mock translation -> restored.
"""

from context_analyzer import ContextAnalyzer
from entity_protector import EntityProtector


def run_tests():
    print("=== RUNNING CONTEXT & ENTITY PROTECTION TESTS ===")

    # Test 1: "My company's name is Fish" -> Fish protected as ORGANIZATION
    t1 = "My company's name is Fish"
    e1 = ContextAnalyzer.detect_entities(t1, "en-IN")
    print("TEST 1: 'My company's name is Fish' -> Detected:", [x['text'] for x in e1])
    assert any(x["text"] == "Fish" and x["type"] == "ORGANIZATION" for x in e1)

    # Test 2: "I ate a fish" -> fish NOT protected
    t2 = "I ate a fish"
    e2 = ContextAnalyzer.detect_entities(t2, "en-IN")
    print("TEST 2: 'I ate a fish' -> Detected:", [x['text'] for x in e2])
    assert len(e2) == 0

    # Test 2b: "I joined Apple" vs "I ate an apple"
    t2b_org = "I joined Apple"
    e2b_org = ContextAnalyzer.detect_entities(t2b_org, "en-IN")
    print("TEST 2b: 'I joined Apple' -> Detected:", [x['text'] for x in e2b_org])
    assert any(x["text"] == "Apple" and x["type"] == "ORGANIZATION" for x in e2b_org)

    t2b_fruit = "I ate an apple"
    e2b_fruit = ContextAnalyzer.detect_entities(t2b_fruit, "en-IN")
    print("TEST 2b: 'I ate an apple' -> Detected:", [x['text'] for x in e2b_fruit])
    assert len(e2b_fruit) == 0

    # Test 3: "நான் Chennai போறேன்" -> Chennai preserved
    t3 = "நான் Chennai போறேன்"
    e3 = ContextAnalyzer.detect_entities(t3, "ta-IN")
    print("TEST 3: Tamil [naan Chennai poren] -> Detected:", [x['text'] for x in e3])
    assert any(x["text"] == "Chennai" for x in e3)

    # Test 4: "Send ₹5000 to Rahul" -> ₹5000 and Rahul preserved
    t4 = "Send \u20b95000 to Rahul"
    e4 = ContextAnalyzer.detect_entities(t4, "en-IN")
    print("TEST 4: 'Send Rs.5000 to Rahul' -> Detected:", [x['text'].encode('ascii', 'replace').decode('ascii') for x in e4])
    assert any("5000" in x["text"] and x["type"] == "CURRENCY" for x in e4)
    assert any(x["text"] == "Rahul" and x["type"] == "PERSON" for x in e4)

    # Test 5: Placeholder round-trip
    t5 = "My company's name is Fish"
    protected_text, entity_map, detected = EntityProtector.protect(t5, "en-IN")
    print("TEST 5 (Protect):", protected_text, "| Map:", entity_map)
    assert "__ENTITY_0__" in protected_text
    assert entity_map["__ENTITY_0__"] == "Fish"

    # Mock translation from translation engine (placeholder retained in Tamil)
    mock_translated = "\u0b8e\u0ba9\u0bcd\u0ba9\u0bc1\u0b9f\u0bc8\u0baf \u0ba8\u0bbf\u0bb1\u0bc1\u0bb5\u0ba9\u0ba4\u0bcd\u0ba4\u0bbf\u0ba9\u0bcd \u0baa\u0bc6\u0baf\u0bb0\u0bcd __ENTITY_0__"
    restored_text = EntityProtector.restore(mock_translated, entity_map)
    print("TEST 5 (Restore):", restored_text.encode('ascii', 'replace').decode('ascii'))
    assert "Fish" in restored_text

    # Test 5b: Variation in placeholder translation spacing (e.g. "__ ENTITY_0 __")
    mock_variation = "Tamil prefix __ ENTITY_0 __"
    restored_variation = EntityProtector.restore(mock_variation, {"__ENTITY_0__": "Rahul"})
    print("TEST 5b (Variation Restore):", restored_variation)
    assert "Rahul" in restored_variation

    print("\nALL CONTEXT & ENTITY PROTECTION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
