"""
Local test suite for testing the 10 requested cases of Evidence-Gated Commit Engine.
"""
from stability_analyzer import ASRStabilityAnalyzer
from completeness_analyzer import CompletenessAnalyzer
from commit_engine import CommitEngine


def run_tests():
    print("=== RUNNING EVIDENCE-GATED COMMIT ENGINE TESTS ===")

    # Case 1: Tamil "நான்" -> Expected: WAIT
    ce = CommitEngine(language_code="ta-IN")
    sa = ASRStabilityAnalyzer()
    s1 = sa.analyze("நான்", 0)
    d1 = ce.evaluate_partial("நான்", s1, 0)
    print("CASE 1: Tamil [naan] -> Decision:", d1['decision'], "| Score:", d1['commit_score'], "| State:", d1['completeness_state'])
    assert d1["decision"] == "WAIT"

    # Case 2: Tamil "நான் நாளைக்கு" -> Expected: WAIT
    s2 = sa.analyze("நான் நாளைக்கு", 0)
    d2 = ce.evaluate_partial("நான் நாளைக்கு", s2, 0)
    print("CASE 2: Tamil [naan naalaikku] -> Decision:", d2['decision'], "| Score:", d2['commit_score'], "| State:", d2['completeness_state'])
    assert d2["decision"] == "WAIT"

    # Case 3: Tamil "நான் நாளைக்கு சென்னைக்கு" -> Expected: WAIT or AMBIGUOUS
    s3 = sa.analyze("நான் நாளைக்கு சென்னைக்கு", 0)
    d3 = ce.evaluate_partial("நான் நாளைக்கு சென்னைக்கு", s3, 0)
    print("CASE 3: Tamil [naan naalaikku chennaikku] -> Decision:", d3['decision'], "| Score:", d3['commit_score'], "| State:", d3['completeness_state'])
    assert d3["decision"] in ("WAIT", "PREVIEW")
    assert d3["completeness_state"] in ("INCOMPLETE", "AMBIGUOUS")

    # Case 4: Tamil "நான் நாளைக்கு சென்னைக்கு போறேன்" -> Expected: COMMIT
    s4 = sa.analyze("நான் நாளைக்கு சென்னைக்கு போறேன்", 0)
    d4 = ce.evaluate_partial("நான் நாளைக்கு சென்னைக்கு போறேன்", s4, 0)
    print("CASE 4: Tamil [naan naalaikku chennaikku poren] -> Decision:", d4['decision'], "| Score:", d4['commit_score'], "| State:", d4['completeness_state'])
    assert d4["decision"] == "COMMIT"
    ce.record_commit(d4["new_commit_candidate"], "I am going to Chennai tomorrow.")

    # Case 5: Hindi "मैं" -> Expected: WAIT
    ce_hi = CommitEngine(language_code="hi-IN")
    sa_hi = ASRStabilityAnalyzer()
    sh1 = sa_hi.analyze("मैं", 0)
    dh1 = ce_hi.evaluate_partial("मैं", sh1, 0)
    print("CASE 5: Hindi [main] -> Decision:", dh1['decision'], "| Score:", dh1['commit_score'], "| State:", dh1['completeness_state'])
    assert dh1["decision"] == "WAIT"

    # Case 6: Hindi "मैं कल" -> Expected: WAIT
    sh2 = sa_hi.analyze("मैं कल", 0)
    dh2 = ce_hi.evaluate_partial("मैं कल", sh2, 0)
    print("CASE 6: Hindi [main kal] -> Decision:", dh2['decision'], "| Score:", dh2['commit_score'], "| State:", dh2['completeness_state'])
    assert dh2["decision"] == "WAIT"

    # Case 7: Hindi "मैं कल चेन्नई जाऊंगा" -> Expected: COMMIT
    sa_hi.analyze("मैं कल चेन्नई", 0)
    sh3 = sa_hi.analyze("मैं कल चेन्नई जाऊंगा", 0)
    dh3 = ce_hi.evaluate_partial("मैं कल चेन्नई जाऊंगा", sh3, 0)
    print("CASE 7: Hindi [main kal chennai jaunga] -> Decision:", dh3['decision'], "| Score:", dh3['commit_score'], "| State:", dh3['completeness_state'])
    assert dh3["decision"] == "COMMIT"

    # Case 8: English "I am going to" -> Expected: WAIT
    ce_en = CommitEngine(language_code="en-IN")
    sa_en = ASRStabilityAnalyzer()
    se1 = sa_en.analyze("I am going to", 0)
    de1 = ce_en.evaluate_partial("I am going to", se1, 0)
    print("CASE 8: English [I am going to] -> Decision:", de1['decision'], "| Score:", de1['commit_score'], "| State:", de1['completeness_state'])
    assert de1["decision"] in ("WAIT", "PREVIEW")
    assert de1["decision"] != "COMMIT"

    # Case 9: English "I am going to Chennai tomorrow" -> Expected: COMMIT
    sa_en.analyze("I am going to Chennai", 0)
    se2 = sa_en.analyze("I am going to Chennai tomorrow", 0)
    de2 = ce_en.evaluate_partial("I am going to Chennai tomorrow", se2, 0)
    print("CASE 9: English [I am going to Chennai tomorrow] -> Decision:", de2['decision'], "| Score:", de2['commit_score'], "| State:", de2['completeness_state'])
    assert de2["decision"] == "COMMIT"

    # Case 10: Revision test - ASR revisions change previously committed prefix
    d_rev = ce.evaluate_partial("நான் இன்று சென்னைக்கு போறேன்", {"stability_score": 0.5}, 0)
    print("CASE 10: Revision -> Decision:", d_rev['decision'], "| Reason:", d_rev['reason'])
    assert d_rev["decision"] == "REVISE"

    print("\nALL 10 TEST CASES PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
