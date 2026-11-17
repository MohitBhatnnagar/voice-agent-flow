import re
import unittest

from voiceflow import Call, Config, Intent, TurnTiming, percentile, redact, report, score
from voiceflow.flow import ENDED, HANDED_OFF, MORE, VERIFY


def intents():
    return [
        Intent("order_status", re.compile(r"\b(order|delivery)\b", re.I), needs_verification=True,
               reply=lambda c: "Your order arrives tomorrow."),
        Intent("hours", re.compile(r"\b(hours|open)\b", re.I), reply=lambda c: "We open at nine."),
        Intent("refund", re.compile(r"\brefund\b", re.I), high_risk=True),
    ]


def call(**cfg):
    c = Call(intents(), Config(**cfg))
    c.start()
    return c


class FlowTests(unittest.TestCase):
    def test_greeting_includes_recording_disclosure(self):
        c = Call(intents())
        self.assertIn("recorded", c.start().text)

    def test_simple_intent_then_goodbye(self):
        c = call()
        a = c.on_user("When do you open?")
        self.assertEqual((a.kind, c.state), ("say", MORE))
        self.assertIn("We open at nine.", a.text)
        end = c.on_user("no that's all")
        self.assertEqual((end.kind, c.state), ("end", ENDED))

    def test_another_request_after_answer(self):
        c = call()
        c.on_user("opening hours?")
        a = c.on_user("yes, what are your hours again")
        self.assertIn("We open at nine.", a.text)

    def test_verification_gate_blocks_until_code_is_correct(self):
        c = call()
        a = c.on_user("Where is my order?")
        self.assertEqual(c.state, VERIFY)
        self.assertNotIn("arrives", a.text)
        b = c.on_user("my code is 1 2 3 4")
        self.assertIn("arrives tomorrow", b.text)
        self.assertTrue(c.verified)

    def test_wrong_code_retries_then_hands_off(self):
        c = call()
        c.on_user("my order please")
        self.assertEqual(c.on_user("9999").kind, "say")
        a = c.on_user("0000")
        self.assertEqual((a.kind, c.state), ("handoff", HANDED_OFF))
        self.assertEqual(a.packet["reason"], "verification failed")
        self.assertFalse(a.packet["verified"])

    def test_verified_caller_is_not_asked_again(self):
        c = call()
        c.on_user("my order"); c.on_user("1234")
        a = c.on_user("yes check my delivery again")
        self.assertIn("arrives tomorrow", a.text)

    def test_unmatched_requests_reprompt_then_hand_off(self):
        c = call()
        self.assertEqual(c.on_user("blah blah").kind, "say")
        self.assertEqual(c.on_user("mumble").kind, "say")
        self.assertEqual(c.on_user("gibberish").kind, "handoff")

    def test_low_asr_confidence_counts_as_no_match(self):
        c = call()
        self.assertIn("did not catch", c.on_user("opening hours", confidence=0.2).text)

    def test_match_resets_the_no_match_counter(self):
        c = call()
        c.on_user("blah")
        c.on_user("opening hours"); c.on_user("yes hours again")
        self.assertEqual(c.no_match, 0)

    def test_high_risk_intent_always_goes_to_a_person(self):
        a = call().on_user("I want a refund")
        self.assertEqual(a.kind, "handoff")
        self.assertIn("high-risk", a.packet["reason"])

    def test_asking_for_a_human_works_in_any_state(self):
        for setup in (lambda c: None, lambda c: c.on_user("my order")):
            c = call()
            setup(c)
            self.assertEqual(c.on_user("let me talk to a real person").kind, "handoff")

    def test_repeated_negative_sentiment_escalates(self):
        c = call()
        self.assertEqual(c.on_user("this is ridiculous but what are your hours").kind, "say")
        self.assertEqual(c.on_user("this is useless").kind, "handoff")

    def test_turn_limit(self):
        c = call(max_turns=2)
        c.on_user("hours"); c.on_user("yes hours")
        self.assertEqual(c.on_user("yes hours").packet["reason"], "call is too long")

    def test_no_events_after_the_call_ends(self):
        c = call()
        c.on_user("hours"); c.on_user("no")
        with self.assertRaises(RuntimeError):
            c.on_user("hello?")


class BargeInTests(unittest.TestCase):
    def test_interrupted_agent_turn_keeps_only_what_was_spoken(self):
        c = call()
        c.on_barge_in("Hello. This call may be")
        self.assertEqual(c.turns[-1].text, "Hello. This call may be")
        self.assertTrue(c.turns[-1].interrupted)

    def test_repeated_barge_in_switches_to_terse_prompts(self):
        c = call()
        for _ in range(3):
            c.on_barge_in("...")
        self.assertTrue(c.terse)
        a = c.on_user("when do you open")
        self.assertEqual(a.text, "We open at nine. Is there anything else I can help with?".split(". ")[0] + ".")

    def test_one_barge_in_does_not_change_style(self):
        c = call()
        c.on_barge_in("Hel")
        self.assertFalse(c.terse)


class RedactionTests(unittest.TestCase):
    def test_redacts_numbers_and_emails(self):
        out = redact("my card is 4111 1111 1111 1111 and mail me at sam@example.com")
        self.assertNotIn("4111", out)
        self.assertNotIn("sam@", out)

    def test_keeps_short_numbers(self):
        self.assertEqual(redact("I want 2 tickets on the 5th"), "I want 2 tickets on the 5th")

    def test_handoff_packet_is_redacted(self):
        c = call()
        a = c.on_user("refund please, email sam@example.com or call 9876543210")
        text = " ".join(a.packet["summary"])
        self.assertNotIn("sam@example.com", text)
        self.assertNotIn("9876543210", text)

    def test_stored_transcript_is_redacted(self):
        c = call()
        c.on_user("my code is 1 2 3 4 and email a@b.co")
        self.assertNotIn("a@b.co", str(c.transcript()))

    def test_packet_contains_only_recent_turns(self):
        c = call()
        for _ in range(4):
            c.on_user("hours"); c.on_user("yes hours")
        self.assertLessEqual(len(c.on_user("refund").packet["summary"]), 6)


class LatencyTests(unittest.TestCase):
    def test_time_to_first_audio_is_the_sum_of_blocking_stages(self):
        self.assertEqual(TurnTiming(200, 300, 150, 50).time_to_first_audio, 700)

    def test_worst_stage(self):
        self.assertEqual(TurnTiming(100, 900, 100).worst_stage(), "llm")

    def test_percentile_nearest_rank(self):
        v = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.assertEqual((percentile(v, 50), percentile(v, 95), percentile(v, 100)), (50, 100, 100))
        self.assertEqual(percentile([5], 50), 5)

    def test_percentile_validation(self):
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1], 0)

    def test_report_ok_within_budget(self):
        r = report([TurnTiming(150, 300, 150)] * 10, budget_ms=800)
        self.assertTrue(r["ok"])
        self.assertEqual((r["breach_rate"], r["bottleneck"]), (0.0, None))

    def test_report_flags_breaches_and_names_the_bottleneck(self):
        ts = [TurnTiming(150, 300, 150)] * 6 + [TurnTiming(150, 1200, 150)] * 4
        r = report(ts, budget_ms=800)
        self.assertEqual(r["breach_rate"], 0.4)
        self.assertEqual(r["bottleneck"], "llm")
        self.assertFalse(r["ok"])

    def test_one_slow_turn_in_many_breaches_p95_not_p50(self):
        ts = [TurnTiming(100, 300, 100)] * 19 + [TurnTiming(100, 3000, 100)]
        r = report(ts, budget_ms=800)
        self.assertLessEqual(r["p50_ms"], 800)
        self.assertEqual(r["max_ms"], 3200)

    def test_empty(self):
        with self.assertRaises(ValueError):
            report([])


class QaTests(unittest.TestCase):
    def good_call(self):
        c = call()
        c.on_user("my order"); c.on_user("1234"); c.on_user("no thanks")
        return c

    def test_clean_call_scores_full_marks(self):
        r = score(self.good_call())
        self.assertEqual((r["score"], r["failed"], r["needs_review"]), (100, [], False))

    def test_missing_recording_disclosure(self):
        c = Call(intents(), Config(disclosure="Welcome."))
        c.start(); c.on_user("hours"); c.on_user("no")
        self.assertIn("greeting mentions that the call may be recorded", score(c)["failed"])

    def test_disclosure_before_verification_is_a_serious_failure(self):
        c = Call([Intent("order_status", re.compile("order"), needs_verification=False,
                         reply=lambda x: "Your order arrives tomorrow.")])
        c.start(); c.on_user("my order"); c.on_user("no")
        r = score(c)
        self.assertFalse(r["passed"]["verified_before_disclosure"])
        self.assertTrue(r["needs_review"])

    def test_sensitive_info_after_verification_is_fine(self):
        self.assertTrue(score(self.good_call())["passed"]["verified_before_disclosure"])

    def test_unfinished_call_is_not_resolved(self):
        c = call()
        c.on_user("hours")
        self.assertFalse(score(c)["passed"]["resolved_or_handed_off"])

    def test_handoff_counts_as_a_valid_outcome(self):
        c = call()
        c.on_user("refund")
        r = score(c)
        self.assertTrue(r["passed"]["resolved_or_handed_off"])
        self.assertTrue(r["passed"]["polite_close"])

    def test_many_interruptions_lower_the_score(self):
        c = call()
        c.on_user("hours")
        for t in c.turns:
            if t.speaker == "agent":
                t.interrupted = True
        self.assertFalse(score(c)["passed"]["few_interruptions"])


if __name__ == "__main__":
    unittest.main()
