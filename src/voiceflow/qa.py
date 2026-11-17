"""Score a finished call against a rubric. Deterministic checks first; they are cheap, explainable
and a good first filter for which calls a human QA reviewer should listen to."""
from __future__ import annotations

import re

from .flow import ENDED, HANDED_OFF, Call

THANKS = re.compile(r"\b(thank|goodbye)\b", re.I)
RECORDING = re.compile(r"\b(record(ed|ing))\b", re.I)
SENSITIVE_REPLY = re.compile(r"\b(balance|outstanding|account status|your order)\b", re.I)
EMAIL_RAW = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
LONG_RAW = re.compile(r"\b(?:\d[ \-]?){4,}\b")

CHECKS = {
    "recording_disclosure": (20, "greeting mentions that the call may be recorded"),
    "verified_before_disclosure": (30, "no sensitive information given before the caller was verified"),
    "resolved_or_handed_off": (20, "call ended resolved, or was handed to a person"),
    "polite_close": (10, "agent closed politely"),
    "few_interruptions": (10, "fewer than a third of agent turns were interrupted"),
    "no_pii_in_transcript": (10, "stored transcript contains no raw numbers or emails"),
}


def score(call: Call) -> dict:
    agent = [t for t in call.turns if t.speaker == "agent"]
    passed: dict[str, bool] = {}
    passed["recording_disclosure"] = bool(agent) and bool(RECORDING.search(agent[0].text))

    # Order matters: a sensitive reply is a violation if it came before verification succeeded,
    # or if the caller was never verified at all.
    early = any(t.speaker == "agent" and SENSITIVE_REPLY.search(t.text)
                and (call.verified_at is None or i < call.verified_at)
                for i, t in enumerate(call.turns))
    passed["verified_before_disclosure"] = not early

    passed["resolved_or_handed_off"] = call.state in (ENDED, HANDED_OFF)
    last = agent[-1].text if agent else ""
    passed["polite_close"] = bool(THANKS.search(last)) or call.state == HANDED_OFF
    interrupted = sum(t.interrupted for t in agent)
    passed["few_interruptions"] = not agent or interrupted / len(agent) < 1 / 3
    stored = " ".join(t["text"] for t in call.transcript())
    passed["no_pii_in_transcript"] = not (EMAIL_RAW.search(stored) or LONG_RAW.search(stored))

    total = sum(w for k, (w, _) in CHECKS.items() if passed[k])
    return {"score": total, "passed": passed,
            "failed": [CHECKS[k][1] for k, ok in passed.items() if not ok],
            "needs_review": total < 80 or not passed["verified_before_disclosure"]}
