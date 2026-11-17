"""Call-flow engine for a voice agent, independent of any speech or telephony provider.

The speech layer (speech-to-text, text-to-speech, telephony) is a provider concern and is passed in
as text events. This engine owns what is hard to get right and independent of the vendor:
the dialogue state machine, identity verification before disclosure, barge-in, repeated-failure
and risk-based escalation, and a redacted hand-off packet for the human who takes over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

GREETING, VERIFY, HANDLE, MORE, ENDED, HANDED_OFF = "greeting", "verify", "handle", "more", "ended", "handed_off"

ESCALATE_WORDS = re.compile(r"\b(human|agent|representative|real person|operator|supervisor|manager)\b", re.I)
NEGATIVE = re.compile(r"\b(angry|furious|ridiculous|useless|terrible|awful|unacceptable|complaint|sue|lawyer)\b", re.I)
YES = re.compile(r"\b(yes|yeah|yep|sure|please|another)\b", re.I)
NO = re.compile(r"\b(no|nope|nothing|that's all|thats all|that is all|goodbye|bye)\b", re.I)
DIGITS = re.compile(r"\d")
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
LONG_NUMBER = re.compile(r"\b(?:\d[ \-]?){4,}\b")


def redact(text: str) -> str:
    return LONG_NUMBER.sub("[NUMBER]", EMAIL.sub("[EMAIL]", text))


@dataclass
class Intent:
    name: str
    pattern: re.Pattern
    needs_verification: bool = False
    high_risk: bool = False     # always goes to a person
    reply: Callable[[dict], str] = lambda ctx: "Done."


@dataclass
class Config:
    disclosure: str = "This call may be recorded for quality."
    verify_code: str = "1234"            # demo only: real systems verify against a backend
    max_verify_attempts: int = 2
    max_no_match: int = 2
    max_negative: int = 2
    min_confidence: float = 0.5
    max_turns: int = 20


@dataclass
class Action:
    kind: str            # "say", "handoff" or "end"
    text: str = ""
    packet: dict | None = None


@dataclass
class Turn:
    speaker: str         # "agent" or "user"
    text: str
    interrupted: bool = False


class Call:
    def __init__(self, intents: list[Intent], cfg: Config | None = None):
        self.intents, self.cfg = intents, cfg or Config()
        self.state = GREETING
        self.turns: list[Turn] = []
        self.verified = False
        self.verified_at: int | None = None   # index in self.turns where verification succeeded
        self.pending: Intent | None = None
        self.last_intent: str | None = None
        self.no_match = self.bad_verify = self.negative = 0
        self.barge_ins = 0
        self.terse = False
        self.user_turns = 0
        self.ctx: dict = {}

    # -- agent speech ----------------------------------------------------------
    def _say(self, text: str) -> Action:
        if self.terse and ". " in text:
            text = text.split(". ")[0] + "."   # callers who keep interrupting get shorter prompts
        self.turns.append(Turn("agent", text))
        return Action("say", text)

    def start(self) -> Action:
        return self._say(f"Hello. {self.cfg.disclosure} How can I help you today?")

    # -- events from the speech layer ---------------------------------------------
    def on_barge_in(self, spoken_part: str) -> None:
        """The caller started talking while the agent was speaking. Keep only what was actually heard."""
        for t in reversed(self.turns):
            if t.speaker == "agent":
                t.text, t.interrupted = spoken_part, True
                break
        self.barge_ins += 1
        if self.barge_ins >= 3:
            self.terse = True

    def on_user(self, text: str, confidence: float = 1.0) -> Action:
        if self.state in (ENDED, HANDED_OFF):
            raise RuntimeError("call is over")
        self.user_turns += 1
        self.turns.append(Turn("user", text))
        if self.user_turns > self.cfg.max_turns:
            return self._handoff("call is too long")
        if ESCALATE_WORDS.search(text):
            return self._handoff("caller asked for a person")
        if NEGATIVE.search(text):
            self.negative += 1
            if self.negative >= self.cfg.max_negative:
                return self._handoff("caller is upset")
        if self.state == VERIFY:
            return self._on_verify(text)
        if self.state == MORE:
            return self._on_more(text)
        return self._on_request(text, confidence)

    # -- state handlers --------------------------------------------------------------
    def _on_request(self, text: str, confidence: float) -> Action:
        intent = next((i for i in self.intents if i.pattern.search(text)), None)
        if intent is None or confidence < self.cfg.min_confidence:
            self.no_match += 1
            if self.no_match > self.cfg.max_no_match:
                return self._handoff("could not understand the request")
            return self._say("Sorry, I did not catch that. Could you say it another way?")
        self.no_match = 0
        self.last_intent = intent.name
        if intent.high_risk:
            return self._handoff(f"high-risk request: {intent.name}")
        if intent.needs_verification and not self.verified:
            self.pending, self.state = intent, VERIFY
            return self._say("Before I can share that, please say your four digit verification code.")
        return self._answer(intent)

    def _on_verify(self, text: str) -> Action:
        digits = "".join(DIGITS.findall(text))
        if digits == self.cfg.verify_code:
            self.verified, self.verified_at = True, len(self.turns)
            return self._answer(self.pending)
        self.bad_verify += 1
        if self.bad_verify >= self.cfg.max_verify_attempts:
            return self._handoff("verification failed")
        return self._say("That code did not match. Please try once more.")

    def _answer(self, intent: Intent) -> Action:
        self.state = MORE
        return self._say(intent.reply(self.ctx) + " Is there anything else I can help with?")

    def _on_more(self, text: str) -> Action:
        if NO.search(text) and not YES.search(text):
            self.state = ENDED
            self.turns.append(Turn("agent", "Thank you for calling. Goodbye."))
            return Action("end", "Thank you for calling. Goodbye.")
        self.state = HANDLE
        return self._on_request(text, 1.0)

    # -- escalation ------------------------------------------------------------------
    def _handoff(self, reason: str) -> Action:
        self.state = HANDED_OFF
        msg = "I will connect you with a colleague now. Please hold."
        self.turns.append(Turn("agent", msg))
        return Action("handoff", msg, self.packet(reason))

    def packet(self, reason: str) -> dict:
        """What the human receives: why, who, what was asked, and a REDACTED recent transcript."""
        return {"reason": reason, "intent": self.last_intent, "verified": self.verified,
                "summary": [f"{t.speaker}: {redact(t.text)}" for t in self.turns[-6:]]}

    def transcript(self) -> list[dict]:
        """Redacted transcript safe to store."""
        return [{"speaker": t.speaker, "text": redact(t.text), "interrupted": t.interrupted} for t in self.turns]
