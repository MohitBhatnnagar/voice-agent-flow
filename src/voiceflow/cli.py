"""Demo: two simulated calls, a latency report and a QA score (no audio, no provider)."""
from __future__ import annotations

import re
import sys

from .flow import Call, Intent
from .latency import TurnTiming, report
from .qa import score


def intents() -> list[Intent]:
    return [
        Intent("order_status", re.compile(r"\b(order|delivery|parcel)\b", re.I), needs_verification=True,
               reply=lambda c: "Your order shipped yesterday and arrives tomorrow."),
        Intent("opening_hours", re.compile(r"\b(hours|open|opening)\b", re.I),
               reply=lambda c: "We are open from nine to six, Monday to Saturday."),
        Intent("refund", re.compile(r"\brefund\b", re.I), high_risk=True),
    ]


def main(argv: list[str] | None = None) -> int:
    c = Call(intents())
    print("agent:", c.start().text)
    for said in ("Where is my order?", "my code is 1 2 3 4", "no thanks"):
        a = c.on_user(said)
        print("user :", said, "\nagent:", a.text)
    print("QA:", score(c)["score"], "\n")

    c2 = Call(intents())
    c2.start()
    a = c2.on_user("I want a refund, my email is sam@example.com")
    print("handoff:", a.packet)
    print("\nlatency:", report([TurnTiming(180, 320, 140), TurnTiming(200, 280, 150), TurnTiming(220, 900, 160)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
