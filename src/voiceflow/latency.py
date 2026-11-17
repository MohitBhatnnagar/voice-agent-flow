"""Latency budgeting for voice turns.

In a voice call the silence between the caller finishing and the agent starting to speak is the
experience. What matters is time to first audio: the sum of the stages that must finish before
the first sound plays. Streaming overlaps the rest, so total generation time is not the metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnTiming:
    asr_final_ms: float          # end of speech to final transcript
    llm_first_token_ms: float    # transcript to first generated token
    tts_first_chunk_ms: float    # first text to first audio chunk
    network_ms: float = 0.0

    @property
    def time_to_first_audio(self) -> float:
        return self.asr_final_ms + self.llm_first_token_ms + self.tts_first_chunk_ms + self.network_ms

    def worst_stage(self) -> str:
        stages = {"asr": self.asr_final_ms, "llm": self.llm_first_token_ms, "tts": self.tts_first_chunk_ms,
                  "network": self.network_ms}
        return max(stages, key=stages.get)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, so it is always an observed value."""
    if not values:
        raise ValueError("no values")
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def report(timings: list[TurnTiming], budget_ms: float = 800.0, p95_budget_ms: float | None = None) -> dict:
    if not timings:
        raise ValueError("no timings")
    ttfa = [t.time_to_first_audio for t in timings]
    p95_limit = p95_budget_ms if p95_budget_ms is not None else budget_ms * 1.5
    breaches = [i for i, v in enumerate(ttfa) if v > budget_ms]
    worst: dict[str, int] = {}
    for i in breaches:
        s = timings[i].worst_stage()
        worst[s] = worst.get(s, 0) + 1
    p50, p95 = percentile(ttfa, 50), percentile(ttfa, 95)
    return {"turns": len(timings), "p50_ms": p50, "p95_ms": p95, "max_ms": max(ttfa),
            "breach_rate": len(breaches) / len(timings), "bottleneck": max(worst, key=worst.get) if worst else None,
            "ok": p50 <= budget_ms and p95 <= p95_limit}
