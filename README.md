# voice-agent-flow

The **call-flow engine** behind a voice agent: the dialogue state machine, identity verification before disclosure, barge-in handling, risk-based escalation with a redacted hand-off packet, latency budgeting and transcript QA.

Speech recognition, text-to-speech and telephony belong to a voice platform (for example Smallest AI's voice products or any other provider), and this repo does not integrate with one. It takes text events in and returns what the agent should say, so the part that decides whether a call is **safe and good** can be tested without audio. Standard library only. 37 tests.

```bash
pip install -e .
voiceflow                     # two simulated calls, a latency report and a QA score
python -m unittest discover -s tests -v
```

## The call flow

```
 greeting (+ recording disclosure)
     │ caller speaks
     ▼
 route intent ──no match/low ASR confidence──> reprompt ──> after 2 misses: hand off
     │
     ├── high-risk intent (refund, legal...) ──────────────> hand off immediately
     ├── needs verification ──> ask for code ──> wrong twice ──> hand off
     │                               │ correct
     ▼                               ▼
 answer ──> "anything else?" ──> yes: back to routing   no: goodbye
 
 at ANY point: caller asks for a human, repeated negative sentiment, or a too-long call ──> hand off
```

## Design decisions

1. **Verify before you disclose.** Sensitive replies are only sent after verification succeeds, and the QA check verifies the *order* (it records the turn at which verification happened), not just that verification occurred at some point. A first version judged from the call's final state and could not tell the two apart.
2. **Escalation is a rule, not a model judgement.** Explicit requests for a person, repeated anger, high-risk intents, repeated misunderstanding and runaway calls all hand off by deterministic rules that a compliance reviewer can read.
3. **The human gets a useful, redacted packet.** Reason, detected intent, whether the caller was verified, and the last six turns with numbers and emails replaced. The stored transcript is redacted the same way.
4. **Barge-in is first-class.** When the caller interrupts, the agent's turn is cut to what was actually spoken. After three interruptions the agent switches to shorter prompts, because a caller who keeps talking over you is telling you something.
5. **Budget the silence, not the compute.** In a voice call the experience is the gap before the first sound. `TurnTiming` sums the stages that must finish before first audio (speech-final, first token, first audio chunk, network); streaming hides the rest. `report` gives p50 and p95, the breach rate, and names the **bottleneck stage** so you know what to fix.
6. **Percentiles, not averages.** One 3-second turn among twenty barely moves the mean but is what the caller remembers; the test suite includes exactly that case.
7. **Deterministic QA first.** A weighted rubric (recording disclosure 20, verified-before-disclosure 30, resolved or handed off 20, polite close 10, few interruptions 10, no personal data in the stored transcript 10) flags which calls a human should listen to. It is a filter, not a verdict.

## Plugging in a voice platform

Feed final transcripts and their ASR confidence to `Call.on_user(text, confidence)`, speak the returned `Action.text`, and call `Call.on_barge_in(spoken_part)` when the platform reports that the caller interrupted. On a `handoff` action, transfer the call and give the human `Action.packet`. Measure your own stage latencies into `TurnTiming`.

## Limits

- Intent routing is keyword matching. A production agent would use an LLM or classifier behind the same `Intent` interface, with these rules still wrapping it.
- Verification code is a demo constant; real systems verify against a backend and also rate-limit by caller.
- Sentiment is a word list, not acoustic or semantic analysis.
- Not tested against any speech provider or telephony stack.
- Calls in several languages, DTMF input and call recording consent by jurisdiction need their own handling.

Related: [`agentflow-hitl`](https://github.com/MohitBhatnnagar/agentflow-hitl) (approval gates), [`ai-governance-framework`](https://github.com/MohitBhatnnagar/ai-governance-framework), [`ai-quality-and-evals`](https://github.com/MohitBhatnnagar/ai-quality-and-evals) (calibrating QA reviewers).

## License

MIT
