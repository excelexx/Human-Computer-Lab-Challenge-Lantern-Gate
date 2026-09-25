# Interaction reliability and confidence follow-through

The second overnight pass fixes four reproduced lifecycle failures without changing model weights, the response prompt or sampling settings. The local stack still contains **4,464,745,375 required learned parameters**. The first-pass revision `050f9f8b4a8d473459eb7c0be567f246649c1709` and original submission baseline remain preserved.

## What changed and what was actually checked

| Real-model lifecycle check | Before | After |
|---|---|---|
| Cancel while encoders load: explicit terminal without invented emotion | Fail | Pass |
| Cancel after state: do not start generation | Fail | Pass |
| Cancel after first text: preserve partial text and stop | Pass | Pass |
| Close consumer after state | Pass | Pass |
| Close consumer after first text | Pass | Pass |
| Reject a different session's cancel/busy admission | Pass | Pass |
| Accepted cancel at real generator exhaustion stays cancelled | Fail | Pass |
| Release before terminal yield; old iterator cannot unlock new turn | Fail | Pass |

Eight fixed development clips were selected by manifest order before baseline execution, with no label/prediction filtering beyond visual eligibility. Real encoders and Qwen supplied all states and text. One transparent callback requests cancellation when the actual generator exhausts; it is a **scheduling probe**, not natural browser timing or fabricated output. Every case was followed by a successful real recovery turn on the same pipeline: **8/8 before and 8/8 after**. The baseline failures were retained rather than discarded.

The new ownership record has its own cancellation flag and session/turn identity. Ownership is released before terminal events, including errors. Stop and New conversation pass the turn identity. A failed UI attempt retains its composer input but is excluded from the next model history, preventing duplicate retry messages. Cancellation cannot preempt a GPU kernel already running.

The revised run then completed **48/48 consecutive turns: 24 real paired video/text and 24 missing-video text fallbacks**, resetting history every four turns. Across cases, recovery and burst, **62 real generator entries closed successfully**, with zero observed close failures and no owner retained. This was a **43.07-second bounded run**, not hours of continuous operation. Response content is retained without a quality score.

The [comparison](comparison.json), [baseline trace](reliability-baseline/report.json), [current trace](reliability-current/report.json), and adjacent frozen protocols/source snapshots make these checks inspectable. Current counters record successful closes; baseline counters recorded close attempts, with no close error observed. Current CLI failures exit nonzero. A cold worker that cannot finish is explicitly recorded as unresolved, no further turns run, and its isolated CLI process exits; that pipeline must not be reused in-process.

## Context and transport boundaries

Before generation, the same loopback server renders the exact chat template and tokenizes it. The prompt, **96 response tokens and 16 reserve tokens** must fit the active **4,096-token** context. Oldest whole history groups are removed if needed. The current message and emotion evidence are not rewritten. If the current message alone is too long, its genuine emotion state remains available and the app asks for a shorter message before opening a generation request.

Four predeclared [actual local context checks](context-boundary/report.json) passed: a short message (548 prompt tokens), 4,000 emoji rejected, 4,000 repeated CJK characters rejected, and a 2,000-emoji message with long history reduced to whole newest groups (3,449 prompt tokens). The two accepted cases produced real complete responses; overflow cases started no generation request. Request-only instrumentation did not consume or replace responses; an independent local recount verified the exact final submitted prompt. These unnatural strings are tokenizer stress cases, not representative user conversations or emotion-quality evidence.

Normal `stop` finishes end immediately. A `length` finish now reports an incomplete response while retaining partial text. Both preflight JSON and SSE response bodies are bounded to 256 KiB, with 64 KiB line/event limits. The generator has a **90-second operation budget**, checked at chunk boundaries, and a maximum **10-second individual read wait**. A silent read or synchronous operation cannot be interrupted instantly; the operation budget is not a hard end-to-end Stop deadline. GPU computation and model loading remain cooperative cancellation boundaries. Controlled CPU tests cover malformed output, truncation, deadlines, slow/incomplete streams and cleanup errors; those are separate from the successful real-server checks.

## Confidence: keep existing behavior

The [independently verified confidence study](confidence-study/REPORT.md) used 1,109 already-used development examples with dialogue-disjoint cross-fit temperature estimation. Pooled NLL barely changed (1.2138 to 1.2137), while Brier and ECE worsened slightly. Predicted categories did not change. Selecting high-confidence examples also removed rare-class examples, including all three disgust examples from the retained fusion half. **No calibration, threshold or model change was deployed.** Scores remain uncertain classifier outputs, not confidence in a person's actual feelings.

## Updated speed and memory

The same fixed replay latency cohort ran after transport changes: 30 fusion and ten fallback turns after three excluded warmups. This uses the already viewed official test split only as a reused latency benchmark; no labels were scored or model selected. [Protocol](benchmark-protocol.json), [raw measurements](benchmark.json) and [run log](benchmark-log.txt) are retained. Capture, upload/preparation and browser rendering are excluded.

| Warm fusion | Median | p95 |
|---|---:|---:|
| Emotion state | 180 ms | 241 ms |
| First word | 384 ms | 440 ms |
| Completed reply | 518 ms | 604 ms |

Total GPU usage peaked at **7,581 MiB**, including desktop/background activity. Python and generator resident-memory peaks were **3,187 / 3,777 MiB**; system memory peaked at **26,293 MiB**, including other applications. Hardware remains the RTX 3080 10 GB / Windows / Ryzen 7 9700X / 31.10 GiB usable RAM setup. This rerun does not isolate the causal timing cost of token preflight, and normal sampler/background variation can affect results.

During the separate 48-turn burst, Python RSS snapshots ranged from 3,108 to 3,144 MiB; generator RSS stayed at 3,289 MiB. PyTorch allocated/reserved VRAM stayed at 1,694 / 1,860 MiB. These are **post-turn snapshots, not peaks**. Python RSS includes retained report events, and PyTorch memory excludes llama.cpp and desktop. These observations do not establish absence of a slow leak or replace the total-GPU benchmark.

## Software and browser evidence

The final combined CPU software suite passed **252 tests** with one upstream deprecation warning. Results are recorded in [the test log](software-tests.txt) and [XML](software-tests.xml). Artificial test fixtures are contract tests; they do not stand in for model results. Live browser verification is recorded separately in [browser checks](browser-verification.json), including preserved inputs and recovery. Browser cancellation still depends on Gradio event ordering; a bounded check is not proof of every interleaving or multi-browser isolation.

No classifier improvement, visual accuracy benefit, clinical value, or general response-quality improvement is claimed by this second pass. Previously recorded grounding failures remain applicable. Physical camera hardware, long-duration stability and clinical validation remain untested.
