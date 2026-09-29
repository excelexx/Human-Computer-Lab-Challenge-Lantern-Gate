# Dialogue and emotion matrix

## Scope declared before generation

Compare the committed `b19e7674a46279c9aa350838be9507a34190faa4` baseline with fixes using the same local Qwen model and case set. Preserve both sets of outputs, including failures. The original plan held sampling fixed; after a separately retained paired sampling probe, the delivered run changes temperature from 0.5 to 0.0. The final comparison therefore measures the combined changes, not an isolated prompt effect. Matrix runs use seed 314. Use the production prompt builder and token-budget check. Stop the camera application during generation so there is one GPU workload, then restore it.

The finite stage matrix covers all four offered lines and all seven MELD emotions at the opening, route selection, and readiness for each selected route: **112 generated cases**. A separate custom matrix covers questions, refusal, explicit feelings, route changes, ambiguity, identity and attempts to override the scene. These use controlled synthetic emotion evidence, not newly measured webcam predictions. Authored prior dialogue is a fixed stage fixture; it is not presented as a generated conversation.

All **21,952** combinations of the three preset choices and their three emotion cues are checked through the deterministic game and direction policy. This is exhaustive three-turn action coverage, not 21,952 generated conversations. Arbitrary custom text, indefinite refusal loops and all stochastic language-model outputs cannot be exhaustively enumerated. Actual generated multi-turn journeys supplement the fixed-history matrix.

Assess whether replies answer the words in context, honor refusal and explicit intent, preserve route facts and the game's current action, and vary delivery without asserting an emotion as fact. Flag formatting, invented events, premature arrival, unsupported safety promises and route contradictions. Automated flags identify candidates for semantic review; they are not a claim of human judgment or universal response quality. Keep remaining weaknesses visible. No MELD test-set tuning, retraining or parameter-count changes are involved.

## Coverage

Each complete generation run contains **1,157 real local Qwen responses**:

| Cases | Count | Coverage |
|---|---:|---|
| Presets | 112 | Four options × seven emotions at opening, route selection, and readiness for both routes |
| Custom text | 924 | 33 authored messages × four stage/route contexts × seven emotions |
| Evidence policy | 49 | 42 conflicting text/vision interventions and seven missing-camera cases |
| Exploratory | 56 | Eight quotation, correction, space, identity/perception and continuity challenges × seven emotions |
| Actual dialogue history | 16 | Two three-turn journeys and two five-turn refusal/recovery journeys, using actual preceding model responses |

The same custom set is used at each stage deliberately: asking about a bridge before selecting one and asking after selecting stairs must receive different context-appropriate treatment. The set includes direct questions, smart-apostrophe negation, hypothetical plans, refusing one route while selecting another, ambiguous alternatives, route changes, explicit feelings, pauses and requests to leave the fiction.

The separate deterministic checks cover **21,952 preset paths** and **2,352 evidence-routing configurations**. A separately authored action audit checks the 924 custom rows (132 unique text/stage pairs, repeated under seven cues). These checks test state and policy, not the quality of generated prose.

## Changes

- The quest parser separates a route preference from consent to move. Negation, quoted commands, hypothetical plans and questions cannot accidentally start walking. “Don't worry” does not mean “stop,” and being unready does not erase a selected route.
- Rejecting the current route clears that plan; choosing another route updates it. An explicit “other route” request switches an established plan and asks readiness. With no existing plan it requests clarification. Presets still take three turns; an early custom route selection followed by explicit readiness can finish in two.
- The reply goal now answers the current question before advancing the quest, honors pauses, and supplies focused facts for identity, perception, route conditions, the beacon task and route changes. Goals are authored instructions/facts, not finished NPC responses.
- Explicit-feeling policy ignores quotations, hypothetical statements and disowned descriptions. Clear current corrections override earlier claims; mixed claims cause the rule to abstain. This is a bounded dialogue policy, not an additional trained classifier.
- Greedy decoding replaces temperature 0.5. A 112-case paired development probe weakly favored it for reducing elaborate inventions; 47 pairs were identical. It is not uniformly better, and does not guarantee identical outputs across hardware/server changes.
- The resumable runner fingerprints sources, cases and sampling; preserves incomplete writes; and stops a generated journey on empty, truncated or failed replies. The baseline's 1,157 replies all finished normally, so the earlier runner's weaker completion check did not affect its observed journey results.

The learned stack and original MELD results are unchanged: **4,464,745,375 total required learned parameters**. No training, test-set selection or additional learned model was introduced.

## Development evidence

`baseline-run` records revision `b19e767`; `candidate-run` is a rejected broad prompt rewrite; `selected-run` is an intermediate targeted revision; `final-run` is the delivered revision. The failed broad rewrite improved pauses but introduced scenery, role confusion and weaker identity answers, so it was rejected. All runs remain in the archive. The 224 sampling-probe outputs are also retained.

The baseline failed **175/924** custom action expectations. The targeted revision reduced this to **0/924**. All **24,304** preset-path and evidence-policy checks passed both before and after; this preserves prior behavior rather than establishing an improvement in those checks. Passing these contracts does not mean every generated sentence is correct.

These are development cases used to find and fix problems. They are **not a fresh holdout**, a human-rated quality study, camera-accuracy evidence or a clinical evaluation. Code agents reviewed the replies for meaning, consent, continuity, known facts, emotional delivery and presentation. The reviews retain exact IDs and uncertainty; automated keyword flags are only review aids. Fixed canonical histories do not exhaust all possible generated histories, and one seed does not exhaust stochastic outputs.

## Reproduction

Use the installed repository environment and the local model server on port 8081. Stop the camera/app workload while benchmarking; `scripts/start-generator.ps1` can run the generator alone. From the repository:

```powershell
python scripts/evaluate-dialogue-matrix.py --output work/my-dialogue-run --kind all
python scripts/audit-dialogue-actions.py work/my-dialogue-run/responses.jsonl --output work/my-dialogue-run/custom-actions.json
```

The same command resumes only when fingerprints match. A changed source, case set or sampling configuration requires a new output directory. `--limit 5` makes a small smoke run; `--sections presets` selects the preset stage matrix; `--kind checks` performs the CPU-only finite checks. Never describe a partial run as full coverage. Full prompts, source hashes, actual outputs, finish reasons, timings and server identity are preserved in compressed JSONL; decompress with Python's `gzip` module. A readable final CSV accompanies the raw transcripts.

## Hardware and limits

Windows 11, Python 3.12.10, RTX 3080 with 10,240 MiB VRAM, and 31.1 GiB observed physical RAM. A generator-only GPU allocation snapshot was 5,680 MiB including desktop use; this is not a peak or full multimodal-path measurement. Timings include local prompt preparation/token-budget checks and generation, but exclude camera capture, encoders and browser rendering. Earlier full-path resource evidence remains in the project reports.

The game grammar intentionally handles a bounded set of clear commands, not unrestricted natural-language intent. The language model can still repeat examples, make weak tone distinctions, add unsupported details, or misunderstand unfamiliar custom text. In particular, factual answers and departure lines can be nearly identical across emotions; correctness takes precedence over forcing seven different phrasings. No claim of universal coherence is made.

## Final results — September 29, 2026

All **1,157 final responses completed without transport or truncation errors**, and all have an associated semantic development review. Of these, **579 changed responses or actual journey turns were newly read**; 578 inherited the earlier review only after exact response, input, evidence/state, direction and history matches. [Review coverage](matrix/coverage.json) maps each case ID to its review file. This is an agent review with explicit residual failures, not a universal pass or independent quality score.

The final custom action audit has **0 failures / 924 cases**, compared with **175 / 924** in the committed baseline. All **21,952 preset paths and 2,352 policy configurations** still pass. The four actual-history journeys preserve route choice, waiting and the distinction between planning to walk and having arrived. This is four generated journeys, not every possible generated conversation history.

Concrete improvements observed in the revised replies include honoring “Not yet,” clarifying quoted offers without treating them as commands, switching an existing route only when requested, answering the local-AI identity question, and treating a corrected current feeling as more relevant than a quoted or retracted feeling. The original nonsensical arrival/storm banter is replaced by replies about the current beacon task in the tested opening examples.

The final review still found substantive failures. Representative cases, all retained in the [readable response table](matrix/delivered-responses.csv), include:

| Case ID | Remaining problem |
|---|---|
| `preset/opening/1/disgust` | “The bridge is windy, not safer” makes an unsupported comparison. |
| `custom/opening/explicit_fear/joy` | Infers bridge height from the player's stated fear of heights. |
| `custom/route/will_not/surprise` | The prose selects stairs after the player only rejects the bridge; the actual game state correctly leaves the route undecided. |
| `custom/route/conditional/sadness` | Says neither route is sheltered, contradicting the established stairs. |
| `custom/ready_bridge/explicit_disgust/neutral` | Invents that the bridge is not damp. |
| `custom/ready_stairs/why/disgust` | Dismisses an estimated emotion despite the player not declaring that feeling. |
| `exploratory/hypothetical_feeling/joy` | Does not answer the hypothetical question about how a different cue affects Mara. |

These limits remain unresolved in the delivered model. The final change is justified by the measured consent/state fixes and specific dialogue corrections, not a claim that every wording change helped. In the two reviewed route/bridge custom blocks, substantive issues fell from 30 to 22 under the same agent's rubric, but new errors also appeared. We did not keep tuning until these development cases appeared perfect or substitute fabricated passing outputs. The camera models and honest MELD benchmark results remain unchanged.

### Timing and verification

| Run | Responses | Median | p95 | Total generation time |
|---|---:|---:|---:|---:|
| Committed baseline | 1,157 | 197.9 ms | 278.7 ms | 232.06 s |
| Rejected broad rewrite | 1,157 | 213.9 ms | 324.7 ms | 253.05 s |
| Intermediate targeted revision | 1,157 | 179.3 ms | 268.1 ms | 215.04 s |
| Delivered revision | 1,157 | 182.8 ms | 277.2 ms | 219.35 s |

These are warm local-server matrix timings with repeated shared prompt prefixes and short replies; they are **not end-to-end camera latency or cold-start guarantees**. The app was stopped during the matrix. [Timing summary](matrix/timing-summary.json) retains the measurements.

An additional [eight-case production streaming check](matrix/streaming-smoke.json) completed without errors: first text delta 212–452 ms and complete replies 280–562 ms, excluding camera and UI. Seven responses matched their matrix counterparts exactly; one differed in wording while preserving the selected stairs and clarifying the quoted bridge offer. Greedy decoding does not establish byte-identical output across request modes.

The final delivered source passed **589 Python tests** (one existing dependency deprecation warning). The game/camera code also passed **13 JavaScript tests** and was not changed afterward. The app and local model server were restarted/verified after generation. No UI screenshot or new real-webcam accuracy test is claimed for this pass.

### Saved evidence

The saved matrix evidence includes complete baseline, rejected, intermediate and final transcripts compressed as JSONL, metadata, action audits, exact review findings, software test output, source snapshots and the sampling probe. [Transcript integrity hashes](matrix/transcript-integrity.json) cover both compressed and decompressed bytes; [source archive hashes](matrix/source-archives.json) identify the snapshots. There are 4,852 recorded development generations across four full runs and the paired sampling probe, eight production streaming replies, and the 70 supplementary generations described below: **4,930 total generations, not independent validation examples**.

The delivered evaluator receives additional resume/completeness validation after the final generation run. The archived [final-source.zip](matrix/final-source.zip) preserves the exact runner and production sources that produced that measured matrix. The corpus remains unchanged. The final run is not silently relabeled as having used a later runner or the additional quest correction below.

### Final code-review correction and targeted verification

The last review found a deterministic conflict that the original custom set missed: readiness combined with a request for space or a change of subject could trigger walking while the dialogue goal said to wait. Shared intent rules now make those requests pause movement. Conversely, adding “stay close” to a confirmed departure no longer gives the generator a contradictory instruction to stay at the gate.

All **1,157 saved inputs and exact histories** were rebuilt with this correction. [The context comparison](matrix/post-review-context-diff.json) found **1,129 identical prompts/states/actions** and **28 changed prompts**, all the existing outside-game requests; none of those saved game actions changed. Those 28 cases were regenerated and read again, and all acknowledge pausing. The original full run remains intact. [Delivered cases](matrix/delivered-cases.jsonl.gz) and the readable CSV combine the 1,129 unchanged cases with the 28 newly generated ones, with an `evidence_origin` on each row. This is a provenance-tagged composite, not another full-run latency measurement.

An additional **42 local generations** cross three combined requests with both readiness routes and all seven cues. All 42 meet their independently specified action expectation, and all were read. Added companionship produces a coherent departure on the requested bridge. Space/topic requests correctly pause the game; a few replies still mention being ready to go after granting space. These wording failures remain in the [supplementary review](matrix/reviews/post-review-review.json), rather than being counted as semantic passes.

There are therefore **1,199 distinct development scenarios** in the original matrix plus this supplement. The delivered composite still passes **924/924 custom action contracts**, with explicit completeness validation. The hardened runner also passes all **24,304** CPU checks, rejects changed cached fixtures and reports partial coverage as partial. [Delivered sources](matrix/delivered-source.zip) preserve the last correction separately from the original final-run snapshot. No further prompt tuning was performed after this supplement.
