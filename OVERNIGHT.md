# Overnight improvement and backtesting log

User authorization: continue improving the local prototype while the user sleeps, with substantial multimodal backtesting. Work began 2026-09-25 around 05:05 UTC. Scheduled follow-through runs every 30 minutes until 13:00 UTC (09:00 Toronto); then summarize and pause automation `overnight-multimodal-backtesting`.

## Fixed constraints

- Fully local text + vision only; no audio, RL, physical robot, paid compute, or remote inference.
- At most six billion total required learned parameters, including every frozen component.
- Windows RTX 3080 10 GB / 32 GB RAM. Coordinate GPU use; preserve a working app for handoff.
- Preserve baseline commit `a7aa0a768130a90fef5bf9e85ff8315a9d81d65d`, its supplied heads, portable archive, and measured reports.
- Model selection uses training cross-validation and development data. Official test results were already inspected in the initial prototype; any later test evaluation is a **reused benchmark**, never a new untouched holdout. No candidate selection from test scores.
- Record plans before experiments, keep failures and null results, and distinguish numerical checks, synthetic interventions, AI qualitative review, and human/clinical validation.

## Current parallel work

1. `upgrade_perception`: CPU-only regularized linear classification and fixed late-fusion study using frozen cached features; train dialogue-group cross-validation, small predefined hyperparameter grid, development selection, frozen decision before one reused-test evaluation. New module `model_study.py`; no production checkpoint changes without root review.
2. `build_ui`: lossless removal of audio when video codec/container allow copying, explicit conversion fallback, real decoded-frame parity checks, and prevention of accidental stale-clip reuse. Owns `app.py`, new `video_input.py`, UI/video tests. No shared encoder/processor fingerprint changes.
3. `upgrade_generator`: fixed-seed controlled emotion-evidence response study with development and reserved authored cases; compare baseline and two predefined prompt candidates, around 144 local generations. Record actual MELD dev states separately from synthetic evidence interventions and retain grounding failures. Owns `generator.py`, new `response_backtest.py`, associated tests.
4. Root: video perturbation robustness and repeated mismatched-modality backtests; integrate verified improvements, regressions, resource audit and updated package. New `robustness.py` and tests.

## Root robustness protocol (declared before results)

Use a class-stratified, seed-20260925 sample of at most 16 visually eligible **development** utterances per emotion. Retain all available rare-class examples. Compare the same original text with original video and seven predefined transformations: lossless silent remux, legacy-style H.264 re-encoding, half-resolution video, half brightness, Gaussian blur, horizontal flip, and a black-frame control. Process variants through the unchanged YuNet/vision encoder and current heads, with explicit text fallback when vision becomes unavailable. Reuse the exact cached text features so only the visual input changes. Record selected IDs, transformation settings, source and checkpoint hashes, coverage, label flips, distribution shifts, class metrics, and per-example outcomes. Transformations are stress tests, not a representative webcam benchmark or a proof that every altered image retains the same expression.

Also run twenty seeded permutations of paired vision plus visual-quality features across all eligible development examples; preserve each text/label pair. Quantify how often mismatched evidence changes the output, alongside real missing-vision fallback. Dialogue-cluster bootstrap comparisons will describe uncertainty in existing/reused evaluation, not select a new model. No generated or transformed MELD videos are included in the portable repository.

## Resume instructions

Read `work/overnight/status.json`, agent messages, and study protocols/results first. Do not duplicate an active study or repeat completed training. The production app may be stopped briefly for GPU work or restarted to load validated changes. Keep the generator available while the response agent is using it. Promote only justified changes, retain baseline reports, and update README/REPORT, hashes, parameter totals and the portable ZIP after final verification.

Detailed results will be appended as each study completes. An interrupted or sleeping machine creates an execution gap; do not claim continuous overnight computation.

## First improvement batch completed (2026-09-25T05:46:52.737693+00:00)

- 72 dialogue-group CV fits, three final classifier fits and a fixed fallback-only hybrid. New linear text head is activated; old vision/fusion remain unchanged. Total 4,464,745,375 required parameters. Baseline commit/ZIP/heads are preserved.
- 95 dev clips Ã— eight real video conditions completed without extraction errors. Twenty visual-evidence permutations and a provenance-checked CPU comparison retained null results and regressions. No visual accuracy advantage is established.
- 204 local response-study generations, then 40 measured benchmark turns and 12 additional actual multi-turn checks. Prompt improvements are bounded and qualitative; harmful assumptions remain recorded.
- Lossless compatible video muting, single-turn clip use, locked composer, mixed-head loader/inventory, reproducible study preparation and explicit cache requirements implemented.
- Final CPU suite: 185 passed, one dependency deprecation warning. Clean head installation, parameter audit and live browser fusionâ†’text-fallback sequence passed. Browser replay still misclassifies 4/2 and presumes excitement; no cherry-picking or retuning on that test case.
- Updated warm fusion p95 state/first-token/completion: 246/458/623 ms. GPU peak 7,566 MiB including desktop. Full details: [REPORT.md](REPORT.md). Source checkpoint and portable package are refreshed after this record.

## Follow-through priorities for subsequent heartbeats

The first batch is finished: do not repeat these studies or change their frozen artifacts. Read the current status and review before starting anything. Useful next work is a bounded confidence/calibration diagnostic using the current saved predictions, with train/dev-only decisions and a protocol written before fitting, or an actual long-running reliability check covering cancellation, session isolation and resource growth. New synthetic tests must be labeled. Preserve the working app and full parameter accounting; avoid loading another full GPU pipeline alongside the warmed app. Do not chase a favorable test result, repeatedly tune the prompt on exposed failures, or run duplicate inference merely to increase a test count. If a further change cannot be justified, retain the measured working version and focus on auditability and the morning handoff. At 13:00 UTC complete the bounded step, refresh the package, report supported results, and pause the existing heartbeat.

## Second pass declared (2026-09-25 around 05:52 UTC)

This pass resumes from first-batch revision `050f9f8b4a8d473459eb7c0be567f246649c1709`; its complete ZIP and package record are preserved in `work/overnight/first-batch`. It does not repeat model selection or tune the prompt on exposed response failures.

- Five dialogue-group cross-fit confidence diagnostics use development predictions only, with route-specific temperature fits on the other folds. Reused development data cannot become a fresh holdout. No learned calibration is deployed unless justified; no selective threshold is chosen from this diagnostic.
- Eight predeclared lifecycle cases use real development clips and local generation: cancellation during cold loading, after state, after the first delta, closing after state/delta, wrong-session/busy admission, cancellation at real generator exhaustion, and ownership at a terminal yield. The exhaustion callback is an explicit scheduling probe, not fabricated model output. Baseline evidence precedes production edits.
- Repeat those fixed cases after justified fixes, followed by 48 consecutive alternating paired-video and text-only turns with four-turn histories. Measure process memory, GPU allocator memory and handle counts; a brief burst does not establish hours of stability or absence of slow leaks.
- Check the exact local tokenizer boundary using four authored cases: a short message, 4,000 emoji, 4,000 CJK characters, and a current message plus long history requiring whole-turn eviction. Retain actual state on rejection; no fake completion, remote fallback or silent current-message rewriting. These are transport/context checks, not emotion accuracy or therapeutic-quality tests.
- Run one updated backend benchmark after transport changes, focused CPU fault/race checks, and live browser stop/reset/recovery verification. Preserve failures, protocols, model/source hashes and operational limits. Leave the app usable and refresh the portable package.

One GPU workload runs at a time. The app is temporarily stopped while direct reliability checks own the encoders; the already running local generator is shared sequentially. Read `work/overnight/status.json` before another pass so studies are not duplicated.

## Second pass completed (2026-09-25T06:26:47.774937+00:00)

- Preserved a real-model baseline with four failures out of eight lifecycle cases. After ownership/cancellation fixes, all eight cases and eight recovery turns passed, followed by 48/48 alternating fusion/fallback turns. All 62 generator entries closed successfully. The bounded run lasted 43.07 seconds; it is not continuous overnight stability evidence.
- Four actual local context checks passed, including preserved state/input on overflow and whole-history eviction. Transport now bounds preflight and SSE data, distinguishes length limits from completion, and has documented cooperative cancellation limits. Prompt, sampler, models and 4,464,745,375 parameter count are unchanged.
- The independently verified development confidence diagnostic did not justify calibration; no calibration or threshold was deployed. All previous classifier/response evidence and limitations remain applicable.
- Final software suite: 252 passed, one dependency warning. Live browser Stop, New during cold loading, preserved overflow input, clearer main-status guidance, clean retry and final fresh conversation were observed. Multi-browser queue isolation and physical camera remain untested.
- Updated warm fusion p95 state/firstword/completion: 241/440/604 ms; peak total GPU 7,581 MiB. No causal speed-improvement claim. Full report: [interaction reliability](reports/reliability/REPORT.md).

### Next heartbeat

Do not rerun completed training, confidence, corruption, response or short-burst studies. The app is warm and usable; both study batches and source checkpoints are saved. Further work should address a concrete new issue or an auditability gap, not inflate inference/test counts. If no justified change remains, preserve this version and stay quiet until the morning handoff. The 48-turn burst cannot support long-duration claims. At 13:00 UTC complete a bounded in-progress step, verify package/app status, summarize supported outcomes and weaknesses, then pause the existing heartbeat. A sleeping machine or inactive interval is a recorded gap, not continuous execution.

## Third bounded pass completed (2026-09-25T07:21:25.366652+00:00)

- A concrete delayed replay/reset overwrite was reproduced in CPU scheduling and the actual browser before editing. The unchanged synthetic delay was then used with the fixed UI: New and Stop preserved fresh drafts, and ordinary loading restored controls.
- Queued replay admission, per-browser epoch/ticket ownership, cancellation and composer locking prevent the reproduced stale callback. Already-dispatched frontend updates remain a documented Gradio transport boundary.
- Full software suite: 268 passed, one dependency warning. A production MELD 4/2 text+video turn used fusion and completed a local response. Its wrong anger tag and presumptive reply remain documented. Models, prompt, sampler and 4,464,745,375 parameter count are unchanged; no new accuracy or speed claim.
- Both synthetic services exited; the real app is warm and ready for a fresh conversation. See [the replay lifecycle report](reports/replay-lifecycle/REPORT.md). Source/ZIP are refreshed after this entry.

Do not duplicate these three completed study batches. Subsequent passes should address only a concrete new issue or an auditability gap; otherwise preserve the app and stay quiet. Inactive intervals are not continuous computation. At 13:00 UTC, verify the handoff and pause the existing heartbeat as already instructed.

## Morning handoff (25 September 2026, after 13:00 UTC)

The authorized overnight window has ended. All three study batches are complete; no training or GPU experiment remains active. Subsequent 30-minute passes found no new justified change and recorded clean source plus healthy app/generator responses. Those intervals were inactive apart from health checks, not continuous computation or inference stability tests.

Final verification confirmed the prior package's archive checks and 416 source-file hashes, matching source revision `705eb8fc1b461c855bfb58df1a90e32fcd3187fa`, consistent parameter inventories, and healthy loopback services. Independent review confirmed the reported scores, counts and retained failures. The final package is refreshed only to add this record and [HANDOFF.md](HANDOFF.md); inference code, weights and measured outcomes are unchanged. The existing heartbeat was paused after packaging, and the local app is left running.
