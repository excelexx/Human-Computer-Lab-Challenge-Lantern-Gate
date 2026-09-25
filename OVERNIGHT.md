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
- 95 dev clips × eight real video conditions completed without extraction errors. Twenty visual-evidence permutations and a provenance-checked CPU comparison retained null results and regressions. No visual accuracy advantage is established.
- 204 local response-study generations, then 40 measured benchmark turns and 12 additional actual multi-turn checks. Prompt improvements are bounded and qualitative; harmful assumptions remain recorded.
- Lossless compatible video muting, single-turn clip use, locked composer, mixed-head loader/inventory, reproducible study preparation and explicit cache requirements implemented.
- Final CPU suite: 185 passed, one dependency deprecation warning. Clean head installation, parameter audit and live browser fusion→text-fallback sequence passed. Browser replay still misclassifies 4/2 and presumes excitement; no cherry-picking or retuning on that test case.
- Updated warm fusion p95 state/first-token/completion: 246/458/623 ms. GPU peak 7,566 MiB including desktop. Full details: [REPORT.md](REPORT.md). Source checkpoint and portable package are refreshed after this record.

## Follow-through priorities for subsequent heartbeats

The first batch is finished: do not repeat these studies or change their frozen artifacts. Read the current status and review before starting anything. Useful next work is a bounded confidence/calibration diagnostic using the current saved predictions, with train/dev-only decisions and a protocol written before fitting, or an actual long-running reliability check covering cancellation, session isolation and resource growth. New synthetic tests must be labeled. Preserve the working app and full parameter accounting; avoid loading another full GPU pipeline alongside the warmed app. Do not chase a favorable test result, repeatedly tune the prompt on exposed failures, or run duplicate inference merely to increase a test count. If a further change cannot be justified, retain the measured working version and focus on auditability and the morning handoff. At 13:00 UTC complete the bounded step, refresh the package, report supported results, and pause the existing heartbeat.
