# Controlled response-grounding study

150 actual local Qwen generations were completed: 126 development responses, six identical-request repeat controls, and 18 reserved authored responses. The compact `facts_first` prompt was selected using development responses before any reserved output was generated, then copied exactly into `generator.py`. Weights and production sampling are unchanged.

A subsequent, separately declared stress replication ran **54 more generations**, for **204 total**. It uses the exact production sampler (temperature 0.5, top_p 0.8, top_k 20, max 96) with seeds 42/43/44 over the same six already revealed reserved cases and three states. This is not a fresh holdout, baseline comparison, or additional selection stage. `production-replication-protocol.json`, `responses-production-replication.jsonl`, and `review-production-replication.json` preserve that extension; `production_replication.py` reproduces/resumes it without changing the prompt. The extension had no request errors or truncation, but one racing-heart reply suggested a “normal reaction” and another questioned whether the experience was “really just physical.” Other contact, grief and mixed-feeling limitations persisted. No further prompt tuning was performed.

Read `study-report.json`, `review-dev.json`, and `review-check.json` for the evidence and retained failures. `responses-*.jsonl` preserve every response, actual timing, state, request hash, finish reason, and narrow pattern match. `protocol.json` contains the complete predeclared cases, split, candidate prompts, sampler, source/model/checkpoint hashes, and exact runtime metadata. `selection.json` records the development-only selection and the exact development output hash.

The selected prompt improves concrete grounding on the ambiguous balancing fragment, actual MELD dev guitar utterance, and some unwanted-promotion conditions. It does not reliably resolve grief fabrication, presumption that an event is welcome, or mixed-feeling omissions. The six reserved cases passed every narrow regex check but still show qualitative failures. This is not clinical validation or a numeric response-quality score.

Exact response changes across states cannot alone demonstrate useful visual conditioning: two of six identical requests changed despite temperature zero and seed 42. Some state differences were punctuation, while others were harmful, such as joy-driven congratulations for an explicitly unwanted responsibility. Authored states are synthetic generator-level interventions, never emotion ground truth. Real states come only from MELD dev caches and trained heads; no test split was opened.

The local runtime parameter upper bound remains **4,464,870,303**, below 6,000,000,000. The study's warm generator completion median was 217.91 ms and p95 468.17 ms. These exclude image capture, encoders, classification and browser rendering, and use temperature zero rather than production's unchanged 0.5.

## Reproduction

From an activated repository environment, with the pinned local model server running and matching trained heads/caches available:

```powershell
python -m checkin.response_backtest prepare --home <runtime-home> --out <new-study-directory>
python -m checkin.response_backtest dev --out <new-study-directory>
```

Review all development replies qualitatively, then create `selection.json` with `protocol_sha256`, `dev_responses_sha256`, `selected_prompt`, and a substantive `development_review`. The selected prompt must be one of the predeclared names. Do not inspect reserved generations before selecting. Then:

```powershell
python -m checkin.response_backtest check --out <new-study-directory>
```

Completed requests are retained and skipped on resumption, including errors; failures are not silently retried away. For this completed study, the frozen `protocol.json` preserves the original baseline even though the production prompt changed afterward. A newly prepared study uses the then-current production prompt as its baseline. The exact study requests can be replayed from its frozen protocol and response payload hashes, but output text is not guaranteed bitwise deterministic across calls, hardware, or runtime builds.

The two `*_review.py` scripts are records of this run's qualitative assessment, not automated model judges. Targeted validation after the prompt change passed eight tests (`test_response_backtest.py` and `test_pipeline.py`) in 3.33 seconds.
