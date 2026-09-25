# Check-in: measured prototype handoff

Run date: 25 September 2026. This is a completed local, turn-based text-and-vision prototype for supportive reflection. It estimates one of seven MELD categories and streams a short response. It is not a therapist or a validated assessment of a person's internal emotional state.

The complete MELD extraction, three classification-head training stages, held-out evaluation, local language-model generation, and resource benchmark ran on the supplied RTX 3080. The trained heads and machine-readable reports are included. **Fusion did not outperform text alone on the held-out test set.** The system demonstrates the complete interaction, with that limitation visible in the evidence.

## Architecture and choices

The browser records or accepts a short clip alongside typed text. A local Python process samples eight frames, selects a usable single face, extracts face and text features, and runs the MELD heads. A local llama.cpp server then receives the text, tentative emotion evidence, visual availability, modality disagreement, and bounded conversation history. It streams its response back through the same pipeline used by the command-line replay.

The classifier produces the tag independently of generation. The generator sees the visual evidence through the structured state; it does not receive video pixels. If vision is unavailable, the trained text head supplies an explicit `text_fallback` result. The interface calls the result an **emotion signal**, avoids presenting softmax scores as certainty, and keeps diagnostics secondary to the conversation.

| Required component | Total parameters |
|---|---:|
| Qwen3-4B-Instruct-2507, Q5_K_M | 4,022,468,096 |
| DeBERTa-v3-large text encoder | 434,012,160 |
| EmotiEffLib AffectNet EfficientNet-B2, including original head | 7,710,857 |
| YuNet face detector, conservative initializer count | 53,121 |
| MELD vision head | 181,255 |
| MELD text head | 132,103 |
| MELD fusion head | 312,711 |
| **Complete inference path** | **4,464,870,303** |

This leaves **1,535,129,697** parameters below the six-billion cap. Frozen parameters count, all three runtime heads count, and quantization does not reduce the parameter count. YuNet's allowance includes 17 nonlearned initializer values. There are no omitted learned trackers, adapters, projectors, speech models, or auxiliary generators. See [verified audit](reports/overnight/baseline/parameters.json), [independent architecture counts](reports/overnight/baseline/architecture-counts.json), and [pinned model inventory](manifests/models.json).

Qwen's four-billion-parameter instruction model gives the response stage most of the budget. The larger text encoder complements the specialized, compact expression encoder. Keeping both encoders frozen makes a weekend training run practical and allows caching. This choice was not tested against the earlier smaller proposed stack, so increased model size is not evidence of an accuracy improvement.

## Data, training, and visual coverage

The SHA256-verified official raw MELD archive and pinned annotations were prepared with the official **9,989 train / 1,109 dev / 2,610 test** utterances. Labels are `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, and `anger`. One development clip is missing; its text remains usable.

Visual eligibility requires at least six usable frames out of eight. Frames with multiple detected faces or abrupt changes in selected face position are rejected. This is a conservative availability heuristic, **not verified speaker attribution**. MELD utterance labels are not independent facial-expression ground truth; a visible listener can differ from the speaker.

| Split | All utterances | Visually eligible |
|---|---:|---:|
| Train | 9,989 | 2,541 |
| Development | 1,109 | 316 |
| Test | 2,610 | 658 (25.21%) |

Test exclusions comprised 1,620 multiple-face cases, 255 track changes, and 77 cases with insufficient valid frames. The low coverage is a material trade-off: ambiguity is reduced at the cost of excluding most clips and changing the population seen by the vision head.

Twelve training contact sheets received an AI visual inspection, recorded in [visual-review.json](manifests/visual-review.json). Seven were accepted and five rejected by the heuristic. This convenience sample is not a human annotation study and does not establish speaker identity or emotion correctness.

MELD training updated only the small MLP heads. Defaults were seed 42, hidden width 128, dropout 0.2, AdamW with learning rate 0.001 and weight decay 0.0001, batch size 128, at most 30 epochs, and early stopping after five non-improving epochs. Unweighted and inverse-square-root weighted cross-entropy were compared using development macro F1; weighted loss was selected for all three stages.

| Stage | Training rows | Selection rows | Selected epoch | Dev macro F1 | Training time |
|---|---:|---:|---:|---:|---:|
| Vision | 2,541 | 316 eligible | 15 | 0.2953 | 13.26 s |
| Text | 9,989 | 1,109 | 18 | 0.3688 | 30.05 s |
| Fusion | 9,989 | 316 eligible | 14 | 0.4017 | 35.80 s |

Fusion training included 15% visual dropout. Features were extracted on the GPU with four CPU preprocessing workers. Extraction completed all splits, with the train run resumed from cached chunks; recorded invocation durations must not be interpreted as a clean full-run timing. The heads trained on CPU after feature extraction. Checkpoints bind label order, feature identity, cache hashes, model revisions, and training settings. Test labels were not used for head selection, and classifiers were not retrained after held-out evaluation.

Three exact video duplicates cross the official train/test boundary; one pair has different annotation text and emotion. Primary results preserve the official split. A secondary diagnostic excludes the three affected test rows, leaving 2,607. See [preparation](reports/preparation.json) and the secondary section of [evaluation](reports/evaluation-test.json).

## Held-out results

All models in each comparison use the **same examples**. Scores are F1 on a 0–1 scale.

| Population | Model | Macro F1 | Weighted F1 |
|---|---|---:|---:|
| Common visual subset, 658 | Train-majority baseline | 0.0941 | 0.3233 |
| Common visual subset, 658 | Vision only | 0.2113 | 0.3958 |
| Common visual subset, 658 | Text only | **0.3750** | **0.5762** |
| Common visual subset, 658 | Text + vision fusion | 0.3671 | 0.5638 |
| Full test, 2,610 | Train-majority baseline | 0.0928 | 0.3127 |
| Full test, 2,610 | Text only | **0.3787** | **0.5873** |
| Full test, 2,610 | Fusion when eligible, text otherwise | 0.3784 | 0.5848 |

The development benefit of fusion did not generalize to held-out test performance. The operational system remains multimodal to demonstrate the requested architecture, with text fallback for unavailable vision. No statistical significance or state-of-the-art claim is made.

Operational per-class F1 is 0.7736 neutral, 0.4824 surprise, 0.1522 fear, 0.2708 sadness, 0.4955 joy, **0.0282 disgust**, and 0.4465 anger. Disgust recall is only 1/68. The eligible development subset includes only three disgust and twelve fear examples, making selection for rare classes particularly unstable. Scores are uncalibrated softmax values.

The [evaluation JSON](reports/evaluation-test.json), [per-example predictions](reports/predictions-test.jsonl), and [operational confusion matrix](reports/confusion-test-full_split-operational_fusion.png) retain full counts and class-level evidence.

## Do both modalities affect the model?

A post-training **development-only diagnostic**, using the same 316 visually eligible examples, compared correctly paired fusion inputs with modified inputs. No training or checkpoint selection was performed by this diagnostic.

| Condition | Macro F1 | Labels changed from paired fusion |
|---|---:|---:|
| Paired text + vision | 0.4017 | 0 / 316 |
| Zero visual features and quality | 0.3299 | 80 / 316 |
| Seeded shuffle of visual features and their quality | 0.3039 | 105 / 316 |
| Zero text features | 0.2638 | 174 / 316 |
| Separately trained text head | 0.3713 | See report |

These checks establish sensitivity to both modalities; they do not establish held-out improvement. Zero-input conditions examine the raw fusion head and are not the deployed fallback behavior. See [ablation-dev.json](reports/ablation-dev.json).

The generator receives the classifier evidence in its prompt, but a controlled experiment changing only that evidence was not performed. Reply examples alone do not quantify how much emotion conditioning changes generation.

## Real-time definition and observed resources

“Real-time” means a single user's turn **after their text and clip are ready**: a warmed emotion state within one second and the first generated token within three seconds. It does not mean continuous video inference or immediate response while the user is speaking. The interface supports successive check-ins with bounded text history and cancellation.

Observed machine: Windows 11, AMD Ryzen 7 9700X (8 cores / 16 threads), 32 GB installed RAM (31.10 GiB usable), NVIDIA RTX 3080 with 10,239.5 MiB VRAM, driver 610.60. Python 3.12.10, PyTorch 2.8.0+cu126, Gradio 5.49.1, and llama.cpp b11146. Qwen ran locally with all layers on GPU, flash attention, a 4,096-token context, one request at a time, and a 96-token response limit. See [environment.json](reports/environment.json) and [requirements.lock.txt](requirements.lock.txt).

The final benchmark used the first 30 eligible official test utterances in manifest order, plus ten no-video replays of their text, excluding three warm-up turns. It measures short MELD snippets with empty conversation history, not a representative workload of long personal check-ins. All times below begin at backend classification; capture, upload/transcoding, browser rendering, and model loading are excluded.

| Path and measurement | p50 | p95 | Maximum |
|---|---:|---:|---:|
| Fusion: emotion state | 202 ms | 264 ms | 348 ms |
| Fusion: first response token | 426 ms | 489 ms | 550 ms |
| Fusion: response complete | 592 ms | 729 ms | 793 ms |
| Text fallback: emotion state | 27 ms | 32 ms | 34 ms |
| Text fallback: first response token | 234 ms | 255 ms | 257 ms |
| Text fallback: response complete | 383 ms | 488 ms | 504 ms |

Both warmed backend targets were met in this run. A fresh pipeline's encoder loading and CUDA shape warm-up took **12.14 seconds** in the saved replay, in addition to turn timing. Starting the separate generator and browser process adds further startup time that this number does not cover. The UI communicates that the first response takes longer.

An earlier benchmark exposed occasional lazy CUDA initialization for six- and seven-face batches (fusion p95 classification 2.016 seconds). Warming the six-, seven-, and eight-frame shapes during model loading removed those observed pauses in the subsequent run; this is an engineering timing change, not a classifier change. The earlier [benchmark](reports/benchmark-before-warmup.json) is retained.

Observed peaks, sampled every 0.5 seconds:

- GPU memory: **7,565 MiB / 7.39 GiB**, including desktop and other GPU processes.
- Python process resident memory: **3,163.5 MiB**.
- Generator resident memory: **3,249.4 MiB**.
- Whole-system RAM in use: **25,494.6 MiB / 24.90 GiB**, including other applications. This is not the app's isolated RAM requirement; process peaks may occur at different times.

These are observed peaks for this workload, not guaranteed upper bounds. Training/extraction and interactive generation were run separately. The full per-turn responses, vision quality, timings, and measurement conditions are in [benchmark.json](reports/overnight/baseline/benchmark.json).

## One real input through both outputs

The saved replay uses official **test:4:2**, text “It kicked! I think the baby kicked!” and its 1.50-second clip. Eight usable frames reached the fusion head. The result was **anger 0.3402**, with surprise close behind at 0.3215, and disagreement between text and vision diagnostic heads. The reference label was **surprise**, so this is a documented classification failure.

Qwen received the actual message and uncertain structured state, then streamed a complete response beginning “That's so exciting”. It also called the movement a “wonderful sign of growth,” an unsupported medical interpretation despite the prompt constraint. This shows both that the generator does not blindly repeat an incorrect tag and that prompt instructions do not guarantee grounding or medical appropriateness. It is not a successful therapeutic response.

The [replay event trace](reports/replay-test-4-2.json) contains input identity, state, token deltas, and final text. The reference label was retained only for evaluation and was not passed into inference. Live versus cached evaluation predicted the same label; a parity record reports the small floating-point difference.

The actual **browser** replay of the same utterance produced **surprise 0.3495** and the response “That's so exciting — congratulations on feeling the baby move! What was the feeling like when it kicked?” Both outputs and the `fusion` source were visibly verified; see [browser-demo.json](reports/browser-demo.json). Gradio 5.49.1 removes the audio track by re-encoding video before inference, so browser and raw CLI media are not byte-identical. This borderline label change exposes preprocessing sensitivity, not an accuracy improvement. The response still assumes a positive interpretation. The browser result must not be substituted into the raw-video held-out evaluation.

The first ten benchmark responses also received an [AI qualitative review](reports/response-review.json). Some ask appropriate clarifying questions, while others invent context or assume a positive interpretation. This is a small convenience review, not a human response-quality study. Generated text should not be relied upon for clinical guidance.

## Completion and handoff

Completed: pinned local-model acquisition and counting; full MELD preparation and feature extraction; vision-only, text-only, and fusion-head training; held-out evaluation and modality diagnostics; structured state and local text streaming; a calm, responsive browser UI; replay; resource measurements; Windows launch/stop scripts; trained-head installation; and software regression checks. The repository includes actual trained heads, with reproducible setup for the larger third-party weights.

The original full software suite passed **77 tests**. After the startup warm-up and prompt changes, **28 targeted tests** passed. After repairing access to the exact replay clips stored outside the repository, **20 UI tests** passed, including the new regression; [its JUnit report](reports/ui-final-tests.xml) is included. These fixture tests verify software behavior, not ML accuracy. Actual model replay and browser verification are separate evidence.

The desktop and 390-pixel mobile layouts were visually inspected, including contrast, stacking, and horizontal overflow. Real prerecorded MELD text/video is used to verify both on-screen outputs. Live camera hardware and its browser permission flow were not exercised; recording/upload support is implemented. Camera data stays on the local machine, but Gradio temporarily stores uploads on disk and schedules cleanup after one hour.

A separate [two-turn browser conversation](reports/browser-conversation.json) verified successive text-fallback turns and clearing the conversation. Its authored test messages are not personal user data. It also preserves imperfect tags and an unsupported response generalization, rather than treating a working interface as evidence of response quality.

Intentionally excluded: audio, ASR/TTS, three-modality fusion, reinforcement learning, robot integration, continuous video, backbone or language-model fine-tuning, calibrated confidence, verified active-speaker tracking, clinical validation, and a crisis-care product. TV dialogue and performed expressions are a substantial domain shift from personal webcam reflection. Future accuracy work should prioritize speaker/face alignment and stronger evaluation over spending the remaining parameter budget.

Launch and reproduction instructions are in [README.md](README.md). External weights, dataset provenance and restrictions, the GGUF conversion, libraries, and AI-generated code are identified in [THIRD_PARTY.md](THIRD_PARTY.md). The raw television videos and third-party model binaries are not bundled in the portable source archive.
