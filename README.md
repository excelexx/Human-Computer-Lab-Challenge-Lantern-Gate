# Check-in

A local text-and-vision prototype that asks how your day went, estimates a MELD emotion category, and streams a short supportive response. Type a message and record a 3–5 second webcam clip. The browser is the interface; Python and a local llama.cpp server perform inference on your computer.

This is a supportive reflection companion, not a clinical system. Facial expressions are uncertain evidence. The person's own account takes precedence over appearance-based speculation.

## Model selection and parameter budget

The upgraded stack has a conservative upper bound of **4,464,745,375 learned parameters**, below the **6,000,000,000** limit. The authoritative inventory is [manifests/models.json](manifests/models.json); it supersedes the earlier implementation plan's smaller model selection.

| Required component | Role | Parameters |
|---|---|---:|
| OpenCV YuNet | Face detection | 53,121 |
| EmotiEffLib `enet_b2_7` | AffectNet-pretrained EfficientNet-B2 face features, including original FER head | 7,710,857 |
| Microsoft DeBERTa-v3-large | Text features | 434,012,160 |
| MELD vision head | 1,408 → 128 → 7 | 181,255 |
| MELD text head | 1,024 → 7, folded training scaler | 7,175 |
| MELD fusion head | 2,435 → 128 → 7 | 312,711 |
| Qwen3-4B-Instruct-2507 | Local response generation, Q5_K_M GGUF | 4,022,468,096 |
| **Total** | **All required learned components** | **4,464,745,375** |

YuNet's allowance includes 17 nonlearned initializer values, making the total slightly conservative. Frozen weights still count. Qwen's shared input/output embeddings count once; quantization changes memory use, not the number of learned parameters. Tokenization, frame sampling, geometric face selection, normalization, pooling, and the browser have no learned weights. There is no extra learned tracker, projector, adapter, auxiliary generator, or learned evaluation model.

The larger generator and text encoder use more of the available budget without adding a difficult video-language training job. The compact visual encoder is already trained for facial expression recognition. This is an engineering choice; a larger parameter total alone does not establish better MELD accuracy.

## Completed and measured

The local prototype has processed all official MELD train/dev/test rows, trained vision/text/fusion heads, streamed local Qwen responses, and run the browser replay. The current stack retains the vision/fusion MLPs and replaces the text fallback with a regularized linear head selected by training-dialogue cross-validation. Both pretrained encoders stay frozen.

The overnight work added **72 grouped CV fits, 760 real video stress conditions, 20 mismatched-modality permutations, and 204 local response-study generations**. Complete protocols and retained failures are under `reports/overnight/`. The current operational model improves full-dev macro F1 from .38059 to .41855; on the **reused** official test benchmark, macro F1 moves from .37845 to .40324, while weighted F1 slightly decreases (.58479 to .58374) and accuracy falls (.60115 to .59042). The original test was already observed; later figures are not a fresh holdout.

**Vision still has no demonstrated accuracy advantage over text alone.** Only 658/2,610 test rows pass the visual checks. The new text control alone scores .41583 macro F1, above the multimodal system. Rare-class improvements and regressions are detailed in [REPORT.md](REPORT.md); [BASELINE_REPORT.md](BASELINE_REPORT.md) preserves the original evidence.

On the supplied RTX 3080, the updated warm benchmark produced emotion state / first token / completed reply at **241 / 440 / 604 ms p95**, across 30 fusion turns. Peak GPU use was **7,581 MiB**, including desktop processes. Capture, upload and browser rendering are excluded. The verified complete parameter count is **4,464,745,375**, below the six-billion cap. [REPORT.md](REPORT.md) links the audit, benchmark, stress results, response review and validation.

The second follow-through pass fixed four reproduced cancellation/ownership failures: all eight real-model lifecycle cases and 48 alternating video/text turns now pass. Exact local context budgeting preserves inputs and returns an actionable overflow error. A development confidence diagnostic did not justify calibration, so it was not deployed. See the [reliability report](reports/reliability/REPORT.md) for protocols, failures, resource limits and browser evidence.

A later browser check reproduced and fixed delayed replay loading overwriting a fresh draft after reset. The fixed UI passed New/Stop/normal-load checks and the full **268-test** software suite. A real text+video replay completed; its known wrong emotion and response assumptions remain recorded in the [replay lifecycle report](reports/replay-lifecycle/REPORT.md).

## Windows setup

The target machine is Windows with an RTX 3080 (10 GB VRAM) and 32 GB system RAM. Use Python **3.12**, a compatible NVIDIA driver, and FFmpeg/ffprobe on `PATH`. FFmpeg is required by Gradio to inspect and mute video inputs; it does not add learned parameters. See the [official FFmpeg download page](https://ffmpeg.org/download.html) for Windows distribution links.

Open PowerShell in the copied repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

The main dependencies are declared in `pyproject.toml`; `requirements.lock.txt` records the resolved environment when included in the handoff. The CUDA PyTorch installation is explicit so that a CPU-only wheel does not silently replace the intended GPU runtime. A separate CUDA compiler toolkit is not required for these prebuilt packages. The pinned llama.cpp release supplies its own corresponding runtime DLLs.

By default, models, data, caches, checkpoints, and reports live in `.artifacts` beside this README. Change the location before running commands if needed:

```powershell
$env:CHECKIN_HOME = 'D:\CheckinArtifacts'
.\.venv\Scripts\python.exe -m checkin.cli doctor
```

All main Python commands also accept `--home`. In the original Codex workspace, the working artifacts directory is outside this repository, at `../../work/runtime`. To reuse that existing directory from this repository:

```powershell
$env:CHECKIN_HOME = (Resolve-Path '..\..\work\runtime').Path
```

Do not copy this workspace-specific path onto a different computer. Large model and dataset artifacts are not embedded in the source repository.

## Acquire MELD and the local models

```powershell
.\.venv\Scripts\python.exe -m checkin.cli download
.\.venv\Scripts\python.exe -m checkin.cli prepare
.\.venv\Scripts\python.exe -m checkin.cli audit-parameters
```

`download` fetches pinned model artifacts, a Windows CUDA llama.cpp release, and the MELD raw archive. Downloads use SHA256 verification and resumable partial files. `--models-only` skips MELD when preparing an inference machine that will receive the trained checkpoints separately. `prepare` extracts MELD, reads its official annotations, and preserves the train/dev/test identities. Review the dataset and weight provenance in [THIRD_PARTY.md](THIRD_PARTY.md).

Allow substantial disk space for the archive, extracted clips, models, and feature caches. The observed inference memory figures above apply to the recorded benchmark configuration; rerun `doctor` and `benchmark` after changing hardware or settings. Avoid running another download process against the same artifacts directory.

## Inspect faces, extract features, then train

```powershell
.\.venv\Scripts\python.exe -m checkin.cli audit-data --split train --count 150
.\.venv\Scripts\python.exe -m checkin.cli extract --split all --device cuda
.\.venv\Scripts\python.exe -m checkin.train --stage vision --device cpu
.\.venv\Scripts\python.exe -m checkin.train --stage text --device cpu
.\.venv\Scripts\python.exe -m checkin.train --stage fusion --device cpu
```

`audit-data` writes contact sheets and an audit JSON file. It marks speaker review as **not reviewed**; generating pictures does not complete human review. Inspect a sample before accepting its visual labels as reliable.

Feature extraction uses four CPU preprocessing workers and GPU encoders, and is resumable in chunks. The default uses the complete official splits; `--limit` is available for a clearly labeled smoke run. Limited caches are not full MELD results. Head training is small enough to default to CPU; `--device cuda` is also supported. `--stage all` runs vision, text, then fusion.

These three training commands reproduce the original MLP baseline. The supplied current text head instead uses the subsequent grouped-CV linear selection; follow [the reproduction guide](reports/overnight/model-study/REPRODUCE.md) to reproduce that selection and prepare the replacement from cached features. Preparation writes to a separate candidate directory and does not overwrite active checkpoints.

The baseline training command compares unweighted cross-entropy with inverse-square-root class-weighted loss. Defaults are 30 maximum epochs, early stopping after 5 non-improving epochs, batch size 128, and seed 42. Checkpoint selection uses development macro F1, never test performance: vision and fusion select on the visually eligible development subset, while text selects on all development examples. Fusion training still uses all training rows, with explicit missing-vision features and visual dropout. Checkpoint metadata records the label order, feature identity, cache hashes, seed, selected loss, and hyperparameters.

### Visual handling and fusion

- Eight frames are sampled from the first ten seconds. Live clips should normally be 3–5 seconds; the UI accepts up to twenty seconds and the state records truncation.
- YuNet selects a usable single face per frame. Multiple faces, too few usable frames, and abrupt changes in face position make the visual branch unavailable. The method is a documented heuristic, not active-speaker recognition.
- Crops use RGB, a 260×260 resize, and the visual model's prescribed normalization. Mean frame features are L2-normalized. Text uses masked-mean DeBERTa features with a 128-wordpiece limit, also normalized.
- The fusion head receives 1,408 visual features, 1,024 text features, and three visual-quality values. During training, some visual features are dropped to expose the head to missing evidence.
- At inference, usable visual evidence selects the fusion head. Missing/unreliable vision selects the text head explicitly. Missing vision never means `neutral`.

MELD labels belong to conversational utterances, not independently verified facial expressions. A visible listener may differ from the speaker. Strict filtering reduces ambiguity but also reduces visual coverage and can bias the retained subset. Report coverage and class distributions rather than hiding excluded examples.

## Evaluate on MELD

```powershell
.\.venv\Scripts\python.exe -m checkin.evaluate --split test
.\.venv\Scripts\python.exe -m pytest -q
```

Evaluation writes `reports/evaluation-test.json`, per-utterance predictions, and confusion-matrix images. It compares the training-majority baseline, text-only, vision-only, and fusion/operational predictions on the applicable common subsets. Macro F1, weighted F1, per-class metrics, counts, and coverage are reported. Inspect both the eligible visual subset and the full set with explicit text fallback. Do not compare scores obtained from different subsets as though their difficulty were identical.

Classification performance does not establish supportive-response quality. Review replayed responses separately for grounding in the person's message, appropriate use of emotion evidence, uncertainty, brevity, and a useful follow-up question. Any small manual review remains a qualitative check, not a clinical validation.

## Run a check-in

### Use supplied trained heads on another machine

If the handoff includes `trained-heads/vision.pt`, `text.pt`, and `fusion.pt`, you can install those heads without downloading MELD or repeating training:

```powershell
.\.venv\Scripts\python.exe -m checkin.cli download --models-only
.\.venv\Scripts\python.exe .\scripts\install-heads.py
.\.venv\Scripts\python.exe -m checkin.cli audit-parameters
```

The installer uses `CHECKIN_HOME` or `.artifacts`, with optional `--home` and `--source` arguments. It checks all three files before copying, verifies copied bytes, skips identical existing files, and refuses to overwrite a differing local checkpoint. Atomic installation requires a filesystem supporting hard links, such as NTFS. The runtime then validates that the supplied heads match the pinned model features. If `trained-heads` was not included, follow the baseline MELD training steps and the linked linear-head reproduction guide above.

### Launch the local application

After setup and training, the convenience launcher starts the generator and browser app in the background, then opens the local page:

```powershell
.\scripts\start.ps1
```

It uses the repository's `.venv` by default. `-PythonPath` selects another prepared environment, and `-ArtifactsRoot` selects another artifacts directory. In the original workspace, this reuses the existing environment and runtime directory:

```powershell
.\scripts\start.ps1 -PythonPath '..\..\work\venv\Scripts\python.exe' -ArtifactsRoot '..\..\work\runtime'
```

The original handoff also includes **Start Check-in.cmd** and **Stop Check-in.cmd** one directory above this repository for double-click use with that workspace's prepared environment. Use the PowerShell launchers after copying the repository to a different location or machine.

For foreground development, start the local response server and browser interface separately:

```powershell
.\scripts\start-generator.ps1
.\.venv\Scripts\python.exe -m checkin.app
```

Stop the background processes created by the launcher with `scripts/stop.ps1`, using the same artifacts directory. It verifies the recorded process identities before stopping them. This is separate from the browser's **Stop** button, which cancels one conversational turn.

The generator binds to `http://127.0.0.1:8081`; the browser app defaults to `http://127.0.0.1:7860`. The generator script starts a hidden background process, records its process identity and log paths, and checks readiness. Keep both components running during an interaction. The UI can open before setup is complete: diagnostics identify missing files or an unavailable generator, and no fake predictions are substituted.

In the browser:

1. Record a short clip with one face clearly in view, or upload one. Audio is not used.
2. Type what happened during your day and select **Send check-in**.
3. Read the tentative emotion signal and streamed response. Open diagnostics to inspect the state and timings.
4. Continue with another message/clip. **Stop** requests cancellation after the current processing step; **New conversation** clears the displayed conversation and requests cancellation for that session. An in-progress GPU operation must finish before its resources are released.

The generator receives the message, structured emotion evidence, and bounded prior text history. It does not receive the video pixels. The classifier determines the reported tag; the language model produces only the response. Generation uses a 4,096-token context, at most 96 new tokens, and one concurrent request. The prompt requests one to three sentences and at most one relevant follow-up question.

A completed or stopped turn clears its message and clip. Add a fresh clip for a new message, or continue with words alone. Failed turns keep the inputs for retry. Compatible uploads/replay clips preserve the video stream while audio is removed; other formats display a conversion warning. User messages are capped at 4,000 characters; the text classifier sees at most 128 wordpieces, so concise check-ins work best.

## Trace one input and measure interaction speed

With all three trained heads and the generator available:

```powershell
.\.venv\Scripts\python.exe -m checkin.cli replay --id test:0:0
.\.venv\Scripts\python.exe -m checkin.cli benchmark --turns 30 --fallback-turns 10 --warmup 3
.\.venv\Scripts\python.exe -m checkin.cli audit-parameters
```

The replay resolves an official MELD utterance, sends its text and video through the same pipeline as the UI, prints the tag and streamed reply, and saves the event trace. The reference label is retained for evaluation only and is not passed to the inference pipeline. Choose a different valid identity from `manifests/meld.jsonl` if the default clip is visually unavailable. The optional browser replay tab lists existing test media.

```text
Typed text + clip
    → face sampling/selection + text encoding
    → trained vision/text diagnostic heads
    → fusion head, or explicit text fallback
    → structured MELD emotion state
    → local Qwen with message, evidence, and prior history
    → streamed supportive response + completed state
```

The state includes session/turn IDs, received time, input availability, seven uncalibrated class probabilities, evidence source, visual quality/rejection reason, modality-specific labels/disagreement, response status/text, and backend timings. Capture-start/end timestamps are currently unavailable and are explicitly null; received time is recorded on the backend.

“Real-time” here means a turn-based check-in after text and clip are ready. Engineering targets are a warmed-up emotion state within one second and first response token within two seconds. The measured backend replay sample met those targets; it does not establish a guarantee for other clips or an end-to-end webcam latency. `benchmark` excludes three warm-up turns, measures 30 usable fusion inputs and ten no-video fallback inputs, and reports p50/p95/max latency. It also samples GPU memory and the Python/generator resident memory. Video capture, upload/transcoding, and browser rendering are outside these backend timings; cold model-load time is reported separately by the pipeline.

## Data handling, limitations, and handoff

Inference endpoints are loopback-only by default, with no public Gradio tunnel or analytics. Setup downloads require internet access; local inference does not call a remote model. Camera permission is controlled by the browser. Gradio temporarily stores uploads on local disk and schedules cleanup after one hour. Clearing conversation history does not immediately erase temporary files. Dataset audit images, replay reports, and benchmark responses remain in the artifacts directory until removed.

The core scope is intentionally text + vision. Audio, ASR, TTS, continuous-video processing, physical robots, reinforcement learning, language-model fine-tuning, identity recognition, and clinical assessment are out of scope. Fusion did not outperform text-only classification in the measured test results. Scores are uncalibrated, and transfer from television dialogue to personal webcam check-ins has not been established. Browser replay was exercised; physical camera hardware and clinical effectiveness were not validated.

For a portable handoff, provide this source tree, dependency/environment records, the selected trained heads and their metadata, observed reports, and the artifact manifest. Acquire third-party models/data under their own terms rather than publishing the television clips as part of the repository. Important external and AI-generated components are identified in [THIRD_PARTY.md](THIRD_PARTY.md).

### Repository map

| Path | Purpose |
|---|---|
| `src/checkin/app.py` | Local browser interface |
| `src/checkin/pipeline.py`, `schema.py` | Shared inference/event path and validated state |
| `src/checkin/data.py`, `features.py` | MELD preparation, visual quality checks, feature caching |
| `src/checkin/encoders.py`, `models.py` | Frozen pretrained encoders and trainable heads |
| `src/checkin/train.py`, `evaluate.py` | Training and evaluation |
| `src/checkin/generator.py` | Bounded prompts and local streaming transport |
| `src/checkin/cli.py`, `downloads.py`, `audit.py` | Reproduction, downloads, replay, measurement, accounting |
| `scripts/start-generator.ps1` | Start/verify the local Qwen server |
| `scripts/start.ps1`, `scripts/stop.ps1` | Start/stop the local application processes |
| `scripts/install-heads.py` | Install supplied trained heads without overwriting different checkpoints |
| `manifests/models.json` | Model revisions, artifact hashes, parameter inventory |
| `tests/` | CPU-only contract tests; no model downloads required |

### Common setup issues

- **CUDA unavailable:** inspect `doctor` output and the installed PyTorch wheel before starting feature extraction. The CPU fallback is useful for debugging, not an asserted real-time substitute.
- **Video preprocessing fails before a turn:** confirm both `ffmpeg` and `ffprobe` are on `PATH`, then reopen the shell/app so it sees the updated environment.
- **Generator not ready:** run its start script and inspect the paths printed in its error or `runtime/generator.json`. The script refuses to adopt an unrelated service already occupying port 8081.
- **Missing classifiers:** run extraction and all three training stages; downloaded pretrained weights alone are insufficient.
- **Checksum/cache mismatch:** investigate the artifact or feature change before replacing files. Checkpoints are bound to the preprocessing/model feature identity; retrain after intentionally changing it.
- **GPU out of memory:** close other GPU workloads and run extraction separately from generation. Memory fit must be verified with the actual checkpoint, context size, and workload.
