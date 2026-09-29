# Lantern Gate

A playable, local text-and-vision NPC prototype built on MELD. Walk around a pixel-art harbor courtyard with **WASD or arrow keys**. Approach **Mara at the northern gate** to open her dialogue and the camera panel. **Escape** returns to the village and stops the camera. **Go to Mara** is a keyboard/touch-accessible shortcut; **Full screen** uses the browser's full-screen mode when supported.

A short **Words + emotion** popup explains how the same line can prompt a different tone. It appears once per browser tab; reopen it with **How it works** in the village or **?** while talking to Mara. Close it with **Got it** or **Escape** to resume playing.

Above each reply, a small cue label shows the direction selected for Mara, such as **Fear cue → Reassuring**. Hover for its evidence source and pose description. The label stays with that reply while the live camera tag changes. Mara's in-world sprite matches the selected style with a grin and raised lantern, open hand, firm nod, or another pixel pose. These are authored visual reactions to the chosen direction, not a second emotion model or proof that generated wording followed the direction perfectly. Explicit words still take precedence. A new reply replaces the cue only when its first text arrives.

Choose one of **four hardcoded example replies** to send it immediately; the set changes after each completed NPC response. Or write your own in **Custom** and use the send arrow (or Enter). Your words and eligible recent webcam evidence produce a MELD emotion tag and streamed local character dialogue. For the exact authored ambiguous lines, an available visual estimate can tentatively choose Mara's delivery (playful, careful, steady, etc.). This rule is disclosed in the structured `interaction` field; it does not change the classifier's emotion label. Explicit words take precedence. Facial expressions cannot reliably eliminate ambiguity.

The game runs in the browser; Python and the existing local llama.cpp server perform inference. The scene includes movement, collision, a following camera, NPC proximity, touch controls and a dialogue overlay. After three completed exchanges, a confirmed route starts a scripted journey: Mara leads the player along the bridge or sea stairs and relights the visible beacon. Escape pauses the journey; E resumes; Restart scene resets it. Custom text remains available. Unclear choices, failures, cancellations and refusals do not trigger departure. Generated prose cannot execute arbitrary actions. Combat, inventory and persistent saves are intentionally absent.

See the [quest and emotion update](reports/game-character/QUEST_UPDATE.md) for the current interaction and paired local generation checks. Start with the [handoff](HANDOFF.md) and [game-character report](reports/game-character/REPORT.md). Historical reports retain the earlier Check-in framing and its results; their supportive-response evaluations do not validate the new NPC prompt. The latest classifier weights are unchanged.

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

On the supplied RTX 3080, the recorded warm **clip-mode** benchmark produced emotion state / first token / completed reply at **241 / 440 / 604 ms p95**, across 30 fusion turns. Peak GPU use was **7,581 MiB**, including desktop processes. Capture, upload and browser rendering are excluded. These measurements predate the live-camera flow and do not measure its latency or accuracy. The verified complete parameter count is **4,464,745,375**, below the six-billion cap. [REPORT.md](REPORT.md) links the audit, benchmark, stress results, response review and validation.

The second follow-through pass fixed four reproduced cancellation/ownership failures: all eight real-model lifecycle cases and 48 alternating video/text turns now pass. Exact local context budgeting preserves inputs and returns an actionable overflow error. A development confidence diagnostic did not justify calibration, so it was not deployed. See the [reliability report](reports/reliability/REPORT.md) for protocols, failures, resource limits and browser evidence.

A later browser check reproduced and fixed delayed replay loading overwriting a fresh draft after reset. That revision, before the live-camera addition, passed New/Stop/normal-load checks and the full **268-test** software suite. A real text+video replay completed; its known wrong emotion and response assumptions remain recorded in the [replay lifecycle report](reports/replay-lifecycle/REPORT.md).

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

- For MELD, uploaded clips and replay, eight frames are sampled from the first ten seconds. The upload tab accepts clips of up to twenty seconds and the state records truncation.
- The live visual display starts with the first usable face frame. It smooths probabilities with 85% weight on the new frame and 15% on the previous display, only when the gap is at most 0.4 seconds. Estimates can still change or flicker; smoothing does not establish greater accuracy.
- Evidence used for a reply remains stricter: at least six usable face samples among at most eight recent samples within four seconds. It uses the same face geometry and feature pooling as clips, with rolling sampling instead of clip positions. Fusion evidence expires after four seconds without a fresh observation. A visible live tag does not by itself mean that this evidence is ready for a submitted text-plus-vision turn.
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

Classification performance does not establish NPC-response quality. Review replayed responses separately for grounding in the person's message, appropriate use of emotion evidence, uncertainty, brevity, and a useful follow-up question. Any small manual review remains a qualitative check, not a clinical validation.

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

Stop the background processes created by the launcher with `scripts/stop.ps1`, using the same artifacts directory. It verifies the recorded process identities before stopping them. This is separate from **Escape / Back to village**, which leaves the encounter, cancels its pending reply and stops the camera.

The generator binds to `http://127.0.0.1:8081`; the browser app defaults to `http://127.0.0.1:7860`. The generator script starts a hidden background process, records its process identity and log paths, and checks readiness. Keep both components running during an interaction. The UI can open before setup is complete: the status message identifies incomplete setup; use `python -m checkin.cli doctor` for details, and no fake predictions are substituted.

Startup preloads the vision components so the camera need not wait for them on its first frame. The text encoder is loaded when a conversational turn first needs it; the first reply can therefore take longer than later replies.

In the browser:

1. Follow the large arrow with **WASD / arrow keys**. The key tutorial above your character disappears after movement. Approach Mara, or use **Go to Mara**.
2. The encounter has three parts: separate floating camera and reply rectangles on the right and a speech bubble anchored above Mara’s actual in-world sprite on the left. Select **Turn camera on** once and permit local webcam access. No audio or recording is used.
3. **Click a sample to send immediately.** For your own words, type in **Custom**, then use the send arrow or Enter. A sample is a separate turn; it does not need another button press.
4. **Escape / Back to village** cancels the pending reply after its current processing step, stops camera tracks, and returns to movement. **E** reopens the conversation nearby. Refresh the page for a new conversation.

The compact game deliberately omits clip-upload, replay, New/Stop buttons and technical diagnostics from the visible encounter. Recorded-video evaluation is still available through the CLI below; the internal UI components remain for regression coverage.

A usable first camera frame can produce a visual tag. Sending only uses camera evidence after six recent samples pass the separate fusion checks; otherwise the classifier uses explicit text fallback. Tracking shares the GPU with conversation turns and resumes after generation releases it. The generator receives text and structured emotion evidence, not image pixels. It uses a 4,096-token context and at most 96 new tokens. The prompt requests one or two short sentences, but the local model does not always obey that limit.

Completed turns clear the Custom field. Failed turns preserve it for retry. Messages are capped at 4,000 characters, and the classifier sees at most 128 wordpieces. The game uses local bundled assets and adds no learned parameters.

## Trace one input and measure interaction speed

With all three trained heads and the generator available:

```powershell
.\.venv\Scripts\python.exe -m checkin.cli replay --id test:0:0
.\.venv\Scripts\python.exe -m checkin.cli benchmark --turns 30 --fallback-turns 10 --warmup 3
.\.venv\Scripts\python.exe -m checkin.cli audit-parameters
```

The replay resolves an official MELD utterance, sends its text and video through the same pipeline as the UI, prints the tag and streamed reply, and saves the event trace. The reference label is retained for evaluation only and is not passed to the inference pipeline. Choose a different valid identity from `manifests/meld.jsonl` if the default clip is visually unavailable. The compact game hides the optional browser replay tab.

```text
Camera frames → first-frame visual tag, lightly smoothed across recent frames
              → separate conservative window of evidence for a reply
Typed text + fresh camera evidence (or an uploaded/replay clip)
    → visual feature pooling + text encoding
    → trained vision/text diagnostic heads
    → fusion head, or explicit text fallback
    → structured MELD emotion state
    → local Qwen with message, evidence, and prior history
    → streamed NPC dialogue + completed state
```

The state includes session/turn IDs, received time, input availability, seven uncalibrated class probabilities, evidence source, visual quality/rejection reason, modality-specific labels/disagreement, response status/text, and backend timings. Browser capture timestamps are unavailable. Live window timestamps describe backend receipt, not camera exposure times; recorded-clip capture-start/end timestamps remain explicitly null.

“Real-time” has two parts: a fast visual display that can use the first valid face frame, and a conversational turn started by the send arrow or a sample button. Camera capture requests run at 5 Hz (every 0.2 seconds), and the display timer polls every 0.1 seconds. These are configured intervals, not guaranteed end-to-end latency; browser scheduling, image quality and GPU availability affect updates. The separate six-sample fusion window still takes several callbacks to become usable. Live inference pauses while the serialized reply owns the GPU. The [latency update report](reports/live-camera-latency/REPORT.md) records this revision's validation; no emotion-accuracy gain is claimed.

The earlier clip-mode engineering targets were a warmed-up emotion state within one second and first response token within two seconds after text and clip were ready. The measured backend replay sample met those targets; it does not establish an end-to-end webcam guarantee. `benchmark` still measures the recorded-clip path: it excludes three warm-up turns, measures 30 usable fusion inputs and ten no-video fallback inputs, and reports p50/p95/max latency. It also samples GPU memory and the Python/generator resident memory. Video capture, upload/transcoding, and browser rendering are outside these backend timings; cold model-load time is reported separately by the pipeline.

## Data handling, limitations, and handoff

Inference endpoints are loopback-only by default, with no public Gradio tunnel or analytics. Setup downloads require internet access; local inference does not call a remote model. Camera permission is controlled by the browser. Gradio temporarily stores uploads on local disk and schedules cleanup after one hour. Clearing conversation history does not immediately erase temporary files. Dataset audit images, replay reports, and benchmark responses remain in the artifacts directory until removed.

The core scope is intentionally text + vision, with sampled live-camera frames and an optional recorded-clip path. Audio, ASR, TTS, physical robots, reinforcement learning, language-model fine-tuning, identity recognition, and clinical assessment are out of scope. Fusion did not outperform text-only classification in the measured test results. Scores are uncalibrated, and transfer from television dialogue to personal webcam check-ins has not been established. A working physical webcam preview in the earlier live-camera build does not validate emotion accuracy or clinical effectiveness; validation of the faster revision is reported separately.

For a portable handoff, provide this source tree, dependency/environment records, the selected trained heads and their metadata, observed reports, and the artifact manifest. Acquire third-party models/data under their own terms rather than publishing the television clips as part of the repository. Important external and AI-generated components are identified in [THIRD_PARTY.md](THIRD_PARTY.md).

### Repository map

| Path | Purpose |
|---|---|
| `src/checkin/app.py` | Local browser interface |
| `src/checkin/live_camera.js`, `live_vision.py` | Native webcam lifecycle bridge and bounded, session-owned live evidence |
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
