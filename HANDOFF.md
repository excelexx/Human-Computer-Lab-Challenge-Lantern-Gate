# Check-in: morning handoff

25 September 2026. The local text-and-vision prototype is ready to try at [127.0.0.1:7860](http://127.0.0.1:7860/). It returns a tentative MELD emotion tag and a streamed supportive response. All inference runs on this computer, within **4,464,745,375 total required learned parameters**.

## Try it

1. Open **Camera**, select **Turn camera on**, and grant the browser's camera permission. Tracking starts automatically; you do not record a clip. Keep one face in view: the first usable frame can produce a live visual tag, without waiting for six samples.
2. Type a concise message and select **Send check-in**. The app combines your words with fresh camera evidence only after its stricter six-sample checks pass; otherwise it uses text fallback, even when a fast tag is visible. Live tracking resumes after the serialized reply releases the GPU.
3. Read the submitted check-in's emotion signal and streamed response. **Diagnostics and setup** shows the structured result, live camera state, modality availability, disagreement and timings.
4. **Turn camera off** stops the camera and clears its signal. **New conversation** clears the conversation and camera evidence while keeping the preview on to collect fresh evidence. **Stop** cancels the current reply or replay load after its current processing step; it does not turn the camera off.

The display requests camera frames every 0.2 seconds and refreshes every 0.1 seconds. Its first-frame estimate is lightly smoothed: 70% new probabilities and 30% previous probabilities, only across gaps of at most 0.6 seconds. Tags may still change or flicker. Reply evidence remains separate and conservative, requiring at least six usable samples among at most eight recent samples within four seconds. These configured intervals do not guarantee end-to-end latency or better accuracy.

Startup preloads the vision components; the text encoder loads when the first conversational turn needs it. The [original live-camera report](reports/live-camera/REPORT.md) retains its integration and missing-duration checks. The [faster live-display report](reports/live-camera-latency/REPORT.md) includes physical webcam verification: a first-frame tag was visible within 289 ms of reset, with ongoing capture at roughly five frames per second. This is a bounded interaction check, not an accuracy or long-duration stability claim.

For a repeatable recorded-video demonstration, choose **MELD replay**, select an utterance, click **Use this utterance**, then **Send check-in**. The reference label is not sent to the model. **Upload clip** also accepts an optional recorded video instead of live camera evidence. Audio is removed; compatible video streams are preserved, with an explicit conversion fallback for other formats. Uploaded/replay clips clear after a completed or stopped turn.

If the app is closed, the original workspace includes `Start Check-in.cmd` in the directory above this repository. For a copied repository or another computer, follow [Windows setup and launch instructions](README.md#windows-setup). The source ZIP includes trained heads, source, tests and reports; pretrained weights, MELD videos, feature caches and the Python environment are separate downloads/artifacts.

## What improved

- A regularized linear text fallback was selected with training-dialogue cross-validation. Full development macro F1 rose **.38059 to .41855**. On the already exposed official test benchmark, macro F1 rose **.37845 to .40324**, while weighted F1 fell **.58479 to .58374** and accuracy fell **.60115 to .59042**. This is a reused benchmark, not a fresh holdout.
- Compatible video now keeps its encoded stream while removing audio; other formats use an explicit conversion fallback. Used clips clear after each completed or stopped turn.
- Cancellation, turn ownership, exact local context budgeting and failed-turn retry handling were corrected. All eight fixed real-model lifecycle cases, eight recovery turns and a 48-turn mixed-modality burst passed. The complete bounded reliability run lasted 43.07 seconds and does not establish hours of stability.
- Delayed replay loading can no longer reproduce the observed fresh-draft overwrite after New or Stop. Normal replay loading remains usable. Frontend updates already dispatched remain a documented transport boundary.
- The camera flow now supports continuous visual feedback without a recording step. This is an interaction change; it adds no claim of better emotion accuracy, clinical benefit or measured live-camera speed.

## Evidence and hardware

The overnight work completed **72 grouped cross-validation fits**, **760 real video stress conditions** (95 development clips, eight conditions), **20 mismatched-vision permutations**, and **204 local response-study generations**. The recorded revision before the live-camera addition passed **268 software tests**, with one existing dependency warning. These are distinct studies; software test counts are not model-evaluation counts.

On the supplied Windows RTX 3080 10 GB / 32 GB RAM machine, the recorded warm **clip-mode** fusion benchmark measured **241 / 440 / 604 ms p95** for emotion state / first token / completed reply, across 30 fusion turns. Capture, upload and browser rendering are excluded; model loading is separate. Observed total GPU peak was **7,581 MiB**, including desktop use. These measurements predate the live-camera flow and do not measure its latency or accuracy. Full system-memory and process measurements are retained in [the reliability report](reports/reliability/REPORT.md).

## Remaining weaknesses

**Vision has no demonstrated accuracy advantage.** Only 658 of 2,610 official test rows pass the visual checks, and text alone scores .41583 macro F1, above the hybrid's .40324. Face selection does not verify the speaker. MELD television dialogue is not a validated webcam or therapy domain.

The local generator still sometimes assumes emotions or circumstances the speaker did not state. The latest real replay retained both a wrong anger tag and an unwarranted excitement assumption. Calibration was investigated and not deployed. A working camera preview does not validate emotion estimates. No clinical validation, multi-browser queue test or long-duration inference stability claim is made.

Audio, reinforcement learning, robot integration, remote inference and clinical diagnosis/treatment were intentionally left out. This is a supportive reflection prototype.

## Reproducibility and work record

The original committed baseline and earlier study artifacts are preserved. Model selection used training/development evidence; exposed official test examples were not used for subsequent tuning. See [the complete report](REPORT.md), [original baseline](BASELINE_REPORT.md), [replay lifecycle evidence](reports/replay-lifecycle/REPORT.md), [external/generated components](THIRD_PARTY.md), and [execution log](OVERNIGHT.md).

Three bounded work batches finished by about 07:22 UTC. Later scheduled passes checked service health; they did not run continuous inference or duplicate experiments. Final handoff verification checked clean source, the parameter inventories, archive integrity and every delivered file hash, with both local services healthy. The source/ZIP inventory is refreshed after this handoff document is added. These checks establish the delivered artifact's integrity, not improved model accuracy or sustained inference stability.
