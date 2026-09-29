# Lantern Gate: handoff

29 September 2026. The local text-and-vision prototype is ready to try at [127.0.0.1:7860](http://127.0.0.1:7860/). It returns a tentative MELD emotion tag and a streamed game-character response. All inference runs on this computer, within **4,464,745,375 total required learned parameters**.

The interface is now a full-screen pixel town: **WASD / arrows** move your traveler; walking close to Mara opens dialogue and the camera panel. **E** reopens it nearby; **Escape** leaves and stops camera tracks. **Full screen** optionally fills the display.

The [game-character report](reports/game-character/REPORT.md) records the new prompt's controlled local generations, software checks and limitations. The [quest/emotion update](reports/game-character/QUEST_UPDATE.md) covers changing example sets, the in-game ending and stronger emotion-conditioned delivery. Earlier response studies below belong to the previous reflection-companion framing.

See the [dialogue matrix and remaining weaknesses](reports/game-character/MATRIX.md) for the latest response/consent testing.

## Try it

1. Follow the arrow with **WASD / arrow keys** to Mara. The floating W/A/S/D tutorial disappears once you move.
2. The camera, estimated emotion, four samples and the custom text input float as separate rectangles on the right. Mara stays visible in the actual town on the left, with her reply in a speech bubble above her. There is no outer panel or replacement portrait, and opening the encounter does not pan or zoom the game.
3. Select **Turn camera on** once to use local visual evidence. Click a sample to **send it immediately**, or type your own words and use the send arrow / Enter.
4. **Escape / Back to village** leaves the encounter, cancels queued/current replies, and stops the camera. **E** reopens it while nearby. Refresh for a fresh conversation.

The three-turn preset path ends after choosing a route and agreeing to go. An early explicit custom route choice followed by readiness can finish in two turns. Mara leads, the player follows, and the beacon lights. Escape pauses movement, E resumes, and Restart scene clears progress. The example buttons are hardcoded; Mara’s replies are generated live.

Live display is more responsive: 85% new-frame probability / 15% prior display, with smoothing only across gaps up to 0.4 seconds. The classifier weights, score calibration and six-sample fusion gate are unchanged. This is a responsiveness adjustment, not a demonstrated accuracy gain. Capture requests remain 5 Hz.

The compact encounter intentionally hides technical diagnostics and recorded-video controls. Use `python -m checkin.cli doctor` for setup and `python -m checkin.cli replay --id test:0:0` for recorded multimodal traces. The [game-character report](reports/game-character/REPORT.md) contains the new generation checks; earlier supportive-response studies are historical.

In the original workspace, `Start Check-in.cmd` still launches this app. For a copied repository follow [setup instructions](README.md#windows-setup). The source ZIP contains code, trained heads, bundled CC0 town tiles, the licensed local font, tests and reports. Pretrained models, datasets and the Python environment remain separate.

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

Audio, reinforcement learning, robot integration, remote inference and clinical diagnosis/treatment were intentionally left out. This is an emotion-aware game-character prototype. Movement and a scripted three-turn beacon quest are implemented. Combat, inventory, persistent saves and arbitrary actions driven by generated prose are intentionally absent.

## Reproducibility and work record

The original committed baseline and earlier study artifacts are preserved. Model selection used training/development evidence; exposed official test examples were not used for subsequent tuning. See [the complete report](REPORT.md), [original baseline](BASELINE_REPORT.md), [replay lifecycle evidence](reports/replay-lifecycle/REPORT.md), [external/generated components](THIRD_PARTY.md), and [execution log](OVERNIGHT.md).

Three bounded work batches finished by about 07:22 UTC. Later scheduled passes checked service health; they did not run continuous inference or duplicate experiments. Final handoff verification checked clean source, the parameter inventories, archive integrity and every delivered file hash, with both local services healthy. The source/ZIP inventory is refreshed after this handoff document is added. These checks establish the delivered artifact's integrity, not improved model accuracy or sustained inference stability.
