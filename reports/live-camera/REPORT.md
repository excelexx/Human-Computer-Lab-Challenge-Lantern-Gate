# Continuous camera and duration repair

25 September 2026. This update changes the interaction flow. Model weights, trained heads, parameter accounting, original baseline and historical benchmark results are unchanged.

## Delivered behavior

The Camera tab now uses a live image stream. Select **Turn camera on** and allow camera access; sampling starts automatically, without recording a clip. A separate **Live camera signal** shows the vision estimate while the preview remains on. **Send check-in** combines recent usable visual features with the message using the existing fusion head, then streams a local Qwen response. The camera remains available for the next message. **Turn camera off** stops its tracks and clears evidence. **New conversation** clears evidence and history while leaving the preview running.

Camera requests target 640×480 preview video, without audio. Gradio sends a JPEG about every 0.5 seconds; busy callbacks are skipped. The server holds at most eight feature vectors within a four-second rolling window, and requires six usable face samples before showing a label. It does not record a live video file. Raw callback frames are transient in-memory inputs; the retained window contains features and face geometry. Missing or multiple faces, a changed face track, stale input, camera-off and session changes remove usable evidence. The display remains an uncertain visual estimate, distinct from the combined tag for a submitted message.

Idle tracking runs only the face detector, visual encoder and vision head. Text encoding and generation occur on Send. One shared GPU lock serializes them; a reply takes priority over further camera samples. A reply waiting for an in-progress camera callback can be cancelled without releasing that callback's lock. Freshness is checked at turn admission, so initial text-model loading does not by itself discard a fresh camera snapshot. Off/New still revoke it. A frame finishing exactly during admission can conservatively force text fallback.

## Missing duration

Some browser-produced WebM files contain no duration metadata. The previous player could display `NaN`, and strict preparation rejected them. The optional **Upload clip** path now remuxes compatible VP8/VP9 WebM losslessly, verifies a finite duration, removes audio, and replaces the preview with that verified file. Limits remain 100 MiB and 20 seconds; overlong media is rejected, never silently truncated. A ticket check prevents delayed upload preparation from overwriting a new draft after reset or replacement.

The [duration check](duration.json) used a synthetic 1.5-second WebM with missing duration. All 18 decoded frames were identical after repair, with no audio. Browser verification then showed `0:00 / 0:02` (rounded display), with `HTMLVideoElement.duration = 1.5` and readyState 4. The live camera has no recording timer, so that duration issue does not apply to its preview.

## Evidence and limits

The [real-model check](runtime.json) replayed one existing MELD training frame into eight live callbacks at 0.5-second intervals. On this Windows RTX 3080 10 GB / 32 GB RAM computer, warm callback processing ranged from **14.2 to 25.0 ms**. Vision-only model loading took **4.17 seconds**. A subsequent cold text-model load took **10.41 seconds**; the fresh-at-admission observation still produced a fused emotion state and completed local Qwen response. Measured classification / first token / completion after loading were **174 / 449 / 605 ms** for that single turn. These are engineering observations, not a latency distribution, webcam capture measurement or emotion-accuracy evaluation.

The same check verified that idle camera startup did not construct the text encoder, a missing face immediately removed the prior label, camera-off revoked the previous snapshot, and late frames could not restart tracking. The required learned-parameter upper bound remains **4,464,745,375**. No additional learned component was added. Historical GPU/system-memory measurements remain in the earlier reports; this small check did not measure a new resource peak.

Browser verification also completed a text-only message and local response, retained the explicit text fallback, repaired the uploaded duration, and reset to a fresh Camera tab. **Physical webcam capture was not verified:** clicking the in-app browser's camera-access control did not yield a media stream during the check. No physical-camera success, long-duration streaming stability or new emotion accuracy is claimed. Browser permission and device availability still need a user check. The controller's permission/start/off/restart/track-end/page-exit behavior is covered by six Node tests using a DOM fixture, separately from actual ML inference.

Python tests cover rolling-window expiry, feature geometry and pooling, session isolation, missing/invalid faces, cold loading, pending cancellation, lock ownership, clip precedence, live UI routing, upload ownership and finite-duration normalization. Their final run is recorded in [validation.json](validation.json). The JavaScript asset is included in the built wheel as well as the source package.

MELD television expressions remain a weak basis for interpreting real users. Earlier evidence found no accuracy advantage from adding vision over text alone; this interaction update does not change that finding. Facial signals are tentative and the person's own account takes precedence. No audio, robot integration, reinforcement learning or clinical validation was added.
