# Faster live emotion feedback

25 September 2026. The prior live view waited for six camera frames and pooled up to four seconds of expressions. This introduced seconds of delay before the first label and made subsequent expression changes slow to affect it.

The live display now emits a tentative tag from the **first usable face frame**. Camera capture targets **5 Hz** (0.2 seconds); the display polls at 0.1 seconds. The visual model warms before the app becomes ready, without loading the large text encoder. Brief probability smoothing weights the newest frame at 70% and the previous estimate at 30%, only when their gap is at most 0.6 seconds. Missing/ambiguous faces, face-track changes and resets clear smoothing. A bounded stream session now lasts 30 seconds instead of 10, reducing periodic admission overhead.

The conversational fusion path still requires six valid samples among at most eight recent samples. A fast visible tag does not make unready fusion evidence eligible for Send. Its `fusion_ready` diagnostic states that distinction. New conversation rotates the camera token as well as the server session, so an older queued frame cannot become the new conversation's first label. Reply generation and camera inference remain serialized on the GPU; display updates can pause while the reply holds it.

## Measured comparison

The [backend timing record](backend.json) compares the exact previous committed implementation (`d2ae25acfa8595d922830cc05f4f0488e8e3a666`) against this implementation using the same detector, encoder, trained head and one existing MELD training image. Both runs used the same already-warm encoder instance sequentially, with 30 scheduled callbacks each, at their configured cadence. No test emotion labels were used or models selected.

| Backend measurement | Previous | Faster display |
|---|---:|---:|
| First tag, from initial callback start | 2,523.8 ms | 46.6 ms |
| First tag's frame | 6 | 1 |
| First fusion-ready frame | 6 | 6 |
| Median callback processing | 23.1 ms | 25.9 ms |
| Callback processing p95 | 29.1 ms | 36.4 ms |

The main gain comes from removing the wait and long display pooling, not from accelerating the encoder. One-time visual startup took 3.12 seconds and is now paid before the ready page opens. The backend measurements exclude webcam capture, transport and rendering. CPU software tests ran alongside this bounded timing comparison, so its small processing differences are not an isolated hardware optimization result.

## Physical browser check

The real webcam streamed into the updated local app at 640×480. A fresh conversation produced a visible tag within **289 ms of clicking New conversation**, from one valid sample, while `fusion_ready` correctly remained false. Later readings held eight samples across **1.406 and 1.391 seconds**—approximately five processed frames per second. Sampled backend processing times were **14.6–38.5 ms**; displayed observation ages were **62–172 ms**. These are sparse engineering observations, not capture-to-display percentiles or proof of correct emotion recognition.

The stream continued past its 30-second session rollover and beyond 333 callbacks after reset. Camera images, visual features, predicted user labels and probabilities were not saved in the timing report. See [browser.json](browser.json) for the numeric observations.

## Validation and limits

The complete software suite passed **335 tests**, with one existing Starlette/AnyIO deprecation warning. Tests cover first-frame display, short smoothing and resets, separate six-sample fusion eligibility, startup lock ownership, invalid inputs, stale observations and cancellation. The final validation record also includes any focused follow-up checks in [validation.json](validation.json). The original accuracy reports and model files are unchanged.

The complete required learned-parameter upper bound remains **4,464,745,375**. All inference stays local on the Windows RTX 3080 10 GB / 32 GB RAM machine. This update adds no learned model and makes no new peak-memory measurement or emotion-accuracy claim. Faster tentative estimates may fluctuate. MELD-trained expressions are still uncertain evidence about a user's feelings.
