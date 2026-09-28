# Lantern Gate: playable local NPC update

This report records the initial game conversion. The subsequent [quest and emotion update](QUEST_UPDATE.md) adds changing choices, actual in-game actions and a revised dialogue prompt; its evidence supersedes the earlier prompt checks for current behavior.

28 September 2026. The previous Check-in interface is now a top-down pixel town.
Walk with WASD/arrows; a large pixel arrow guides the traveler toward Mara. The
floating W/A/S/D tutorial disappears after actual movement. Proximity opens the
encounter: Mara remains visible in the game with a speech bubble above her;
separate floating rectangles on the right contain camera preview, estimated emotion, four sample
replies, and a compact Custom field with a centered send arrow. Desktop dialogue
uses 16 px pixel text; samples and custom input use 14 px. Samples send immediately.
Mara stands at the left side of the gate (world x=260), with a short 130 px stepped pixel
bubble and an enlarged camera extending from the right toward the center. Opening the encounter does not pan or zoom the scene. No outer panel obscures the town.
Escape cancels queued/current dialogue and stops the camera before returning to
movement. E reopens the encounter nearby. Keyboard and pointer/touch controls
are supported, with a Go to Mara shortcut and optional browser full-screen mode.

A [verified scene screenshot](final-scene.png) and [validation record](validation.json)
are included.

## Evidence and scope

- **345 Python tests passed**, with the existing Starlette/AnyIO deprecation
  warning. **12 Node tests passed** across camera lifecycle and new movement
  contracts. Movement tests cover diagonal speed, collision, suspended-frame
  limits, the spawn-to-Mara route, boundaries, and actual-displacement semantics.
  UI tests cover immediate sample submission, camera-mode isolation, stream
  closure, and the existing cancellation/replay/upload contracts.
- **45 real local generations** were retained across three prompt revisions.
  Each revision uses 15 fixed cases: three ambiguous lines crossed with joy,
  fear, anger and missing visual input, two explicit-word disagreement cases,
  and one request to pause roleplay. These are **synthetic evidence interventions**
  into the existing local Qwen generator, not real face predictions, human ratings,
  independent holdouts, or accuracy evaluations. Earlier outputs are retained in
  [initial](controlled-responses-initial.json) and
  [revision 2](controlled-responses-revision2.json); the final outputs are in
  [controlled responses](controlled-responses.json).
- The final cases show different delivery under the same ambiguous text: playful
  language for joy, accompaniment and route explanation for fear, and brief choices
  for anger. The explicit sea-stairs request is acknowledged despite a conflicting
  joy cue. The pause request is honored in the revised prompt. All 15 final
  responses completed without transport/generation errors. These observations
  are narrow qualitative checks, not a general success rate for NPC quality.
- A [synthetic smoothing probe](smoothing-probe.json) compares the original 70%
  new-frame EMA with 85%. On the fixed neutral-to-joy transition, the visible tag
  switches on frame 2 instead of frame 3. At the configured 5 Hz this represents
  one 200 ms interval in this artificial sequence, not a measured webcam gain.
  Smoothing now resets after gaps above 0.4 seconds rather than 0.6 seconds.
  Less smoothing can increase flicker. No neutral-class bias, confidence inflation,
  model retraining or calibration was introduced.

## Multimodal policy

The classifier still emits its actual seven-class MELD result. The optional
`interaction` object records Mara's authored delivery style and its source. For
one of the four exact ambiguous sample lines, eligible visual evidence can choose
the delivery even when text and vision disagree. Other messages follow the
combined estimate, with disagreement handled cautiously. The language model is
instructed to prioritize explicit words and chosen routes. The portrait/sprites,
map and policy rules add no learned weights. The camera tag still needs only one
usable frame; fusion still needs six eligible recent samples. No reference MELD
labels are supplied at inference time. No official test rows were used for tuning
this update, and earlier official-test results remain a reused benchmark.

## Runtime and remaining weaknesses

The stack stays fully local at **4,464,745,375 required learned parameters**.
Pretrained encoders, classifier heads and generator weights are unchanged.
The target is Windows, RTX 3080 10 GB and 32 GB RAM. A warm app/generator snapshot
reported **7,597 MiB GPU memory used out of 10,240 MiB**, including desktop use;
this is a point-in-time observation, not a measured peak for the new game.
Prior latency/accuracy reports belong to the older interaction framing.

The final generator still sometimes adds unsupported route details, mentions
emotion too directly, uses emoji, asks redundant questions, or exceeds the
requested two sentences. Prompting did not eliminate these weaknesses. Expression
estimates are uncertain; they cannot reliably disambiguate intent. Vision has no
demonstrated accuracy advantage over the text baseline, and MELD is not a validated
webcam-game dataset. No new emotion-accuracy claim is made.

This is a small walkable conversation scene, not a complete RPG. There is no combat,
inventory, saved quest progression, or execution of generated world actions.
Camera inference pauses while a reply holds the shared GPU. The camera requires
browser permission and is optional; the scene remains usable with words alone.

## Reproduction and provenance

Use the standard launch instructions in [README](../../README.md). Run the Python
suite with `python -m pytest tests -q` and the browser-side contracts with
`node --test tests/game_world.test.cjs tests/live_camera.test.cjs`.
To repeat the fixed generation interventions, stop the app's GPU workload while
keeping the local generator running, then execute
`python scripts/evaluate-character.py --output .artifacts/reports/character-rerun`.
The result is written to the requested directory without replacing these reports.

The town atlas is Kenney Tiny Town 1.1 (CC0); Silkscreen is locally bundled under
its OFL license. The map, player/NPC sprites, policy and game interaction code are
AI-assisted project work. Asset notices are in [THIRD_PARTY](../../THIRD_PARTY.md).
No new runtime network calls or models were added.
