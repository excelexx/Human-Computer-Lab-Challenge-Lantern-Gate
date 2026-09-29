# Keeping Mara's replies in the current conversation

29 September 2026. A response to “Oh, fantastic.” treated the line as a surprising
arrival and invented storm banter instead of continuing Mara's route offer.
This update supplies missing context and tightens the authored tone instructions.
It does not change a learned model.

## Changes

The opening speech is now one shared `OPENING_LINE` used by the UI and generator.
On the first game turn, the generator receives it as Mara's prior assistant
message. Previously it existed only as a visual placeholder: the model never
received the line the player was answering. Follow-up turns preserve actual
dialogue without inserting the opening again. Context overflow cannot silently
discard the opening and leave a first reaction without its referent; it requests
a shorter input instead.

A whitelisted `reply_goal` identifies the next conversational step. The authored
crossing question is explicitly optional, an acknowledged route leads to
readiness, and agreed departure leads to an invitation to follow. It contains no
canned NPC answer and cannot execute world actions. The prompt keeps emotional
tone but limits theatrical speculation, speaker-role confusion and new scene
facts. Grounded examples show how to respond to the task.

The deterministic quest, classifiers, camera sampling, evidence freshness and
required parameter count remain unchanged: **4,464,745,375**. Replies still come
from the same local Qwen3-4B model. Prompt examples are authored; the model can
repeat their wording.

## Local comparison

[`evaluate-dialogue-grounding.py`](../../scripts/evaluate-dialogue-grounding.py)
runs 34 fixed cases: seven emotion interventions on one opening at two seeds,
three other sample lines under joy/fear/anger, four custom messages, one
camera-unavailable case, and two complete three-turn route conversations.
Emotion evidence is synthetic; generation uses the real local model. Sampling
matches production, with fixed seeds and nonstreaming transport for comparison.
Full prompts, messages, outputs and finish reasons are retained. This is targeted
development evidence, not a fresh holdout, camera-accuracy evaluation or
human-rated quality study.

The [baseline](grounding-before.json) used the prompt and message construction
from revision `39bf252`. Three intermediate candidates are preserved:
[candidate 1](grounding-candidate-1.json), [candidate 2](grounding-candidate-2.json)
and [candidate 3](grounding-candidate-3.json). They exposed repetition, weaker
emotion differentiation, reversed speaker roles and unsupported safety claims.
They were not promoted. The [selected revision](grounding-selected.json) was
chosen after reviewing the same cases; these are development results.
There were **170 generated responses** across all five runs.

| Input and cue | Baseline behavior | Selected behavior |
|---|---|---|
| “Oh, fantastic.” / surprise | Invented a smile or grin, puzzling over “a fantastic” and a new arrival | “Quite a welcome, I know. Bridge or sea stairs?” at both seeds |
| “You want me to cross that?” / anger | Picked the bridge for the player | Said they did not have to go that way and offered the stairs |
| “Sure. Whatever.” / fear | Added sharp wind and asked a general question | “No rush. I'll stay close. Bridge or sea stairs?” |
| “Lead the way.” / stairs | Repeated breathing language without naming the route | “Sea stairs it is. I’ll lead the way now.” |

Both baseline and selected runs completed 34/34 responses without generation
errors or length-limit endings. Responses containing visible asterisks fell
from 8 to 0; those containing multiple question marks fell from 3 to 0.
These are narrow formatting checks, not an automatic semantic-quality score.

An additional [three production-streaming checks](grounding-streaming.json)
used no seed override. Surprise produced the grounded choice above; fear added
an offer to stay close and go at the player's pace; joy added a warm line about
company making the beacon task less dreary. They also used synthetic evidence,
not a live camera or the user's photograph.

## Verification and limits

**89 affected Python tests passed** across character, quest, generator transport
and context boundaries, UI and reply cues, with one existing dependency warning.
New checks cover the missing opening, preservation of follow-up history,
context overflow, and reply-goal whitelisting.

The selected model still repeats examples, sometimes exceeds two sentences,
and does not make every emotion equally distinct. One explicit-fear case added
“no peaks” to the stairs, which is not an established fact. Some banter remains
awkward. This improves the reproduced conversational failure but does not
guarantee perfect grounding. Existing MELD benchmarks are unchanged.
