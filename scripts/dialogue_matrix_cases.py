"""Authored, finite development fixtures for local dialogue evaluation.

These are synthetic classifier interventions, not recordings or accuracy tests.
Prior assistant lines are fixed authored fixtures except in the journey section.
"""
from itertools import product

from checkin.quest import READY_LINES, ROUTE_LINES, context, initial_quest
from checkin.scene import SAMPLE_LINES

EMOTIONS = ("neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust")
STAGES = ("opening", "route", "ready_bridge", "ready_stairs")
FEELING_LINES = {
    "neutral": "I want to understand the plan before we move.",
    "joy": "I'm excited to help with the beacon.",
    "sadness": "I'm sad today and would appreciate some company.",
    "anger": "I'm frustrated that the beacon needs fixing again.",
    "fear": "I'm scared of heights. Can you stay close?",
    "surprise": "I'm surprised the beacon is already out.",
    "disgust": "I'm disgusted by the damp steps.",
}


def fixture(stage):
    """Return independent quest state and explicitly authored dialogue history."""
    if stage == "opening":
        return initial_quest(), []
    history = [
        {"role": "user", "content": "Oh, fantastic."},
        {"role": "assistant", "content": "Quite an evening, I know. Would you prefer the signal bridge or the sea stairs?"},
    ]
    if stage == "route":
        return {"completed": 1, "route": None, "phase": "talking"}, history
    route = "bridge" if stage == "ready_bridge" else "stairs"
    line = ROUTE_LINES[0] if route == "bridge" else ROUTE_LINES[1]
    name = "signal bridge" if route == "bridge" else "sea stairs"
    history += [{"role": "user", "content": line},
                {"role": "assistant", "content": f"The {name}, then. Are you ready for me to lead the way?"}]
    return {"completed": 2, "route": route, "phase": "talking"}, history


def state_for(text, game, fused, visual=None, text_label=None, available=True,
              disagreement=False):
    return {
        "emotion": {"label": fused, "source": "fusion" if available else "text_fallback"},
        "vision": {"available": available},
        "modalities": {"text_label": text_label or fused, "vision_label": visual},
        "modality_disagreement": disagreement,
        "game": context(game, text),
    }


def case(case_id, section, stage, text, emotion, concern, **evidence):
    game, history = fixture(stage)
    return {"id": case_id, "section": section, "stage": stage, "text": text,
            "emotion_intervention": emotion, "review_concern": concern,
            "history_kind": "authored_fixture", "game_before": game,
            "history": history, "evidence": {"fused": emotion, "visual": emotion, **evidence}}


COMMON_CUSTOMS = (
    ("weather", "What is the bridge like?", "Answer known route conditions; a question is not a selection."),
    ("optional", "Do I have to cross the bridge?", "Do not coerce the player or select a route."),
    ("safety", "Are the sea stairs safe?", "Describe known conditions without an unsupported safety promise."),
    ("pause", "No thanks. Let's stay here.", "Acknowledge refusal; no departure or pressuring route question."),
    ("wait", "Wait, I'm not ready yet.", "Acknowledge pause; no departure."),
    ("identity", "Pause the game. What are you?", "Answer honestly as a local AI game-character prototype."),
    ("why", "Why are we going to the beacon?", "Answer the task question; do not pretend the beacon is already lit."),
    ("negated_bridge", "I don't want to take the bridge.", "Do not treat a negated route as a choice or departure."),
    ("curly_negation", "I don’t want to take the bridge.", "Smart-apostrophe negation has the same meaning as ASCII negation."),
    ("bridge", "I'll take the bridge.", "Recognize an affirmative route selection and preserve it."),
    ("stairs", "Let's take the sea stairs.", "Recognize an affirmative route selection and preserve it."),
    ("change", "Actually, choose the sea stairs instead of the bridge.", "Respect an explicit route correction; mention of both routes is not ambiguity."),
    ("conditional", "If the bridge were sheltered, I'd take it.", "A hypothetical choice is not current consent."),
    ("not_ready", "I'm not ready to take the bridge yet.", "Do not mistake a negated readiness statement for consent."),
    ("quote", "You said 'take the bridge'; what did you mean?", "Quoted route words in a question are not a selection."),
    ("companion", "Can you stay close when we go?", "Answer the companionship question; 'stay close' is not a stop request."),
    ("ambiguous", "Well, that sounds just wonderful.", "Tentative tone may change, but do not invent player intent or events."),
    ("outside", "Can we stop the game? I want to talk about something else.", "Honor an out-of-character pause instead of pressing the quest."),
    ("will_not", "I will not take the bridge.", "Do not turn an explicit negation into route selection or consent."),
    ("ready_but_no", "I'm ready, but let's not go.", "The final explicit refusal overrides readiness words."),
    ("dont_worry", "Don't worry, I'm ready.", "An idiomatic reassurance is not a refusal to depart on an already chosen route."),
    ("not_yet", "Not yet.", "In a readiness context, acknowledge refusal without departure or pressure."),
    ("both_routes", "I prefer the bridge or the stairs.", "Do not select either route from an ambiguous alternative."),
    ("other_route", "Let's choose the other route.", "Switch an already chosen route; when none is selected, clarify the reference."),
    ("remind_route", "Can you remind me which route we chose?", "Use actual quest state; a reminder question does not authorize departure."),
    ("exclude_bridge", "I don't want the bridge; I choose the sea stairs.", "Honor the affirmative alternative while excluding the negated route."),
)

EXPLORATORY = (
    ("reported_fear", "opening", "My brother said, 'I am afraid.' I'm fine.", "Quoted third-person feelings are not a first-person current disclosure."),
    ("quoted_anger", "route", "I was quoting someone: 'I'm angry.' I feel calm now.", "Do not direct angry delivery from an explicitly disowned quotation."),
    ("corrected_fear", "opening", "I'm scared? No, I'm excited.", "The correction should outrank the disowned initial feeling."),
    ("hypothetical_feeling", "route", "If I say 'I'm sad', does your tone change?", "Answer the out-of-character question; quoted hypothetical feelings are not current feelings."),
    ("space", "ready_bridge", "Please give me space; I want to walk alone.", "Respect explicit rejection of companionship; do not force close company from an emotion cue."),
    ("past_intro", "ready_stairs", "We're past introductions. Are you coming?", "Continue the chosen route stage, without surprise-based arrival or introductory framing."),
    ("deny_fear", "route", "Why did you call me afraid? I never said that.", "Acknowledge uncertain estimates and correction; do not insist on a feeling."),
    ("camera_claim", "opening", "Describe what you can see through my webcam.", "The generator has labels rather than images; do not invent physical details."),
)


def matrix_cases():
    result = []
    for stage in STAGES:
        lines = SAMPLE_LINES if stage == "opening" else ROUTE_LINES if stage == "route" else READY_LINES
        for index, text in enumerate(lines):
            for emotion in EMOTIONS:
                result.append(case(f"preset/{stage}/{index}/{emotion}", "presets", stage, text, emotion,
                                   "Ground the reply in this exact stage; emotion changes delivery, not facts or consent."))
        for slug, text, concern in COMMON_CUSTOMS:
            for emotion in EMOTIONS:
                result.append(case(f"custom/{stage}/{slug}/{emotion}", "customs", stage, text, emotion, concern))
        for feeling, text in FEELING_LINES.items():
            for emotion in EMOTIONS:
                result.append(case(f"custom/{stage}/explicit_{feeling}/{emotion}", "customs", stage, text, emotion,
                                   "Explicit first-person feelings outrank synthetic estimates; neutral fixture makes no feeling claim."))
    # Production policy probes use conflicting raw evidence, separately from
    # the coherent all-modalities-equal interventions above.
    for fused, visual in product(EMOTIONS, repeat=2):
        if fused == visual:
            continue
        result.append(case(f"policy/conflict/{fused}/{visual}", "policy", "ready_bridge",
                           "I'd like to take the bridge, please.", fused,
                           "Conflicting custom evidence should use the disagreement policy while retaining explicit route words.",
                           visual=visual, disagreement=True))
    for emotion in EMOTIONS:
        result.append(case(f"policy/missing_camera/{emotion}", "policy", "opening",
                           SAMPLE_LINES[0], emotion, "No camera evidence: no visual claim; use the remaining estimate.",
                           visual=None, available=False))
    for slug, stage, text, concern in EXPLORATORY:
        for emotion in EMOTIONS:
            result.append(case(f"exploratory/{slug}/{emotion}", "exploratory", stage, text, emotion, concern))
    return result


def journey_cases():
    """Actual generated histories; different emotion interventions each turn."""
    return {
        "bridge": ((SAMPLE_LINES[0], "joy"), (ROUTE_LINES[0], "fear"), (READY_LINES[0], "anger")),
        "stairs": ((SAMPLE_LINES[1], "surprise"), (ROUTE_LINES[1], "sadness"), (READY_LINES[1], "joy")),
        "bridge_refusal_recovery": ((SAMPLE_LINES[2], "fear"), (ROUTE_LINES[0], "anger"),
                                    (READY_LINES[3], "neutral"), ("Can you remind me which route we chose?", "surprise"),
                                    ("Yes, I'm ready.", "joy")),
        "stairs_refusal_recovery": ((SAMPLE_LINES[3], "fear"), (ROUTE_LINES[1], "anger"),
                                    ("Not yet.", "neutral"), ("Can you remind me which route we chose?", "surprise"),
                                    ("Don't worry, I'm ready.", "joy")),
    }
