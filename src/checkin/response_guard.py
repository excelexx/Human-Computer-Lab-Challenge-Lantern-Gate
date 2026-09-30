"""Bounded checks for observed NPC failures; not a general factuality judge."""
import re

from . import quest

REPAIR = {
    "question": "End with a statement; no question or request to choose/leave.",
    "length": "Write at most two sentences and 45 words.",
    "echo": "Do not repeat the player's line as Mara's own words.",
    "grounding": "Use only established route conditions; no new hazards, promises, deadlines, or ship outcomes.",
    "perception": "You receive words and an estimated tag, no image. Answer only that perception question, without quest commentary.",
    "pause": "Both remain at the gate. Respect the pause; do not suggest departure.",
    "inspection": "Say you do not know whether it was inspected. Unknown records do not mean absent records.",
    "route": "Use the authoritative selected route and do not replace it or ask again after departure consent.",
    "judgment": "Do not judge the player's reaction, interpret it as strange, or mock their wording.",
    "correction": "Accept the correction about feelings; do not return to the beacon task or route choice.",
}


def problems(text, state, response):
    value = response.casefold().replace("’", "'")
    line = text.casefold().replace("’", "'")
    intent = quest.player_intent(text)
    game = quest.safe_context(state.get("game"))
    found = []
    opening_example = game.get("completed") == 0 and text.strip().casefold() in {s.casefold() for s in quest.SAMPLE_LINES}
    can_ask = not (intent["question"] or intent["pause"] or game.get("next_action") == "walk_and_relight") and (
        opening_example or intent["route"] or intent["switch_route"] or intent["uncertain"])
    if not can_ask and "?" in response:
        found.append("question")
    sentences = re.split(r"[.!?]+(?:\s+|$)", response.strip())
    if sum(bool(s.strip()) for s in sentences) > 2 or len(response.split()) > 45:
        found.append("length")
    if text.strip().casefold() in {s.casefold() for s in quest.ALL_EXAMPLES} and value.startswith(line.strip().rstrip('.!?')):
        found.append("echo")
    if re.search(r"\b(?:narrow|winding|slippery|slip|slipping|gale|crash|crashing|rocks|railing|rails|solid footing|safe passage|no surprises|stormy nights|night deepens|drifting into danger)\b", value):
        found.append("grounding")
    for claim in re.finditer(r"\b(?:safe|safer|safely|safest)\b", value):
        prefix = re.split(r"[.!?;]", value[:claim.start()])[-1]
        if not re.search(r"\b(?:not|no|unknown|can't|cannot|don't|neither)\b", prefix):
            found.append("grounding")
    if re.search(r"\b(?:keep|still) (?:the )?beacon (?:lit|flickering)|\blantern(?:'s| is) out\b", value):
        found.append("grounding")
    if re.search(r"\b(?:i (?:can |did |didn't )?see|you look|your face shows)\b[^.!?]*(?:smil|frown|fear|face|wear|emotion|calm|happy|worried|sad|angry)", value):
        found.append("perception")
    perception = intent["question"] and re.search(r"\b(?:camera|webcam|face|wearing|clothes|appearance)\b", line)
    if perception and re.search(r"\b(?:beacon|bridge|stairs|lantern)\b", value):
        found.append("perception")
    if perception and re.search(r"\b(?:camera|webcam|face)\b", line) and not re.search(r"\b(?:tag|cue|estimate|estimated)\b", value):
        found.append("perception")
    if intent["pause"] and re.search(r"\b(?:let's (?:go|take|leave)|go on without|lead (?:the|you)|we(?:'ll| will|'re| are) (?:go|leave|start)|ready to go)\b", value):
        found.append("pause")
    if re.search(r"\b(?:inspect|inspected|inspection|checked|tested)\b", line) and not re.search(r"\b(?:don't know|do not know|unknown|cannot confirm|can't confirm|not know)\b", value):
        found.append("inspection")
    if re.search(r"\b(?:no inspection|no (?:inspection )?records exist|was never inspected)\b", value):
        found.append("inspection")
    if game.get("next_action") == "walk_and_relight":
        route = game.get("route")
        if route and (route not in value or ("stairs" if route == "bridge" else "bridge") in value):
            found.append("route")
        if re.search(r"\b(?:ready when you are|when you're ready|when you are ready|are you ready)\b", value):
            found.append("route")
    if re.search(r"\b(?:strange thing to say|bold take|not inviting|calm down)\b", value):
        found.append("judgment")
    correction = re.search(r"\b(?:i'm|i am) not (?:afraid|scared|nervous|sad|angry|happy|excited|disgusted)\b", line)
    if correction and not re.search(r"\b(?:beacon|bridge|stairs|lantern)\b", line) and re.search(r"\b(?:beacon|bridge|stairs|lantern)\b", value):
        found.append("correction")
    return list(dict.fromkeys(found))


def fallback(text, state):
    """Explicitly marked authored fallback after two rejected model drafts."""
    game = quest.safe_context(state.get("game"))
    intent = quest.player_intent(text)
    line = text.casefold().replace("’", "'")
    route = "signal bridge" if game.get("route") == "bridge" else "sea stairs" if game.get("route") == "stairs" else None
    style = state.get("interaction", {}).get("response_style", "practical")
    endings = {"playful": "I'm glad to have your company.", "careful": "We can take it at your pace.",
        "patient": "There's no need to rush.", "steady": "I'll keep it straightforward.",
        "curious": "One step at a time.", "wry": "An inconvenient chore, admittedly.", "practical": "The choice is yours."}
    if intent["question"] and re.search(r"\b(?:camera|webcam|face|wearing|clothes|appearance)\b", line):
        return "I receive your words and an estimated emotion tag, not a camera image. I can't see your appearance."
    if re.search(r"\b(?:inspect|inspected|inspection|checked|tested)\b", line):
        return "I don't know whether it was inspected. I can't confirm its safety."
    if intent["question"] and "help" in line and re.search(r"\b(?:have to|must|need to|required|forced)\b", line):
        return "Helping is optional. You're welcome to stay at the gate."
    if "why" in line and "beacon" in line:
        return "The beacon marks the harbor for navigation, and it's out after the storm. Helping to relight it is optional."
    if intent["question"] and re.search(r"\b(?:which|what) route\b", line):
        return f"You chose the {route}; we're still at the gate." if route else "No route has been chosen yet."
    if intent["pause"] or "go without me" in line:
        if intent["question"] and re.search(r"\b(?:what happens|never go|stay here)\b", line):
            return "We can stay at the gate. The beacon remains out until it is relit."
        return "We can pause here; I'll wait at the gate."
    if re.search(r"\b(?:i'm|i am) not (?:afraid|scared|nervous|sad|angry|happy|excited|disgusted)\b", line):
        return "Thanks for clarifying; I'll follow what you tell me rather than guessing from the emotion tag."
    if game.get("next_action") == "walk_and_relight" and route:
        return f"I'll lead you along the {route} now. " + endings[style]
    if intent["question"] and ("wrong" in line or re.search(r"\b(?:safe|safety|risk|dangerous)\b", line)):
        return "I can't vouch for safety: the bridge is windy and exposed, while the stairs are damp and sheltered. " + endings[style]
    if intent["route"] and route:
        return f"We'll plan on the {route} and wait until you're ready. " + endings[style]
    return "We can take our time deciding what to do. " + endings[style]
