"""Deterministic quest choices: explicit player consent, never model-issued actions."""
import html
import re

from .scene import SAMPLE_LINES

ROUTE_LINES = (
    "Fine. The bridge it is.", "The sea stairs, then.",
    "That bridge sounds inviting.", "Those stairs sound delightful.",
)
READY_LINES = (
    "Lead the way.", "Let's do this.",
    "Ready as I'll ever be.", "Actually, let's stay here.",
)
ALL_EXAMPLES = SAMPLE_LINES + ROUTE_LINES + READY_LINES
_SPACE_REQUEST = re.compile(r"\b(?:give me (?:some )?space|leave me alone|(?:don't|do not) (?:follow|stay close)|keep your distance|stay back|stop following|go away)\b")
_OUTSIDE_REQUEST = re.compile(r"\b(?:talk about something else|talk something else|change the subject|stop roleplaying|stop role-playing)\b")


def initial_quest():
    return {"completed": 0, "route": None, "phase": "talking"}


def options(quest):
    if quest["completed"] == 0:
        return SAMPLE_LINES
    if quest.get("route") is None:
        return ROUTE_LINES
    return READY_LINES


def _words(text):
    return re.sub(r"\s+", " ", text.casefold().translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": ";", "–": ";"}))).strip()


def player_intent(text):
    """Conservative authored command grammar, independent of estimated emotion.

    This is intentionally not a general language understanding model. Questions,
    reported speech and conditional plans never grant movement permission.
    Selecting a preference alone is also separate from agreeing to leave now.
    """
    words = _words(text)
    # Quoted commands describe words, not the player's consent. Apostrophes in
    # contractions are retained; double-quoted spans are excluded from actions.
    actions = re.sub(r'"[^"\n]*"', "", words)
    actions = re.sub(r"\b(?:don't|do not) worry\b", "", actions)
    clauses = [re.sub(r"^(?:(?:okay|ok|well|actually|fine|yes|then|please|so)\s+)+", "", part.strip())
               for part in re.split(r"[.!?;,\n]|\b(?:but|and)\b", actions) if part.strip()]
    question = "?" in words or bool(re.match(r"^(?:i (?:want|need|would like) to (?:know|understand)|tell me|explain|describe|what|why|how|who|where|when|which|are you|do you|does|can (?:you|we|i)|could|should|would)\b", words))
    switch_route = bool(re.fullmatch(r"(?:let's|lets|please|i want to) (?:choose|take|try|switch to) the other route[.!]*", words))
    uncertain = bool(re.search(r"\b(?:if|unless|maybe|perhaps|might|could|suppose|hypothetically|not sure|changed my mind|other route)\b", actions))
    deferred = bool(re.search(r"\b(?:later|tomorrow|after|eventually|someday|not yet)\b", actions))
    pause = bool(re.search(
        r"\b(?:not (?:quite |really |yet )?ready|not yet|not now|no thanks|rather not|"
        r"stay (?:here|put|back|at the gate)|staying here|"
        r"(?:let's|lets) not (?:go|leave|move|depart|start|continue)|"
        r"(?:don't|do not|won't|will not|cannot|can't) (?:want to )?(?:go|leave|move|depart|start|continue|proceed))\b", actions
    )) or any(re.match(r"^(?:stop|pause|wait|hold on)\b", clause) or clause == "no" for clause in clauses)
    pause = pause or bool(_SPACE_REQUEST.search(actions) or _OUTSIDE_REQUEST.search(actions))
    route_names = {"bridge": r"(?:signal )?bridge", "stairs": r"(?:sea )?(?:stairs|stairway)"}
    rejected = set()
    for name, pattern in route_names.items():
        for clause in clauses:
            rejection_clause = re.sub(r"\bnot (?:quite |really |yet )?ready\b", "unready", clause)
            if re.search(r"\b(?:not|don't|do not|won't|will not|never|no|avoid|rather than|instead of)\b[^.!?;,]{0,45}\b" + pattern + r"\b", rejection_clause):
                rejected.add(name)

    selected = []
    ambiguous_route = False
    ready = False
    if not question and not uncertain:
        for line, route in zip(ROUTE_LINES, ("bridge", "stairs", "bridge", "stairs")):
            if words == _words(line):
                selected.append(route)
        for clause in clauses:
            match = re.match(
                r"^(?:(?:i|we)(?:'ll| will| would| would like to| want to|'d(?: like to| rather)?)?\s+|let's\s+|lets\s+)?"
                r"(?P<verb>take|choose|pick|use|prefer|go via|go by|switch to|go with|head for|head via)\s+"
                r"(?:the\s+)?(?P<route>(?:signal\s+)?bridge|(?:sea\s+)?stairs|stairway)\b", clause
            )
            if match:
                route = "bridge" if "bridge" in match["route"] else "stairs"
                selected.append(route)
                mentioned = {name for name, pattern in route_names.items() if re.search(r"\b" + pattern + r"\b", clause)}
                ambiguous_route = ambiguous_route or len(mentioned - rejected) > 1
                # Choosing/preferring is not permission to start walking.
                ready = ready or match["verb"] not in ("choose", "pick", "prefer")
            ready = ready or bool(re.match(r"^(?:i'm ready|i am ready|let's go|lets go|lead the way|let's do this|lets do this|ready as i'll ever be)\b", clause))
    choices = set(selected) - rejected
    route = next(iter(choices)) if len(choices) == 1 and not ambiguous_route else None
    # Conflicting affirmative route commands need clarification, not an
    # arbitrary first/last route selection or departure on the previous route.
    conflict = len(set(selected)) > 1 or ambiguous_route
    return {"route": route, "ready": bool(ready and not (pause or question or uncertain or deferred or conflict or (rejected and not route))),
            "pause": pause, "question": question, "uncertain": (uncertain and not switch_route) or deferred or conflict, "switch_route": switch_route,
            "rejected_routes": tuple(sorted(rejected))}


def preview(quest, text):
    """Tentative next state; commit only after successful response completion."""
    result = dict(quest)
    intent = player_intent(text)
    if not (intent["question"] or intent["uncertain"]) and result.get("route") in intent["rejected_routes"]:
        result["route"] = None
    if intent["switch_route"] and quest.get("route") in ("bridge", "stairs"):
        result["route"] = "stairs" if quest["route"] == "bridge" else "bridge"
    if intent["route"]:
        result["route"] = intent["route"]
    result["completed"] = min(3, quest["completed"] + 1)
    # Presets still form the authored three-turn path. A custom route choice
    # followed by explicit readiness may finish in two; do not ask twice.
    can_depart = result["completed"] >= 3 or (result["completed"] >= 2 and quest.get("route") is not None)
    result["phase"] = "depart" if can_depart and result["route"] and intent["ready"] else "talking"
    return result


def context(quest, text):
    candidate = preview(quest, text)
    return {"completed": quest["completed"], "route": candidate["route"],
            "next_action": "walk_and_relight" if candidate["phase"] == "depart" else "confirm_departure" if candidate["route"] else "choose_route"}


def safe_context(value):
    if not isinstance(value, dict):
        return {}
    count = value.get("completed")
    return {"completed": count if type(count) is int and 0 <= count <= 3 else 0,
            "route": value.get("route") if value.get("route") in ("bridge", "stairs") else None,
            "next_action": value.get("next_action") if value.get("next_action") in ("walk_and_relight", "confirm_departure", "choose_route") else "choose_route"}


def dialogue_goal(value, text):
    """Specify a conversational purpose, never a canned answer or game action."""
    game = safe_context(value)
    if not game:
        return "Answer the player's actual message."
    route = {"bridge": "signal bridge", "stairs": "sea stairs"}.get(game["route"])
    intent = player_intent(text)
    line = _words(text)
    pause_goal = "The player asked to pause or stay: remain at the gate and do not ask them to choose or leave yet. A selected route is only a plan, not permission to move."
    question_goal = "Asking is not agreement to travel. Do not start walking, assume a route change, or replace the answer with a readiness question."

    def answer(focus):
        return focus + " " + (pause_goal if intent["pause"] else question_goal)

    # Give the local generator a short, relevant fact set for common questions.
    # These are purposes and facts, not completed NPC replies. The strings come
    # from the app; no player-authored instructions are copied into the goal.
    if re.search(r"\b(?:(?:what|who) are you|are you (?:an? )?(?:ai|bot|human|person|real))\b", line):
        return answer("Answer the out-of-character question: you are a local AI game-character prototype playing Mara. State that identity clearly and conclude the answer. The conversation can remain paused at the gate.")
    if intent["question"] and re.search(r"\byou (?:call(?:ed)? me|said i (?:am|was)|say i'm)\b", line) and re.search(r"\b(?:afraid|scared|nervous|sad|angry|happy|excited|disgusted)\b", line):
        return answer("Accept the player's correction about their feelings. Explain briefly that an estimated emotion cue can be wrong, and their own words take priority. Acknowledge the mistaken assumption and conclude this explanation.")
    if intent["question"] and re.search(r"\b(?:tone|delivery|response|responses|reply)\b", line) and re.search(r"\b(?:emotion|emotions|feel|feeling|sad|afraid|angry|happy|excited)\b", line):
        return answer("Answer the out-of-character question about emotion conditioning: tentative emotion estimates can change Mara's warmth, pace and phrasing. The player's explicit words take priority. This is an uncertain cue, not knowledge of their inner feelings; a hypothetical example is only an example.")
    if intent["question"] and (re.search(r"\b(?:camera|webcam|image|face)\b", line) or re.search(r"\b(?:(?:can|do) you see|what (?:can|do) you see)\b", line)):
        return answer("Answer the player's actual question about perception: the local language model receives their typed words and an estimated emotion tag, not a camera image. The estimate can be wrong; do not claim to see their face or know their feelings. Do not resume the quest.")
    if _SPACE_REQUEST.search(line):
        return "Respect the player's request for space. Mara will keep her distance and wait at the gate. Do not offer to stay close, follow them, or ask them to depart; an emotion cue does not override this request."
    if _OUTSIDE_REQUEST.search(line):
        return answer("Honor the request to leave the fictional conversation. Pause the game interaction and invite the subject they want to discuss. Do not add beacon, route, lantern or travel pressure.")
    if re.search(r"\b(?:you (?:seem|sound|are|look)|you're|stop|don't|do not)\b.*\b(?:eager|rush|rushing|hurry|hurrying|push|pushing|pressure|pressuring)\b", line):
        return "Respond to the player's concern that Mara is rushing or pressuring them. Acknowledge your own impatient delivery and make clear you can wait. End there; do not ask about routes or readiness, infer a choice, or defend the pressure."
    if re.search(r"\b(?:explain|explanation|plan|instructions)\b", line) and re.search(r"\b(?:simple|simply|brief|briefly|short|shorter|plain|plainly)\b", line):
        return answer("Give the requested simple explanation: the beacon is out and the optional task is to reach it and relight it. Explain briefly rather than merely acknowledging the request. Both are still at the gate. End the explanation without a route or readiness question.")
    if re.search(r"\b(?:suspicious|suspiciously|too easy|catch|skeptical|sceptical)\b", line):
        return "Answer the player's skepticism about the task directly. Explain that the goal is relighting the beacon, while how difficult the journey will be is unknown. Do not promise an easy trip or invent a hidden catch. End without asking them to choose a route."
    if intent["question"] and re.search(r"\b(?:safe|safer|safety|dangerous|risk)\b", line):
        known = "The bridge is short, windy and exposed; the sea stairs are longer, damp and sheltered."
        if re.search(r"\b(?:stairs|stairway)\b", line) and "bridge" not in line:
            known = "The sea stairs are longer, damp and sheltered."
        elif "bridge" in line and not re.search(r"\b(?:stairs|stairway)\b", line):
            known = "The signal bridge is short, windy and exposed."
        return answer("Answer the player's actual question about safety: whether either route is safe or unsafe is unknown. Say you cannot vouch for safety. " + known + " Do not add evidence of an inspection, rails, missing signs, stability, heights, or a comparison of safety.")
    if intent["question"] and re.search(r"\bwhy\b", line) and re.search(r"\b(?:beacon|going|relight|help)\b", line):
        return answer("Answer the player's actual question about the purpose: the harbor beacon is out after the storm, and the offered task is to reach it and relight it. Helping is optional. Do not invent fog, a deadline, previous failures or consequences for ships.")
    if intent["question"] and (re.search(r"\bremind\b.*\broute\b", line) or re.search(r"\b(?:which|what) route\b.*\b(?:choose|chose|chosen|picked|selected)\b", line)):
        fact = f"The selected route is the {route}." if route else "No route has been chosen yet; do not guess one."
        return answer("Answer the player's actual question about the existing plan. " + fact + " This reminder does not change the plan or grant permission to move.")
    if intent["question"] and "bridge" in line and re.search(r"\byou (?:said|told|meant)\b", line):
        return answer("The player asks what YOUR earlier words meant. Explain that you offered the bridge as an optional route to the beacon; the player can decide which route they prefer. This is clarification of Mara's offer, not a route choice by the player. Conclude the explanation.")
    if intent["question"] and re.search(r"\b(?:what (?:is|are)|tell me about|describe)\b", line) and re.search(r"\b(?:bridge|stairs|stairway)\b", line):
        known = "The signal bridge is short, windy and exposed."
        if re.search(r"\b(?:stairs|stairway)\b", line):
            known = "The sea stairs are longer, damp and sheltered." if "bridge" not in line else known + " The sea stairs are longer, damp and sheltered."
        return answer("Answer the player's actual question by describing only these established conditions: " + known + " Conclude after those conditions.")
    if line == _words(SAMPLE_LINES[1]) or (intent["question"] and re.search(r"\b(?:have to|must|need to|want me|forced|expected)\b", line) and re.search(r"\b(?:cross|bridge)\b", line)):
        return answer("Answer directly whether crossing the bridge is required: it is optional. The player may choose the sea stairs or stay at the gate. Do not choose an alternative for them or make safety claims.")
    if intent["question"] and intent["pause"]:
        return "Answer the question about waiting: both can remain at the gate, and the beacon remains out until it is relit. The player can take their time. Do not invent a deadline, penalty or danger from waiting. End this answer without asking them to choose a route or depart."
    if intent["question"] and re.search(r"\b(?:plan|what (?:happens|are we doing|do we do))\b", line):
        return answer("Explain the actual plan briefly: the beacon is out, Mara can accompany the player by their chosen route, and together they can relight it after traveling. The bridge is short and windy; the sea stairs are longer, damp and sheltered. Wait for the player's route choice and agreement before moving.")
    if re.search(r"\b(?:are you coming|will you join)\b", line):
        plan = f"The future plan is the {route}." if route else "The player can still choose the route."
        return answer("Answer the request directly: Mara is available to accompany the player when they agree to leave. Both are still at the gate. " + plan + " Respond to the invitation rather than repeating introductions.")
    if re.search(r"\b(?:stay close|stay with me|walk beside|walk with me|some company|accompany|go with me|come with me|will you come)\b", line):
        if route and game["next_action"] == "walk_and_relight":
            return f"Departure is agreed on the {route}. The player also requested companionship. Name the selected {route}, say you will lead them now, and offer to stay beside them at their pace. Do not ask for readiness again or claim arrival."
        return answer("Answer the request for companionship: Mara can walk beside the player at their pace when they choose to go. Offer that directly without inventing a rail, height, hazard or safety guarantee. Company does not choose a route or grant departure consent.")
    if intent["pause"]:
        return pause_goal + " Answer any accompanying direct or out-of-character question honestly; do not restart the quest."
    if intent["question"]:
        if line == _words(SAMPLE_LINES[3]):
            focus = "Explain that you cannot predict the journey. The known conditions are a short, windy, exposed bridge and longer, damp, sheltered sea stairs. Describe those conditions briefly and conclude the answer."
        else:
            focus = "Answer the player's actual question or request directly in the selected tone, using only the known scene facts."
        return answer(focus)
    if intent["switch_route"] and route:
        old_route = "sea stairs" if game["route"] == "bridge" else "signal bridge"
        return f"The player changed the plan FROM the {old_route} TO the {route}. Acknowledge the NEW {route} and ask if they are ready. The previous {old_route} plan is superseded. Do not confirm the old route or start walking yet."
    if intent["switch_route"] and not route:
        return "No route has been chosen, so the other route has no established reference. Ask one brief clarification: do they mean the signal bridge or the sea stairs? Do not guess, claim a selection or start walking."
    if intent["uncertain"]:
        if "bridge" in line and re.search(r"\b(?:stairs|stairway)\b", line) and re.search(r"\bor\b", line):
            return "The player named both possible routes without choosing one. Ask one brief clarification about which route they prefer: bridge or sea stairs. Any existing route is only the previous plan; resolve this new ambiguity before confirming it. Both remain at the gate."
        if "bridge" in line and "sheltered" in line and re.search(r"\bif\b", line):
            return "Respond to the hypothetical using the established conditions: the bridge is exposed and windy; the sea stairs are sheltered, longer and damp. The hypothetical is not a route selection. Leave the choice with the player and remain at the gate."
        return "The player's message is conditional, deferred, or ambiguous. Answer their actual concern in the selected tone without treating it as agreement to leave now. If the route is unclear, ask one brief clarification."
    if intent["rejected_routes"] and not intent["route"]:
        plan = f"The {route} remains the existing plan, but this message is not departure consent." if route else "No route is selected."
        declined = " and ".join({"bridge": "signal bridge", "stairs": "sea stairs"}[item] for item in intent["rejected_routes"])
        return f"Acknowledge that the player declined the {declined}. Respect that refusal. " + plan + " There is no new agreement to travel. Both remain at the gate; wait for the player's preference."
    if route and game["next_action"] == "walk_and_relight":
        return f"Departure is agreed on the {route}. This is the current route even if earlier dialogue mentioned another; follow this latest selection. In the selected tone, name the {route} and tell the player you will lead them now. Do not ask for readiness again or claim arrival."
    if route and game["next_action"] == "confirm_departure" and intent["route"]:
        return f"Both are still at the gate. The latest selected route is the {route}, a future plan superseding any earlier route. Acknowledge that plan in the selected tone, then ask whether they are ready. Keep the answer focused on the plan and readiness. Do not switch routes."
    if game["completed"] == 0 and text.strip().casefold() in {item.casefold() for item in SAMPLE_LINES}:
        return "Acknowledge their reaction to the beacon task in the selected tone. This is the opening demonstration: one brief route choice is enough. Do not repeat the briefing, invent an arrival, or select a route."
    plan = f"The {route} is only the existing plan; this message does not change it or authorize departure." if route else "No route has been selected. Do not announce either route as agreed."
    return "Respond specifically to the meaning of this latest remark in the ongoing conversation, using the chosen tone. Address any feeling or concern they express. " + plan + " Both remain at the gate. Do not repeat an earlier acknowledgment or append a route/readiness question; wait for the player to bring the plan forward."


def note(quest):
    if quest["phase"] == "depart":
        return "Mara will lead the way after this reply. The journey starts in a moment."
    return "Mara’s response is generated live from your words and emotion estimate."


def signal(quest, session_id):
    return f'<div id="quest-signal" data-session="{html.escape(str(session_id), quote=True)}" data-count="{quest["completed"]}" data-route="{quest.get("route") or ""}" data-phase="{quest["phase"]}"></div>'
