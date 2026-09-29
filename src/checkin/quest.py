"""Deterministic three-turn demo: authored choices, never model-issued actions."""
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


def initial_quest():
    return {"completed": 0, "route": None, "phase": "talking"}


def options(quest):
    if quest["completed"] == 0:
        return SAMPLE_LINES
    if quest["completed"] == 1 or quest.get("route") is None:
        return ROUTE_LINES
    return READY_LINES


def preview(quest, text):
    """Tentative next state; commit only after successful response completion."""
    result = dict(quest)
    words = text.strip().casefold()
    # A pause/refusal takes precedence over route words, including custom text.
    pause = bool(re.search(r"\b(stop|pause|wait|stay|don't|do not|not ready|rather not|no thanks)\b", words))
    bridge = bool(re.search(r"\bbridge\b", words))
    stairs = bool(re.search(r"\b(stairs|stairway)\b", words))
    # Examples are exact authored selections. Custom route instructions must be
    # affirmative and unambiguous; merely asking about a bridge is not a choice.
    example_route = words in {line.casefold() for line in ROUTE_LINES}
    explicit = bool(re.search(r"\b(take|choose|prefer|pick|go via|use)\b", words)) and "?" not in words
    if not pause and bridge != stairs and (example_route or explicit):
        result["route"] = "bridge" if bridge else "stairs"
    result["completed"] = min(3, quest["completed"] + 1)
    ready = words in {line.casefold() for line in READY_LINES[:3]} or bool(re.search(r"\b(let's go|lets go|lead the way|i'm ready|i am ready|let's do this)\b", words)) or (explicit and bridge != stairs)
    result["phase"] = "depart" if result["completed"] >= 3 and result["route"] and ready and not pause else "talking"
    return result


def context(quest, text):
    candidate = preview(quest, text)
    return {"completed": quest["completed"], "route": candidate["route"],
            "next_action": "walk_and_relight" if candidate["phase"] == "depart" else "confirm_departure" if candidate["completed"] >= 2 and candidate["route"] else "choose_route"}


def safe_context(value):
    if not isinstance(value, dict):
        return {}
    count = value.get("completed")
    return {"completed": count if type(count) is int and 0 <= count <= 3 else 0,
            "route": value.get("route") if value.get("route") in ("bridge", "stairs") else None,
            "next_action": value.get("next_action") if value.get("next_action") in ("walk_and_relight", "confirm_departure", "choose_route") else "choose_route"}


def note(quest):
    if quest["phase"] == "depart":
        return "Mara will lead the way after this reply. The journey starts in a moment."
    return "Mara’s response is generated live from your words and emotion estimate."


def signal(quest, session_id):
    return f'<div id="quest-signal" data-session="{html.escape(str(session_id), quote=True)}" data-count="{quest["completed"]}" data-route="{quest.get("route") or ""}" data-phase="{quest["phase"]}"></div>'
