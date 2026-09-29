"""Present the direction used for one NPC reply, independently of live camera tags."""

import html

from .character import STYLES

REACTIONS = {
    "playful": ("Playful", "Mara grins and raises her lantern."),
    "careful": ("Reassuring", "Mara offers an open hand."),
    "steady": ("Direct", "Mara gives a firm nod."),
    "patient": ("Gentle", "Mara rests a hand over her heart."),
    "curious": ("Curious", "Mara raises an eyebrow."),
    "wry": ("Wry", "Mara gives a crooked smile."),
    "practical": ("Practical", "Mara gives a small nod."),
}


def reply_cue_html(state=None):
    """Only show stored, whitelisted turn direction; never guess from a live tag."""
    state = state if isinstance(state, dict) else {}
    direction = state.get("interaction")
    direction = direction if isinstance(direction, dict) else {}
    style = direction.get("response_style")
    source = direction.get("direction_source")
    cue = direction.get("cue_emotion")
    if style not in REACTIONS or source not in {
        "combined_estimate", "ambiguous_demo_visual_cue", "explicit_player_words", "modality_disagreement"
    }:
        return '<div id="reply-cue-signal" data-reaction="idle" data-ready="false">Awaiting your line</div>'
    visual = (state.get("vision") or {}).get("available") is True
    name, pose = REACTIONS[style]
    cue_name = cue.capitalize() + " cue" if cue in STYLES else "Emotion cue"
    if source == "modality_disagreement":
        cue_name, name = "Mixed cues", "Clarifying"
        origin = "Words + vision disagree"
    elif source == "explicit_player_words":
        origin = "Your stated feeling"
    elif source == "ambiguous_demo_visual_cue" and visual:
        origin = "Vision estimate"
    elif source == "combined_estimate":
        origin = "Text + vision estimate" if visual else "Text estimate · no usable camera cue"
    else:
        # Inconsistent evidence must never be advertised as a camera-driven reply.
        return '<div id="reply-cue-signal" data-reaction="idle" data-ready="false">Awaiting your line</div>'
    turn = html.escape(f"{state.get('session_id', '')}:{state.get('turn_id', '')}", quote=True)
    description = html.escape(f"{origin}. {pose}", quote=True)
    return (
        f'<div id="reply-cue-signal" data-reaction="{style}" data-ready="true" data-turn="{turn}" '
        f'role="status" aria-live="polite" title="{description}">'
        f'<strong>{cue_name} <span aria-hidden="true">→</span> {name}</strong>'
        '</div>'
    )
