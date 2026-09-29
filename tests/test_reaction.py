"""Reply provenance must follow the actual direction, not an unrelated live label."""
import pytest

from checkin.character import character_context
from checkin.generator import _emotion_evidence
from checkin.reaction import reply_cue_html


def turn(message="Oh, fantastic.", visual="fear", fused="neutral"):
    state = {"session_id": "s", "turn_id": "1", "emotion": {"label": fused, "source": "fusion" if visual else "text_fallback"},
             "vision": {"available": bool(visual)}, "modalities": {"text_label": "neutral", "vision_label": visual},
             "modality_disagreement": bool(visual and visual != "neutral")}
    state["interaction"] = character_context(_emotion_evidence(state), message)
    return state


@pytest.mark.parametrize("label,style,name", [("fear", "careful", "Reassuring"), ("joy", "playful", "Playful"), ("anger", "steady", "Direct")])
def test_authored_ambiguous_line_reports_selected_visual_cue_not_fused_label(label, style, name):
    state = turn(visual=label)
    markup = reply_cue_html(state)
    assert state["emotion"]["label"] == "neutral"
    assert f"{label.capitalize()} cue" in markup and name in markup
    assert 'Vision estimate' in markup and f'data-reaction="{style}"' in markup


def test_explicit_words_override_conflicting_visual_cue_in_label_and_pose():
    markup = reply_cue_html(turn("I'm afraid.", "joy", "joy"))
    assert "Fear cue" in markup and "Your stated feeling" in markup
    assert 'data-reaction="careful"' in markup and "Joy cue" not in markup


def test_missing_camera_and_disagreement_are_not_claimed_as_single_visual_emotion():
    fallback = reply_cue_html(turn(visual=None, fused="joy"))
    assert "Text estimate · no usable camera cue" in fallback
    mixed = reply_cue_html(turn("Tell me about the harbor."))
    assert "Mixed cues" in mixed and "Clarifying" in mixed
    assert "Words + vision disagree" in mixed and "Fear cue" not in mixed


def test_initial_state_has_no_fabricated_reaction_and_ids_are_escaped():
    assert 'data-ready="false"' in reply_cue_html()
    state = turn()
    state["session_id"] = '\"><script>alert(1)</script>'
    assert '<script>' not in reply_cue_html(state)
    state["vision"]["available"] = False
    assert 'data-reaction="idle"' in reply_cue_html(state)
