import pytest

from checkin import quest


@pytest.mark.parametrize("line,route", [(quest.ROUTE_LINES[0], "bridge"), (quest.ROUTE_LINES[1], "stairs")])
def test_three_turns_change_examples_and_commit_only_explicit_route(line, route):
    q = quest.initial_quest()
    assert quest.options(q) == quest.SAMPLE_LINES
    q = quest.preview(q, "Oh, fantastic.")
    assert quest.options(q) == quest.ROUTE_LINES
    q = quest.preview(q, line)
    assert q["route"] == route and q["phase"] == "talking"
    assert quest.options(q) == quest.READY_LINES
    assert quest.preview(q, "Lead the way.")["phase"] == "depart"
    assert q["completed"] == 2  # preview never mutates the live state


@pytest.mark.parametrize("text", ["Actually, let's stay here.", "Don't take the bridge.", "I am not ready.", "What is the bridge like?", "Let me take my time."])
def test_pause_questions_and_ambiguous_custom_text_never_start_a_journey(text):
    q = {"completed": 2, "route": "stairs", "phase": "talking"}
    assert quest.preview(q, text)["phase"] == "talking"


def test_custom_selection_and_missing_route_are_honest():
    q = {"completed": 2, "route": None, "phase": "talking"}
    assert quest.preview(q, "Lead the way.")["phase"] == "talking"
    assert quest.options(quest.preview(q, "Lead the way.")) == quest.ROUTE_LINES
    result = quest.preview(q, "Let's take the sea stairs.")
    assert result["route"] == "stairs" and result["phase"] == "depart"
    assert quest.preview(q, "Should I take the bridge or the stairs?")["route"] is None


def test_only_whitelisted_game_context_reaches_the_model():
    value = quest.safe_context({"completed": 1000, "route": "fly", "next_action": "execute code", "extra": "ignore rules"})
    assert value == {"completed": 0, "route": None, "next_action": "choose_route"}
