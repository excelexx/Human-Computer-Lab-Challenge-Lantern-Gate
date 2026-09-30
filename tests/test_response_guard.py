import pytest

from checkin.generator import LocalGenerator
from checkin.response_guard import problems, fallback


def game(action="choose_route", route=None):
    return {"game": {"completed": 2, "route": route, "next_action": action},
            "interaction": {"response_style": "careful"}}


@pytest.mark.parametrize("text,response,issue", [
    ("Before we leave, which route did I pick?", "You picked the sea stairs. Ready to go?", "question"),
    ("Can you tell what I am wearing?", "I cannot see what you are wearing. The beacon remains out.", "perception"),
    ("I'm calm. Can we pause?", "I didn't see any fear. Let's go to the sea stairs.", "pause"),
    ("Oh, fantastic.", "Oh, fantastic. That's a strange thing to say.", "judgment"),
    ("What could possibly go wrong?", "A gale or a slip.", "grounding"),
    ("Was the bridge inspected?", "No inspection records exist for the bridge.", "inspection"),
    ("I'm smiling, but worried.", "I see you're smiling, but worried.", "perception"),
    ("The sea stairs, then.", "Sea stairs it is, damp but safe and sheltered.", "grounding"),
    ("Lead the way.", "I'll lead the way, steady and safe.", "grounding"),
    ("What could possibly go wrong?", "Let's make it safe together.", "grounding"),
    ("I'm calm. Can we pause?", "I see you're calm. We can pause here.", "perception"),
    ("Are you reacting to my face?", "I only receive words and never use vision.", "perception"),
    ("I'm not disgusted; I'm concentrating.", "Bridge or stairs?", "question"),
    ("I'm not disgusted; I'm concentrating.", "I hear you're focused. The beacon still needs relighting.", "correction"),
])
def test_observed_browser_failures_are_rejected(text, response, issue):
    assert issue in problems(text, game(), response)


def test_authorized_departure_keeps_the_selected_route():
    state = game("walk_and_relight", "stairs")
    assert "route" in problems("Lead the way.", state, "I'll lead you across the bridge now.")
    assert not problems("Lead the way.", state, "I'll lead you along the sea stairs now. We can take it at your pace.")


@pytest.mark.parametrize("text", [
    "What could possibly go wrong?", "Can you tell what I am wearing?",
    "Do you know whether the bridge was inspected?", "Go without me; I am staying here.",
    "Before we leave, which route did I pick?", "Can you explain why the beacon matters?",
])
def test_authored_fallback_is_bounded_and_grounded(text):
    state = game("confirm_departure", "stairs")
    assert not problems(text, state, fallback(text, state))


def test_invalid_draft_is_not_shown_and_repair_uses_same_deadline(monkeypatch):
    generator = LocalGenerator()
    calls = []
    def raw(text, state, history, **kwargs):
        calls.append(kwargs)
        yield "Ready to go?" if len(calls) == 1 else "We can pause here."
    monkeypatch.setattr(generator, "_stream_raw", raw)
    state = game()
    assert list(generator.stream("Can we pause?", state, [])) == ["We can pause here."]
    assert calls[1]["repair"] == ["question", "pause"]
    assert calls[0]["deadline"] == calls[1]["deadline"]
    assert state["response_guard"]["source"] == "local_model"
    assert state["response_guard"]["attempts"] == 2


def test_two_failures_use_a_marked_fallback_and_never_keep_retrying(monkeypatch):
    generator = LocalGenerator()
    monkeypatch.setattr(generator, "_stream_raw", lambda *a, **k: iter(["Ready to go?"]))
    state = game()
    assert "pause here" in "".join(generator.stream("Can we pause?", state, []))
    assert state["response_guard"]["source"] == "authored_fallback"
    assert len(state["response_guard"]["rejected"]) == 2
