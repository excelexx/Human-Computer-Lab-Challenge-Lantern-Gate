"""Local, streaming response generation through a loopback llama.cpp server."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx


LABELS = frozenset({"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"})
MAX_OUTPUT_TOKENS = 96
GENERATION_TEMPERATURE = 0.0
CONTEXT_RESERVE_TOKENS = 16
TOTAL_BUDGET_SECONDS = 90.0
READ_TIMEOUT_SECONDS = 10.0
MAX_SSE_LINE_BYTES = 65536
MAX_SSE_EVENT_BYTES = 65536
MAX_STREAM_BYTES = 262144
from .character import SYSTEM_PROMPT, character_context
from .quest import dialogue_goal, safe_context
from .scene import OPENING_LINE


class GeneratorError(RuntimeError):
    """The real local generator could not produce a usable response."""


def _label(value: Any) -> str | None:
    return value if isinstance(value, str) and value in LABELS else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _emotion_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    """Send selected classifier data, never unrestricted state as instructions."""
    emotion = _mapping(state.get("emotion"))
    modalities = _mapping(state.get("modalities"))
    vision = _mapping(state.get("vision"))
    source = emotion.get("source")
    scores: dict[str, float] = {}
    for label, score in _mapping(emotion.get("probabilities")).items():
        if label in LABELS and isinstance(score, (float, int)) and not isinstance(score, bool):
            if math.isfinite(score) and 0 <= score <= 1:
                scores[label] = round(float(score), 4)
    visual_available = vision.get("available") is True
    return {
        "predicted_emotion": _label(emotion.get("label")),
        "source": source if isinstance(source, str) and source in {"fusion", "text_fallback", "vision_only"} else "unknown",
        "score_semantics": "uncalibrated classifier scores, not certainty about feelings",
        "class_scores": scores,
        "text_emotion": _label(modalities.get("text_label")),
        "vision_emotion": _label(modalities.get("vision_label")) if visual_available else None,
        "vision_available": visual_available,
        "modality_disagreement": state.get("modality_disagreement") is True,
    }


def _turn_context(text: str, state: dict[str, Any]) -> dict[str, Any]:
    """Only bounded, whitelisted application data may steer delivery and goals."""
    evidence = _emotion_evidence(state)
    game = safe_context(state.get("game"))
    direction = character_context(evidence, text)
    goal = dialogue_goal(game, text)
    if direction["direction_source"] == "explicit_player_words":
        goal += f" The player's current explicit self-report is {direction['cue_emotion']}. Respond to this current disclosure; quoted or corrected earlier descriptions are not current feelings."
        goal += " Speak directly to the feeling they stated. You learned it from their words: do not narrate their appearance or paraphrase a facial expression as something you can see."
    # Turn-specific positive delivery tasks prevent the small local model from
    # giving the same factual answer under every camera tag. These are authored
    # purposes, not inserted NPC sentences or changes to the game state.
    delivery = {
        "practical": "Use a plain, friendly acknowledgment and a clear answer.",
        "playful": "Show warm enthusiasm for working together. Use upbeat, companionable wording; keep every practical statement literal and grounded in the known facts. No invented joke scenarios.",
        "careful": "Make the answer reassuring: give the player control of the pace and, if welcome, offer to stay beside them. Do not make a safety promise.",
        "patient": "Make the answer quietly patient: allow time and offer undemanding company if welcome, without pushing or cheering them up.",
        "steady": "Make the answer candid and steady: acknowledge the inconvenience and spare them any sales pitch. No teasing or pressure.",
        "curious": "Make the answer orienting: calmly put the immediate step into perspective, without inventing surprise or asking an extra question.",
        "wry": "Make the answer dryly companionable: use gentle understatement about the beacon chore. Never correct or mock the player's choice of words.",
    }
    if direction["direction_source"] != "modality_disagreement":
        goal += " Delivery for this turn: " + delivery[direction["response_style"]]
    goal += " Respond in at most two sentences. Start with your answer, not a quotation or repetition of the player's wording."
    return {"emotion_evidence": evidence, "npc_direction": direction,
            "game_context": game, "reply_goal": goal}


def _messages(text: str, state: dict[str, Any], history: list[Any]) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A nonempty check-in message is required.")
    if len(text) > 4000:
        raise ValueError("Keep the check-in message to 4,000 characters or fewer.")
    # This first-stage character bound keeps at most six prior messages. The
    # local template/tokenizer preflight in _fit_context verifies the token budget.
    previous: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, Mapping):
            role, content = item.get("role"), item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                previous.append({"role": role, "content": content})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            for role, content in zip(("user", "assistant"), item):
                if isinstance(content, str) and content.strip():
                    previous.append({"role": role, "content": content})
    bounded: list[dict[str, str]] = []
    remaining = 3000
    for item in reversed(previous[-6:]):
        if len(item["content"]) > remaining:
            break  # Omit an old message rather than silently rewriting its meaning.
        bounded.append(item)
        remaining -= len(item["content"])
    bounded.reverse()
    while bounded and bounded[0]["role"] != "user":
        bounded.pop(0)
    context = _turn_context(text, state)
    game = context["game_context"]
    # The visible opening is actual dialogue the player is replying to, even
    # though it is a UI placeholder rather than a completed generated turn.
    opening = [{"role": "assistant", "content": OPENING_LINE}] if game and game["completed"] == 0 and not bounded else []
    # Put the player's words first and the bounded, authored task last. With
    # the reverse order the local model echoed the line and missed the goal.
    turn = json.dumps({"message": text, **context}, ensure_ascii=False)
    system = SYSTEM_PROMPT
    if game:
        # Only recomputed, application-authored goals enter the system message;
        # never promote a player's text or a state-provided instruction here.
        from .quest import player_intent
        intent = player_intent(text)
        system += "\n\nYOUR TASK FOR THIS REPLY:\n" + context["reply_goal"]
        if intent["question"] or intent["pause"] or game["next_action"] == "walk_and_relight":
            system += "\nEnd with a statement. Ask no question in this reply."
        system += "\nKeep the answer to two sentences. Facts and consent outrank tone."
    return [{"role": "system", "content": system}, *opening, *bounded, {"role": "user", "content": turn}]


def _sse_data(lines: Iterator[str]) -> Iterator[str]:
    """Decode standard SSE events, including multiline data and keepalives."""
    data: list[str] = []
    event_bytes = 0
    for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
            event_bytes = 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            event_bytes += len(value.encode("utf-8")) + 1
            if event_bytes > MAX_SSE_EVENT_BYTES:
                raise GeneratorError("The local generator returned an oversized stream event.")
            data.append(value)
    if data:
        yield "\n".join(data)


def _check_deadline(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GeneratorError("The local response exceeded its time budget. Any partial response is preserved.")
    return remaining


def _request_timeout(deadline: float) -> httpx.Timeout:
    remaining = _check_deadline(deadline)
    return httpx.Timeout(connect=min(5.0, remaining), read=min(READ_TIMEOUT_SECONDS, remaining),
                         write=min(10.0, remaining), pool=min(5.0, remaining))


def _bounded_chunks(response: httpx.Response, deadline: float) -> Iterator[bytes]:
    """Bound preprocessing and streaming bodies at incoming chunk boundaries.

    No worker thread is created. A completely quiet socket may block until its
    read timeout; the total budget is observed at the next chunk/read boundary.
    """
    chunks = iter(response.iter_bytes())
    total = 0
    while True:
        _check_deadline(deadline)
        try:
            chunk = next(chunks)
        except StopIteration:
            break
        _check_deadline(deadline)
        total += len(chunk)
        if total > MAX_STREAM_BYTES:
            raise GeneratorError("The local generator exceeded the response stream size limit.")
        yield chunk


def _bounded_lines(response: httpx.Response, deadline: float) -> Iterator[str]:
    """Keep UTF-8 boundaries intact while limiting unfinished SSE lines."""
    pending = b""
    for chunk in _bounded_chunks(response, deadline):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if len(line) > MAX_SSE_LINE_BYTES:
                raise GeneratorError("The local generator returned an oversized stream line.")
            try:
                yield line.removesuffix(b"\r").decode("utf-8")
            except UnicodeDecodeError as error:
                raise GeneratorError("The local generator returned invalid UTF-8 stream data.") from error
            _check_deadline(deadline)
        if len(pending) > MAX_SSE_LINE_BYTES:
            raise GeneratorError("The local generator returned an oversized stream line.")
    if pending:
        try:
            yield pending.removesuffix(b"\r").decode("utf-8")
        except UnicodeDecodeError as error:
            raise GeneratorError("The local generator returned invalid UTF-8 stream data.") from error


def _context_request(client: httpx.Client, method: str, url: str, operation: str,
                     deadline: float, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    with client.stream(method, url, json=payload, timeout=_request_timeout(deadline)) as response:
        if response.status_code != 200:
            raise GeneratorError(f"Cannot verify the local model context: {operation} failed (HTTP {response.status_code}).")
        body = b"".join(_bounded_chunks(response, deadline))
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise GeneratorError(f"Cannot verify the local model context: invalid {operation} response.") from error
    if not isinstance(value, dict):
        raise GeneratorError(f"Cannot verify the local model context: invalid {operation} response.")
    return value


def _drop_oldest_history_group(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove one prior user turn and its following assistant text as a unit."""
    history = messages[1:-1]
    for index in range(1, len(history)):
        if history[index]["role"] == "user":
            return [messages[0], *history[index:], messages[-1]]
    return [messages[0], messages[-1]]


def _fit_context(client: httpx.Client, base_url: str, messages: list[dict[str, str]],
                 deadline: float) -> list[dict[str, str]]:
    """Count the server-rendered prompt, evicting whole old turns if necessary."""
    props = _context_request(client, "GET", base_url + "/props", "properties", deadline)
    _check_deadline(deadline)
    context = _mapping(props.get("default_generation_settings")).get("n_ctx")
    if isinstance(context, bool) or not isinstance(context, int) or context <= MAX_OUTPUT_TOKENS + CONTEXT_RESERVE_TOKENS:
        raise GeneratorError("Cannot verify the local model context: invalid context capacity.")
    # _messages retains at most six history messages; each iteration drops a
    # complete oldest group, so this loop is bounded by that existing limit.
    while True:
        formatted = _context_request(client, "POST", base_url + "/apply-template", "chat template", deadline, {"messages": messages})
        _check_deadline(deadline)
        prompt = formatted.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise GeneratorError("Cannot verify the local model context: missing formatted prompt.")
        tokenized = _context_request(client, "POST", base_url + "/tokenize", "tokenizer", deadline,
            {"content": prompt, "add_special": True, "parse_special": True, "with_pieces": False})
        _check_deadline(deadline)
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
            raise GeneratorError("Cannot verify the local model context: invalid tokenizer result.")
        if len(tokens) + MAX_OUTPUT_TOKENS + CONTEXT_RESERVE_TOKENS <= context:
            return messages
        if len(messages) <= 2 or messages[1:-1] == [{"role": "assistant", "content": OPENING_LINE}]:
            raise GeneratorError("Please shorten your message: it exceeds the local model's token budget even without conversation history.")
        messages = _drop_oldest_history_group(messages)


class LocalGenerator:
    """A synchronous token iterator for one local non-thinking Qwen generator."""

    def __init__(self, base_url: str = "http://127.0.0.1:8081") -> None:
        parts = urlsplit(base_url)
        if (
            parts.scheme != "http"
            or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            raise ValueError("Generation requires an HTTP loopback server URL without a path.")
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(connect=5.0, read=READ_TIMEOUT_SECONDS, write=10.0, pool=5.0)

    def status(self) -> dict[str, Any]:
        """Return health without raising or pretending an unavailable model works."""
        result: dict[str, Any] = {"available": False, "status": "unavailable", "base_url": self.base_url}
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=3.0) as client:
                response = client.get(f"{self.base_url}/health")
                result["http_status"] = response.status_code
                if response.status_code == 200:
                    body = response.json()
                    if isinstance(body, dict) and body.get("status") == "ok":
                        result.update(available=True, status="ready")
                elif response.status_code == 503:
                    result["status"] = "loading"
        except (httpx.HTTPError, ValueError):
            pass
        return result

    def stream(self, text: str, state: dict[str, Any], history: list[Any]) -> Iterator[str]:
        """Check game replies before display; never stream a rejected draft."""
        if not safe_context(state.get("game")):
            yield from self._stream_raw(text, state, history)
            return
        from .response_guard import problems, fallback
        rejected = []
        deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
        repair = []
        for attempt in range(2):
            chunks = list(self._stream_raw(text, state, history, repair=repair, deadline=deadline))
            candidate = "".join(chunks).strip()
            repair = problems(text, state, candidate)
            if not repair:
                state["response_guard"] = {"source": "local_model", "attempts": attempt + 1,
                    "rejected": rejected, "delivery": "validated_before_display"}
                yield from chunks
                return
            rejected.append({"text": candidate, "issues": repair})
        state["response_guard"] = {"source": "authored_fallback", "attempts": 2,
            "rejected": rejected, "delivery": "validated_before_display"}
        yield fallback(text, state)

    def _stream_raw(self, text: str, state: dict[str, Any], history: list[Any], *, repair=(), deadline=None) -> Iterator[str]:
        """Yield real text deltas; preserve partial output and raise on failure."""
        deadline = deadline if deadline is not None else time.monotonic() + TOTAL_BUDGET_SECONDS
        payload = {
            "model": "checkin-qwen",
            "messages": _messages(text, state, history),
            "stream": True,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": GENERATION_TEMPERATURE,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        }
        if repair:
            from .response_guard import REPAIR
            instructions = [REPAIR[issue] for issue in repair if issue in REPAIR]
            payload["messages"][0]["content"] += "\nREWRITE REQUIRED: " + " ".join(instructions)
            data = json.loads(payload["messages"][-1]["content"])
            data["rewrite_requirements"] = instructions
            payload["messages"][-1]["content"] = json.dumps(data, ensure_ascii=False)
        produced_text = False
        completed = False
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=self.timeout) as client:
                payload["messages"] = _fit_context(client, self.base_url, payload["messages"], deadline)
                with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload,
                                   timeout=_request_timeout(deadline)) as response:
                    if response.status_code != 200:
                        raise GeneratorError(f"Local generation failed (HTTP {response.status_code}).")
                    for data in _sse_data(_bounded_lines(response, deadline)):
                        if data.strip() == "[DONE]":
                            raise GeneratorError("The local generation stream ended without a completion reason.")
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise GeneratorError("The local generator returned invalid streaming data.") from error
                        if not isinstance(event, dict) or event.get("error"):
                            raise GeneratorError("The local generator returned a stream error.")
                        choices = event.get("choices", [])
                        if not isinstance(choices, list):
                            raise GeneratorError("The local generator returned malformed choices.")
                        if not choices:  # Optional usage-only event.
                            continue
                        if len(choices) != 1:
                            raise GeneratorError("The local generator returned an unexpected number of choices.")
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            raise GeneratorError("The local generator returned a malformed choice.")
                        raw_delta = choice.get("delta", {})
                        if not isinstance(raw_delta, Mapping):
                            raise GeneratorError("The local generator returned a malformed text delta.")
                        delta = raw_delta
                        content = delta.get("content")
                        if content is not None and not isinstance(content, str):
                            raise GeneratorError("The local generator returned a malformed text delta.")
                        if isinstance(content, str) and content:
                            produced_text = produced_text or bool(content.strip())
                            yield content
                        finish_reason = choice.get("finish_reason")
                        if finish_reason == "length":
                            raise GeneratorError("The local response reached its length limit. Any partial response is preserved; try a shorter check-in.")
                        if finish_reason == "stop":
                            completed = True
                            break
                        if finish_reason is not None:
                            raise GeneratorError("The local generator returned an unsupported completion reason.")
            if not completed:
                raise GeneratorError("The local generation stream ended before completion.")
            if not produced_text:
                raise GeneratorError("The local generator returned no response text.")
        except httpx.TimeoutException as error:
            raise GeneratorError("The local generator timed out. The emotion state is still available.") from error
        except httpx.HTTPError as error:
            raise GeneratorError("Cannot reach the local generator. Start the local model server.") from error
