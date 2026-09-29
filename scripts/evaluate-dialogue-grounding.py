"""Fixed synthetic emotion interventions through the real local Qwen server.

Compare saved runs; no camera or emotion-accuracy measurement is performed.
The nonstreaming transport permits a fixed seed with production sampling values.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

from checkin.character import SYSTEM_PROMPT
from checkin.generator import _messages, _fit_context
from checkin.quest import context, initial_quest, preview
from checkin.scene import SAMPLE_LINES


def state_for(text, visual, game):
    return {"emotion": {"label": "neutral", "source": "fusion" if visual else "text_fallback"},
            "vision": {"available": visual is not None},
            "modalities": {"text_label": "neutral", "vision_label": visual},
            "modality_disagreement": visual is not None and visual != "neutral",
            "game": context(game, text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Preserve prior results; choose a new output file.")
    record = {"created_at": datetime.now(timezone.utc).isoformat(),
              "protocol": "Same fixed cases, seeds and production sampling for each run. Synthetic evidence; real local Qwen. No webcam accuracy or general quality claim.",
              "system_prompt": SYSTEM_PROMPT,
              "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "cases": []}
    base = "http://127.0.0.1:8081"
    with httpx.Client(trust_env=False, timeout=90) as client:
        def run(case_id, text, visual, game=None, history=None, seed=314):
            game = initial_quest() if game is None else game
            history = [] if history is None else history
            state = state_for(text, visual, game)
            messages = _messages(text, state, history)
            item = {"id": case_id, "text": text, "visual": visual, "seed": seed,
                    "state": state, "messages": messages, "error": None}
            start = time.perf_counter()
            try:
                messages = _fit_context(client, base, messages, time.monotonic() + 90)
                result = client.post(base + "/v1/chat/completions", json={
                    "model": "checkin-qwen", "messages": messages, "stream": False,
                    "max_tokens": 96, "temperature": .5, "top_p": .8,
                    "top_k": 20, "min_p": 0., "seed": seed}).raise_for_status().json()
                choice = result["choices"][0]
                item.update(response=choice["message"]["content"], finish_reason=choice["finish_reason"])
            except Exception as error:
                item.update(error=str(error), response="")
            item["seconds"] = time.perf_counter() - start
            record["cases"].append(item)
            args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({key: item.get(key) for key in ("id", "response", "finish_reason", "error")}), flush=True)
            return item["response"]

        for visual in ("neutral", "joy", "fear", "anger", "sadness", "surprise", "disgust"):
            for seed in (314, 2718):
                run(f"opening-{visual}-{seed}", SAMPLE_LINES[0], visual, seed=seed)
        for index, line in enumerate(SAMPLE_LINES[1:], 1):
            for visual in ("joy", "fear", "anger"):
                run(f"sample-{index}-{visual}", line, visual)
        for index, text in enumerate(("What is the bridge like?", "I'm scared of heights. Let's take the sea stairs.",
                                      "Pause the game. What are you?", "No thanks, I'm staying here.")):
            run(f"custom-{index}", text, "joy")
        run("no-camera", "Oh, fantastic.", None)
        for route, visual in (("bridge", "joy"), ("stairs", "fear")):
            history, game = [], initial_quest()
            lines = ("Oh, fantastic.", "Fine. The bridge it is." if route == "bridge" else "The sea stairs, then.", "Lead the way.")
            for index, text in enumerate(lines):
                response = run(f"journey-{route}-{index}", text, visual, game, history)
                history += [{"role": "user", "content": text}, {"role": "assistant", "content": response}]
                game = preview(game, text)


if __name__ == "__main__":
    main()
