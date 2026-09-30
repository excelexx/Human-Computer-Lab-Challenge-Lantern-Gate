"""Opt-in local browser QA: synthetic camera tags, real production dialogue.

Run with --home PATH; browse localhost:7862. No camera is acquired. This tests
tag conditioning, not perception or learned fusion accuracy. Production launch
does not import this module. Stage resets install declared conversation fixtures.
"""
import argparse
from pathlib import Path

import gradio as gr

from checkin.app import build_app
from checkin.character import character_context
from checkin.generator import _emotion_evidence
from checkin.pipeline import CheckInPipeline
from checkin.settings import LABELS
from checkin import quest


class CameraQAPipeline(CheckInPipeline):
    def __init__(self, home):
        super().__init__(home, device="cpu")
        self.cues = {}

    def classify(self, text, video_path, session_id, turn_id, **kwargs):
        state = super().classify(text, None, session_id, turn_id, **kwargs)
        label = self.cues.get(session_id, "neutral")
        # Controlled classifier-output boundary. Real text inference is retained
        # for disagreement; fused scores are a fixture, not a measured prediction.
        state["emotion"].update(label=label, source="fusion",
            probabilities={name: .88 if name == label else .02 for name in LABELS})
        state["vision"].update(available=True, input_kind="simulated_camera_tag",
            reason="browser_qa_fixture", simulated=True)
        state["modalities"]["vision_label"] = label
        state["modality_disagreement"] = state["modalities"]["text_label"] != label
        state["input"]["video_present"] = False
        state["input"]["simulated_camera_emotion"] = label
        state["interaction"] = character_context(_emotion_evidence(state), text)
        return state


def build_qa(home):
    pipe = CameraQAPipeline(home)
    app = build_app(home, pipeline=pipe)
    functions = {fn.name: fn for fn in app.fns.values() if fn.fn is not None}
    reset = functions["new_conversation"]
    with app:
        with gr.Row(elem_id="camera-qa"):
            gr.Markdown("**SIMULATED CAMERA · QA ONLY** — no webcam images. Real local replies; stage resets use fixtures.")
            cue = gr.Radio(list(LABELS), value="neutral", label="Simulated camera emotion")
            stage = gr.Radio(["Opening", "Route choice", "Ready: bridge", "Ready: stairs"], value="Opening", label="Conversation fixture")
            apply = gr.Button("Reset QA conversation")
            status = gr.Markdown("QA ready", elem_id="qa-status")

        # Apply the selected label atomically with Reset; asynchronous radio
        # events must not race a reset and apply a tag to the previous session.

        def reset_qa(label, stage_name, *args):
            old_session = args[0]
            result = list(reset.fn(*args))
            pipe.cues.pop(old_session, None)
            new_session = result[5]
            pipe.cues[new_session] = label
            current = quest.initial_quest()
            history = []
            if stage_name != "Opening":
                current["completed"] = 1
                history = [{"role": "user", "content": "Oh, fantastic."},
                           {"role": "assistant", "content": "The beacon needs relighting. Would you prefer the signal bridge or the sea stairs?"}]
            if stage_name.startswith("Ready:"):
                current.update(completed=2, route="bridge" if stage_name.endswith("bridge") else "stairs")
                route = "signal bridge" if current["route"] == "bridge" else "sea stairs"
                history += [{"role": "user", "content": f"I prefer the {route}."},
                            {"role": "assistant", "content": f"The {route}, then. Are you ready to go?"}]
            args[2]["quest"] = current
            result[0] = result[1] = history
            result[-6:] = [gr.update(value=line, interactive=True) for line in quest.options(current)] + [quest.note(current), quest.signal(current, new_session)]
            return tuple(result) + (f"Simulated camera: {label} · {stage_name} fixture ready",)

        apply.click(reset_qa, [cue, stage] + reset.inputs, reset.outputs + [status],
                    queue=False, show_progress="hidden", api_name=False)

    app.css += """
    #camera-qa {position:fixed;bottom:0;left:0;right:0;z-index:100;padding:8px!important;
      background:#142b3c!important;border:2px solid #e0be75;max-height:180px;overflow:auto;}
    #camera-qa * {font-family:Arial,sans-serif!important;font-size:13px!important;color:#fff!important;}
    #camera-qa button {background:#355b65!important;min-width:130px;}
    #camera-qa label {background:#243f53!important;}
    #live-camera {pointer-events:none;}
    #live-camera::after {content:'SIMULATED CAMERA TAG — QA';position:absolute;top:20%;left:10%;color:#fff0c2;}
    """
    # QA-only stage resets keep Mara in view and hold departure for inspection.
    # This is declared test UI behavior; production game.js is not modified.
    app.js = app.js.replace("player=W.create()", "player={...W.create(),x:W.NPC.x,y:W.NPC.y+28}")
    app.js = app.js.replace("readingLeft=6", "readingLeft=3600")
    app.js = app.js.replace("s.count==='0' && questPhase!=='talking'", "questPhase!=='talking'")
    app.js = app.js.replace("Object.assign(player,W.create());", "Object.assign(player,W.create(),{x:W.NPC.x,y:W.NPC.y+28});")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    build_qa(args.home).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=False)
