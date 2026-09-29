"""A local browser interface for short, multimodal daily check-ins.

Importing this module does not import Gradio, initialize a model, or contact a
remote service. ``build_app`` accepts an injected pipeline for UI smoke checks.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Iterator
import uuid

from .scene import PIXEL_CSS, SAMPLE_LINES
from . import quest
from .game_ui import game_html, game_css, MARA_PORTRAIT
from .reaction import reply_cue_html


CSS = """
:root, body, .dark {color-scheme:dark !important; background:#152B3C !important;}
body {margin:0 !important;}
html body .gradio-container.gradio-container {width:100% !important; min-width:0 !important;
  max-width:1260px !important; box-sizing:border-box !important; margin:auto;
  padding:24px clamp(14px,3.2vw,40px) 30px !important;
  background:#152B3C !important; color:#DFE9EF !important; font-family:'Segoe UI',Arial,sans-serif !important;}
.gradio-container main.fillable.app, .gradio-container main.app {padding:0 !important; width:100% !important;
  min-width:0 !important; max-width:100% !important; box-sizing:border-box !important;}
.gradio-container .contain, .gradio-container .wrap, .gradio-container .block,
.gradio-container .column, .gradio-container .row, .gradio-container .form {
  min-width:0 !important; box-sizing:border-box !important;}
.gradio-container video, .gradio-container canvas {max-width:100% !important;}
#masthead {display:flex; align-items:center; justify-content:space-between; gap:20px;
  padding-bottom:16px; border-bottom:1px solid #3E596C; color:#A8BDCB; font-size:12px;}
.brand-lockup {display:flex; align-items:center; gap:11px;}
.brand-lockup strong {font-size:21px; letter-spacing:-.7px; color:#DFE9EF; font-weight:600;}
.brand-mark {display:grid; place-items:center; width:36px; height:36px; border:1px solid #718999;
  border-radius:50%; background:#243F53;}
.privacy-note {display:flex; align-items:center; gap:8px;}
.privacy-note::before {content:''; height:6px; width:6px; background:#78C6C1; border-radius:50%;}
#invitation {display:flex; align-items:flex-end; justify-content:space-between; gap:36px;
  padding:21px 0 17px;}
#invitation h1 {font-family:Georgia,'Times New Roman',serif; font-size:clamp(36px,3.8vw,51px);
  line-height:1.12; letter-spacing:-1.8px; font-weight:400; color:#DFE9EF; margin:0;}
#invitation p {font-size:15px; line-height:1.75; color:#A8BDCB; max-width:35ch; margin:0 0 4px;}
#content-grid {gap:24px !important; align-items:stretch !important;}
#input-column {background:#1C3346; border:1px solid #3E596C; border-radius:18px;
  padding:23px; gap:15px;}
#conversation-column {background:#1C3346; border:1px solid #3E596C; border-radius:18px;
  padding:24px; gap:11px; box-shadow:0 10px 28px rgba(37,58,45,.035);}
.panel-heading h2 {font-size:18px; font-weight:600; letter-spacing:-.4px; color:#DFE9EF; margin:0 0 5px;}
.panel-heading p {font-size:13px; line-height:1.6; color:#A8BDCB; margin:0 0 5px;}
#input-column .block {box-shadow:none !important;}
#input-column .form {background:transparent !important; border:0 !important; box-shadow:none !important;}
#input-column .tab-nav {border-bottom:1px solid #3E596C; padding-bottom:3px;}
#input-column .tab-nav button {font-size:12px; color:#A8BDCB !important; padding:7px 12px;}
#input-column .tab-nav button.selected {color:#78C6C1 !important; border-bottom-color:#78C6C1 !important;}
#camera-clip {background:#203B50 !important; border:1px solid #3E596C !important; border-radius:12px;}
#camera-clip .wrap, #camera-clip .upload-container {background:#203B50 !important; color:#A8BDCB !important;}
#live-camera {background:#203B50 !important; border:1px solid #3E596C !important; border-radius:12px; overflow:hidden;}
#live-camera video {object-fit:contain; background:#294959;}
#live-camera-control {display:none !important;}
#live-camera-tag {padding:0 2px 3px; min-height:54px;}
#live-camera-tag .emotion-line {justify-content:space-between;}
#live-camera-tag .emotion-pill {background:#294959;}
#live-camera .button-wrap {border-radius:100px; padding:9px 14px;}
#capture-note p {font-size:12px !important; color:#A8BDCB !important; line-height:1.6 !important;}
#checkin-message {border:0 !important; background:transparent !important; padding:0 !important;}
#checkin-message .wrap, #checkin-message .container {background:transparent !important; box-shadow:none !important;}
#checkin-message textarea {background:#142B3C !important; color:#DFE9EF !important;
  border:1px solid #526F82 !important; border-radius:10px !important; line-height:1.65; padding:14px !important;}
#checkin-message textarea::placeholder {color:#A8BDCB !important;}
#conversation {border:0 !important; background:#1C3346 !important; border-radius:0;}
#conversation .message {font-size:15px; line-height:1.7; color:#DFE9EF !important;}
#conversation .user {background:#294959 !important; color:#DFE9EF !important; border:0 !important; border-radius:15px 15px 3px 15px;}
#conversation .bot {background:#243F53 !important; color:#DFE9EF !important; border:0 !important; border-radius:15px 15px 15px 3px;}
.conversation-empty {text-align:center; max-width:34ch; margin:auto; color:#A8BDCB;}
.conversation-empty .empty-shape {width:62px; height:70px; border:1px solid #526F82;
  border-radius:48% 48% 43% 43%; margin:0 auto 23px; background:#243F53; position:relative;}
.conversation-empty .empty-shape::after {content:''; position:absolute; width:24px; height:28px;
  border:1px solid #78C6C1; border-radius:65% 10% 65% 10%; top:20px; left:18px; transform:rotate(-12deg);}
.conversation-empty h3 {font-family:Georgia,'Times New Roman',serif; font-size:25px; font-weight:400;
  line-height:1.35; color:#E0BE75; margin:0 0 13px;}
.conversation-empty p {font-size:13px; line-height:1.8; margin:0; color:#A8BDCB;}
#send {background:#78C6C1 !important; color:#1C3346 !important; border:1px solid #78C6C1 !important;
  font-weight:600; min-height:47px; border-radius:10px; box-shadow:none !important; font-size:14px;}
#send:hover {background:#8EDAD4 !important;}
#send:disabled {opacity:.62;}
#stop {background:#1C3346 !important; color:#A8BDCB !important; border:1px solid #526F82 !important;
  border-radius:10px; min-height:47px; font-size:13px; box-shadow:none !important;}
#stop:disabled {color:#91A4B1 !important; border-color:#3E596C !important; opacity:.75;}
#new-conversation {background:transparent !important; color:#A8BDCB !important; border:0 !important;
  padding:6px 9px !important; box-shadow:none !important; font-size:12px !important; min-height:31px;}
#emotion-panel {border-top:1px solid #3E596C; padding:15px 0 2px; margin-top:2px;}
.emotion-line {display:flex; flex-wrap:wrap; gap:9px; align-items:center;}
.emotion-caption {font-size:12px; color:#A8BDCB;}
.emotion-pill {display:inline-block; border:1px solid #526F82; background:#294959;
  color:#A6DBD1; border-radius:100px; padding:4px 11px; font-size:12px; font-weight:500;}
.emotion-note {font-size:11px; line-height:1.5; color:#A8BDCB; margin-top:7px;}
#activity p {font-size:12px !important; color:#A8BDCB !important; line-height:1.5 !important;}
#activity {min-height:23px;}
#diagnostics {border:1px solid #3E596C; border-radius:10px; background:#152B3C !important; margin-top:12px;}
#diagnostics .label-wrap {color:#A8BDCB !important;}
#care-note p {font-size:11px !important; line-height:1.7 !important; color:#A8BDCB !important; max-width:100ch;}
#care-note {padding:5px 2px 0;}
.gradio-container button:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible {outline:3px solid #E0BE75 !important; outline-offset:3px;}
.gradio-container footer {display:none !important;}
@media (max-width:800px) {
  html body .gradio-container.gradio-container {width:100% !important; min-width:0 !important;
    max-width:100% !important; padding:18px !important; box-sizing:border-box !important;}
  .gradio-container main.fillable.app, .gradio-container main.app {padding:0 !important; margin:0 !important;}
  #masthead {font-size:11px; padding-bottom:17px;}
  #invitation {display:block; padding:25px 0 20px;}
  #invitation h1 {letter-spacing:-1px;}
  #invitation p {margin-top:15px; max-width:54ch; font-size:14px;}
  #input-column, #conversation-column {padding:18px; min-width:0 !important; width:100% !important;
    max-width:100% !important; flex:0 1 auto !important;}
  #content-grid {gap:18px !important; flex-direction:column !important; min-width:0 !important; width:100% !important;}
}
@media (max-width:420px) {.privacy-note {max-width:16ch; line-height:1.4;} .brand-lockup strong {font-size:19px;}}
@media (prefers-reduced-motion:reduce) {*, *::before, *::after {animation:none !important; transition:none !important;}}
"""


def resolve_home(home: str | Path | None = None) -> Path:
    """Resolve storage without depending on the shell's current directory."""
    configured = home or os.environ.get("CHECKIN_HOME")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2] / ".artifacts"


def clean_history(history: Any) -> list[dict[str, str]]:
    """Keep only plain user/assistant text from the UI's conversation state."""
    return [
        {"role": item["role"], "content": item["content"]}
        for item in history or []
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]


def video_path(value: Any) -> str | None:
    """Accept Gradio's filepath and defensive serialized video representations."""
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("video")
        return video_path(candidate)
    if isinstance(value, (list, tuple)) and value:
        return video_path(value[0])
    return None


def failure_message(error: Exception) -> str:
    """Expose useful recovery steps for known errors, keeping details in diagnostics."""
    detail = str(error)
    if detail.startswith("Please shorten your message:"):
        return "Your message is too long for the local model. Shorten it and retry; your draft and clip are kept."
    if detail.startswith("The local response reached its length limit."):
        return "The response reached its length limit. Any partial reply is shown above. Try a shorter check-in; your draft and clip are kept."
    return "This check-in could not finish. Your message and clip are kept so you can retry. See diagnostics for the error."


def emotion_html(state: dict[str, Any] | None = None) -> str:
    state = state or {}
    emotion = state.get("emotion") or state.get("final_emotion")
    if isinstance(emotion, dict):
        emotion = emotion.get("label") or emotion.get("category")
    label = html.escape(str(emotion).replace("_", " ").capitalize()) if emotion else "Awaiting your line"
    note = "A tentative interpretation, not a statement of how you feel." if emotion else "Your words and available camera signal will be considered together."
    if emotion and state.get("vision", {}).get("available") is False:
        note = "Based on your words; a usable face was not available. This interpretation can be wrong."
    direction = state.get("interaction", {})
    style = direction.get("response_style") if isinstance(direction, dict) else None
    if emotion and style:
        note += " Mara’s approach: " + html.escape(str(style)) + "."
        if direction.get("direction_source") == "ambiguous_demo_visual_cue":
            note += " The ambiguous line uses the visual cue for delivery."
    return (
        '<div class="emotion-line"><span class="emotion-caption">Player signal</span>'
        f'<span class="emotion-pill">{label}</span></div><div class="emotion-note">{note}</div>'
    )


def live_emotion_html(status: dict[str, Any] | None = None) -> str:
    """Keep live visual evidence distinct from a submitted text+vision check-in."""
    status = status or {"status": "off"}
    kind = status.get("status", "off")
    ready = kind == "ready" and bool(status.get("available")) and bool(status.get("label"))
    held = not ready and kind not in {"off", "discarded", "multiple_faces", "track_change"} and bool(status.get("held_label"))
    label = (str(status["label"]).capitalize() if ready else
             str(status["held_label"]).capitalize() + " · last" if held else
             "Camera off" if kind == "off" else "No reading yet")
    note = ("Tentative emotion from your camera." if ready else
            "Last observed emotion; not a current reading. Fresh frames update it automatically." if held else
            "Turn on your camera for an emotion tag." if kind == "off" else
            "Keep one face in view for the first emotion reading.")
    return ('<div class="emotion-line">'
            f'<span class="emotion-pill" title="{html.escape(note, quote=True)}" '
            f'data-held="{str(held).lower()}">{html.escape(label)}</span></div>')


def status_message(status: dict[str, Any]) -> str:
    if status.get("ready"):
        return "Ready on this computer. The first response may take longer while models load."
    return "Setup is incomplete. Open diagnostics below to see the missing files and next steps."


def load_replay_rows(home: Path) -> list[dict[str, Any]]:
    """Offer local test utterances when a prepared MELD manifest exists."""
    path = home / "manifests" / "meld.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if row.get("split") != "test" or not row.get("text") or not row.get("video_path"):
                continue
            clip = Path(row["video_path"])
            if not clip.is_absolute():
                clip = home / clip
            if not clip.is_file():
                continue
            rows.append({**row, "video_path": str(clip.resolve())})
            if len(rows) >= 100:
                break
    return rows


def replay_values(row: dict[str, Any]) -> tuple[str, str | None]:
    """Resolve one selected replay; kept separate for controlled scheduling probes."""
    return row["text"], row["video_path"]


def build_app(home: str | Path | None = None, pipeline: Any = None) -> Any:
    """Build the UI without downloading models or starting a web server."""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr
    from checkin.video_input import SilentVideo
    from checkin.live_vision import LiveVisionBuffer

    data_home = resolve_home(home)
    owns_pipeline = pipeline is None
    if pipeline is None:
        from checkin.pipeline import CheckInPipeline

        pipeline = CheckInPipeline(data_home)

    def get_status() -> dict[str, Any]:
        try:
            return pipeline.status()
        except Exception as exc:
            return {"ready": False, "errors": [f"{type(exc).__name__}: {exc}"], "components": {}}

    initial_status = get_status()
    if owns_pipeline and initial_status.get("classification_ready"):
        # Pay the small visual model's cold start before showing a ready app.
        # Idle webcam feedback never needs the large text encoder or generator.
        try:
            camera_startup = pipeline.prepare_live()
            initial_status = get_status()
            initial_status["live_camera_startup"] = camera_startup
        except Exception as exc:
            initial_status["ready"] = False
            initial_status.setdefault("errors", []).append(f"Camera startup: {exc}")
    replay_rows = load_replay_rows(data_home)
    theme = gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=["Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    )
    # Set both variants, since Gradio can inherit an OS/browser dark preference.
    # Keeping every surface on one intentional palette prevents mixed-theme text.
    palette = {
        "body_background_fill": "#152B3C", "body_text_color": "#DFE9EF",
        "body_text_color_subdued": "#A8BDCB", "background_fill_primary": "#1C3346",
        "background_fill_secondary": "#203B50", "border_color_primary": "#3E596C",
        "border_color_accent": "#718999", "border_color_accent_subdued": "#3E596C",
        "color_accent_soft": "#243F53", "link_text_color": "#78C6C1",
        "link_text_color_hover": "#DFE9EF", "link_text_color_active": "#78C6C1",
        "link_text_color_visited": "#78C6C1", "block_background_fill": "#1C3346",
        "block_border_color": "#3E596C", "block_info_text_color": "#A8BDCB",
        "block_label_background_fill": "#1C3346", "block_label_border_color": "#1C3346",
        "block_label_text_color": "#E0BE75", "block_title_background_fill": "#1C3346",
        "block_title_text_color": "#DFE9EF", "panel_background_fill": "#1C3346",
        "panel_border_color": "#3E596C", "accordion_text_color": "#A8BDCB",
        "input_background_fill": "#142B3C", "input_background_fill_focus": "#1C3346",
        "input_background_fill_hover": "#142B3C", "input_border_color": "#526F82",
        "input_border_color_focus": "#E0BE75", "input_border_color_hover": "#718999",
        "input_placeholder_color": "#A8BDCB", "button_primary_background_fill": "#78C6C1",
        "button_primary_background_fill_hover": "#8EDAD4", "button_primary_border_color": "#78C6C1",
        "button_primary_border_color_hover": "#8EDAD4", "button_primary_text_color": "#1C3346",
        "button_primary_text_color_hover": "#1C3346", "button_secondary_background_fill": "#1C3346",
        "button_secondary_background_fill_hover": "#294959", "button_secondary_border_color": "#526F82",
        "button_secondary_border_color_hover": "#718999", "button_secondary_text_color": "#A8BDCB",
        "button_secondary_text_color_hover": "#78C6C1", "code_background_fill": "#203B50",
        "loader_color": "#78C6C1", "table_text_color": "#DFE9EF", "table_border_color": "#3E596C",
        "table_even_background_fill": "#1C3346", "table_odd_background_fill": "#243F53",
        "error_background_fill": "#492D2D", "error_text_color": "#FFD3BF", "error_border_color": "#9D6551",
    }
    theme.set(**{key + suffix: value for key, value in palette.items() for suffix in ("", "_dark")},
              block_radius="12px", input_radius="10px", block_shadow="none", block_shadow_dark="none",
              button_primary_shadow="none", button_primary_shadow_dark="none", body_text_size="14px")

    with gr.Blocks(
        title="Lantern Gate | Speak with Mara",
        theme=theme,
        css=CSS + PIXEL_CSS + game_css(),
        js="() => {" + Path(__file__).with_name("game_world.js").read_text(encoding="utf-8") + "\n(" + Path(__file__).with_name("live_camera.js").read_text(encoding="utf-8") + ")();(" + Path(__file__).with_name("game.js").read_text(encoding="utf-8") + ")();}",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        session = gr.State(lambda: str(uuid.uuid4()))
        turn = gr.State(0)
        conversation_state = gr.State([])
        live_buffer = gr.State(LiveVisionBuffer())
        input_mode = gr.State("camera")
        # This input stays mounted for the browser's camera lifecycle signals.
        live_control = gr.Textbox(value="off:initial", elem_id="live-camera-control", show_label=False)
        live_refresh = gr.Timer(.1)
        # Gradio deep-copies the initial dictionary for each browser session.
        # Callbacks in that session share this object, including while queued.
        initial_replay_epoch = str(uuid.uuid4())
        replay_guard = gr.State({"owner": None, "epoch": initial_replay_epoch})
        replay_epoch = gr.Textbox(value=initial_replay_epoch, visible=False)
        replay_ticket = gr.JSON(visible=False)
        upload_ticket = gr.JSON(visible=False)
        load_replay = None
        gr.HTML(game_html())
        with gr.Column(elem_id="dialogue-panel"):
            gr.HTML('<div class="dialogue-title"><span>Mara, keeper of Lantern Gate</span><button type="button" data-game-help aria-label="How it works" aria-haspopup="dialog">?</button><button id="close-dialogue" type="button">Back to village [Esc]</button></div>', elem_id="dialogue-header")
            with gr.Row(equal_height=False, elem_id="content-grid"):
                with gr.Column(scale=3, min_width=170, elem_id="input-column"):
                    with gr.Tabs():
                        with gr.Tab("Camera") as live_tab:
                            live_camera = gr.Image(label="Your live camera", show_label=False, sources=["webcam"],
                                type="numpy", streaming=True, height=160, show_share_button=False,
                                show_download_button=False, show_fullscreen_button=False, elem_id="live-camera",
                                webcam_options=gr.WebcamOptions(mirror=False, constraints={"video": {
                                    "width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}}}))
                            live_emotion = gr.HTML(live_emotion_html(), elem_id="live-camera-tag")
                        with gr.Tab("Upload clip", visible=False) as upload_tab:
                            camera = SilentVideo(
                                label="A recorded clip",
                                sources=["upload"],
                                include_audio=False,
                                max_length=20,
                                height=180,
                                show_share_button=False,
                                elem_id="camera-clip",
                            )
                            gr.Markdown("An optional short video instead of the live camera. Each uploaded clip is used once. You can also continue with words alone.")
                        if replay_rows:
                            with gr.Tab("MELD replay", visible=False) as replay_tab:
                                replay_select = gr.Dropdown(
                                    choices=[
                                        (f"{row.get('dialogue_id', '?')}/{row.get('utterance_id', '?')}: {row['text'][:70]}", str(index))
                                        for index, row in enumerate(replay_rows)
                                    ],
                                    label="Recorded utterance",
                                )
                                load_replay = gr.Button("Use this utterance")
                                gr.Markdown("Loads a test utterance into the same check-in pipeline. Its reference label is not sent to the model.")
                with gr.Column(scale=7, min_width=260, elem_id="conversation-column"):
                    gr.HTML(MARA_PORTRAIT, elem_id="mara-portrait")
                    emotion = gr.HTML(reply_cue_html(), elem_id="reply-cue")
                    chat = gr.Chatbot(
                        label="Dialogue with Mara",
                        type="messages",
                        value=[],
                        height=180,
                        layout="bubble",
                        show_copy_button=False,
                        show_share_button=False,
                        show_label=False,
                        sanitize_html=True,
                        render_markdown=False,
                        autoscroll=False,
                        placeholder='<div class="conversation-empty"><p>“The beacon is out. The bridge is quick; the sea stairs are sheltered. What do you say, traveler?”</p></div>',
                        elem_id="conversation",
                    )
                    activity = gr.Markdown("Choose a reply to begin." if initial_status.get("ready") else status_message(initial_status), elem_id="activity")
            with gr.Column(elem_id="player-replies"):
                sample_note = gr.Markdown(quest.note(quest.initial_quest()), elem_id="example-note")
                sample_buttons = []
                for offset in range(0, len(SAMPLE_LINES), 2):
                    with gr.Row(elem_classes="sample-row"):
                        for line in SAMPLE_LINES[offset:offset + 2]:
                            sample_buttons.append(gr.Button(line, size="sm", elem_classes="sample-line", min_width=100))
                with gr.Row(elem_id="custom-reply-row"):
                    message = gr.Textbox(label="Your reply", show_label=False, placeholder="Or say something of your own…", lines=1,
                        max_lines=2, max_length=4000, elem_id="checkin-message", scale=6, min_width=120)
                    send = gr.Button("Send custom reply", variant="primary", elem_id="send", scale=1, min_width=75)
                    stop = gr.Button("Stop", elem_id="stop", scale=1, min_width=65, interactive=False)
            leave_dialogue = gr.Button("Leave dialogue", elem_id="leave-dialogue")
            new = gr.Button("New conversation", elem_id="new-conversation", size="sm")
            quest_event = gr.HTML(quest.signal(quest.initial_quest(), "initial"), elem_id="quest-event")
            with gr.Accordion("Diagnostics and setup", open=not bool(initial_status.get("ready")), elem_id="diagnostics", visible=False):
                gr.Markdown("The emotion state is produced by the classifier; the response generator uses that state with your message. Scores are not proof of a person's feelings.")
                output_state = gr.JSON(label="Latest structured state", value={})
                live_diagnostics = gr.JSON(label="Live camera state", value={"status": "off"}, elem_id="live-camera-diagnostics")
                setup = gr.JSON(label="Local component status", value=initial_status)
                refresh = gr.Button("Refresh local status", size="sm")
                gr.Markdown("Live camera frames are processed in memory on this computer. Only a short window of visual features is kept while the camera is on; turning it off clears that window. Uploaded clips use temporary files eligible for cleanup after one hour. New conversation clears the displayed history.")
                gr.Markdown("Audio is removed by copying the original video stream for H.264 MP4 and VP8/VP9 WebM. Other formats may require H.264 conversion, which changes pixels and displays a warning. Replay and uploaded clips use this same preparation.")

        compose_controls = [camera, message] + ([load_replay, replay_select] if load_replay is not None else [])
        outputs = [chat, conversation_state, emotion, output_state, activity, turn, send, stop] + compose_controls + [replay_epoch]
        game_outputs = sample_buttons + [sample_note, quest_event]
        base_output_count = len(outputs)
        outputs += game_outputs

        def game_updates(current, session_id, busy=False):
            enabled = not busy and current["phase"] != "depart"
            return tuple(gr.update(value=line, interactive=enabled) for line in quest.options(current)) + (quest.note(current), quest.signal(current, session_id))

        def compose_update(*, active: bool = False, clear: bool = False) -> tuple[Any, ...]:
            video_update = gr.update(interactive=not active, **({"value": None} if clear else {}))
            text_update = gr.update(interactive=not active, **({"value": ""} if clear else {}))
            replay_update = (gr.update(interactive=not active), gr.update(interactive=not active)) if load_replay is not None else ()
            return (video_update, text_update) + replay_update

        def invalidate_replay(guard: dict[str, Any] | None) -> bool:
            if guard is None:
                return False
            pending = guard.get("owner") is not None
            guard["owner"] = None
            guard.pop("upload_payload", None)
            guard["epoch"] = str(uuid.uuid4())
            return pending

        def epoch_value(guard: dict[str, Any] | None) -> tuple[Any]:
            return (guard["epoch"] if guard is not None else gr.skip(),)

        def run_turn_base(text: str, clip: Any, history: Any, session_id: str, turn_id: int, guard: dict[str, Any] | None = None,
                     buffer: Any = None, mode: str | None = None) -> Iterator[tuple[Any, ...]]:
            invalidate_replay(guard)
            prior = clean_history(history)
            next_turn = int(turn_id or 0)
            if not text or not text.strip():
                yield (prior, prior, gr.skip(), gr.skip(), "Write a line for Mara before sending.", next_turn, gr.update(interactive=True), gr.update(interactive=False)) + compose_update() + epoch_value(guard)
                return
            if len(text) > 4000:
                yield (prior, prior, gr.skip(), gr.skip(), "Please shorten your message to 4,000 characters or fewer.", next_turn, gr.update(interactive=True), gr.update(interactive=False)) + compose_update() + epoch_value(guard)
                return

            next_turn += 1
            messages = prior + [{"role": "user", "content": text.strip()}]
            state: dict[str, Any] = {}
            response = ""

            def cue_update():
                # Keep the old reply's cue until its replacement has actual text.
                if response:
                    return reply_cue_html(state)
                return gr.skip() if any(item["role"] == "assistant" for item in prior) else reply_cue_html()

            cancelled_before_state = False
            stream = None
            selected_clip = None if mode == "camera" else video_path(clip)
            turn_owner = str(uuid.uuid4())
            if guard is not None:
                guard["turn_owner"] = turn_owner
            try:
                yield (messages, messages, gr.skip(), state, "Mara is considering your words…", next_turn, gr.update(interactive=False), gr.update(interactive=True)) + compose_update(active=True) + epoch_value(guard)
                observation = buffer.snapshot(session_id) if buffer is not None and mode == "camera" else None
                live_kwargs = {"live_observation": observation} if observation is not None else {}
                if guard is not None and "quest" in guard:
                    live_kwargs["game_context"] = quest.context(guard["quest"], text)
                stream = pipeline.stream(text.strip(), selected_clip, prior, session_id, str(next_turn), **live_kwargs)
                for event in stream:
                    event_type = event.get("type")
                    if (("session_id" in event and str(event["session_id"]) != str(session_id))
                            or ("turn_id" in event and str(event["turn_id"]) != str(next_turn))):
                        raise RuntimeError("Received an event for a different check-in.")
                    if event_type in {"state", "done"}:
                        state = event.get("state") or state
                        if event_type == "done" and not response:
                            completed_response = state.get("response") or ""
                            response = str(completed_response.get("text") or "") if isinstance(completed_response, dict) else str(completed_response)
                    elif event_type == "text_delta":
                        response += str(event.get("text") or "")
                    elif event_type == "cancelled":
                        cancelled_before_state = True
                        # This is a cancellation record, not a fabricated emotion state.
                        state = {"cancelled": True, "session_id": event.get("session_id"),
                                 "turn_id": event.get("turn_id"), "phase": event.get("phase"),
                                 "reason": event.get("reason")}
                    elif event_type == "error":
                        raise RuntimeError(str(event.get("error") or "The local pipeline could not finish this turn."))
                    visible = messages + ([{"role": "assistant", "content": response}] if response else [])
                    progress = "Stopped before the emotion signal was ready." if cancelled_before_state else "Responding…" if response else "Emotion signal ready. Preparing a response…" if state else "Mara is considering your words…"
                    yield (visible, prior if cancelled_before_state else visible, cue_update(), state, progress, next_turn, gr.update(interactive=False), gr.update(interactive=True)) + tuple(gr.skip() for _ in compose_controls) + (gr.skip(),)

                visible = messages + ([{"role": "assistant", "content": response}] if response else [])
                cancelled = cancelled_before_state or (isinstance(state.get("response"), dict) and state["response"].get("status") == "cancelled")
                complete = "Stopped before the emotion signal was ready." if cancelled_before_state else "Stopped. Any partial response is shown above." if cancelled else "" if response else "The turn finished without response text. See diagnostics for details."
                yield (visible, prior if cancelled_before_state else visible, cue_update(), state, complete, next_turn, gr.update(interactive=True), gr.update(interactive=False)) + compose_update(clear=True) + (gr.skip(),)
            except Exception as exc:
                visible = messages + ([{"role": "assistant", "content": response}] if response else [])
                error_state = {**state, "error": f"{type(exc).__name__}: {exc}"}
                # The draft remains for retry, but a failed attempt must not also
                # enter model history and duplicate the next submitted message.
                yield (visible, prior, cue_update(), error_state, failure_message(exc), next_turn, gr.update(interactive=True), gr.update(interactive=False)) + compose_update() + (gr.skip(),)
            finally:
                if guard is not None and guard.get("turn_owner") == turn_owner:
                    guard.pop("turn_owner", None)
                if stream is not None and hasattr(stream, "close"):
                    stream.close()

        def run_turn(text, clip, history, session_id, turn_id, guard=None, buffer=None, mode=None):
            current = guard.setdefault("quest", quest.initial_quest()) if guard is not None else quest.initial_quest()
            if current["phase"] == "depart":
                yield tuple(gr.skip() for _ in range(base_output_count)) + game_updates(current, session_id)
                return
            candidate = quest.preview(current, text or "")
            stream = run_turn_base(text, clip, history, session_id, turn_id, guard, buffer, mode)
            own_token = None
            try:
                for result in stream:
                    if own_token is None and guard is not None:
                        own_token = guard.get("turn_owner")
                    enabled = result[6].get("interactive") is True
                    state = result[3]
                    succeeded = (enabled and isinstance(state, dict) and not state.get("error")
                        and isinstance(state.get("response"), dict)
                        and state["response"].get("status") == "complete"
                        and bool(state["response"].get("text"))
                        and (guard is None or own_token is not None and guard.get("turn_owner") == own_token))
                    if succeeded:
                        current = candidate
                        if guard is not None:
                            guard["quest"] = current
                    if current["phase"] == "depart":
                        result = list(result)
                        result[4] = "Mara is setting off. Escape pauses the journey."
                        result[6] = gr.update(interactive=False)
                        result[9] = gr.update(value="", interactive=False)
                    yield tuple(result) + game_updates(current, session_id, busy=not enabled)
            finally:
                stream.close()

        replay_events = []
        replay_outputs = [activity, send, stop] + compose_controls
        if load_replay is not None:

            def begin_replay(index: str | None, epoch: str, guard: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
                if epoch != guard.get("epoch"):
                    yield tuple(gr.skip() for _ in range(len(replay_outputs) + 1))
                    return
                if index is None:
                    yield (None, "Choose a recorded utterance first.") + tuple(gr.skip() for _ in replay_outputs[1:])
                    return
                ticket = {"id": str(uuid.uuid4()), "index": int(index), "epoch": epoch}
                guard["owner"] = ticket["id"]
                yield (ticket, "Loading the recorded check-in…", gr.update(interactive=False), gr.update(interactive=True)) + compose_update(active=True)

            def select_replay(ticket: dict[str, Any] | None, guard: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
                # The ticket is a non-State input captured when this request is
                # queued. A reset/edit cannot give an old request a new ticket.
                if not ticket or guard.get("owner") != ticket.get("id") or guard.get("epoch") != ticket.get("epoch"):
                    yield tuple(gr.skip() for _ in replay_outputs)
                    return
                try:
                    yield ("Loading the recorded check-in…",) + tuple(gr.skip() for _ in replay_outputs[1:])
                    text, clip = replay_values(replay_rows[ticket["index"]])
                    if guard.get("owner") != ticket["id"]:
                        yield tuple(gr.skip() for _ in replay_outputs)
                        return
                    guard["owner"] = None
                    yield ("Recorded check-in ready. You can edit it before sending.", gr.update(interactive=True), gr.update(interactive=False),
                           gr.update(value=clip, interactive=True), gr.update(value=text, interactive=True),
                           gr.update(interactive=True), gr.update(interactive=True))
                except Exception:
                    if guard.get("owner") == ticket["id"]:
                        guard["owner"] = None
                        yield ("The recorded check-in could not load. Your draft is kept; choose another utterance or try again.",
                               gr.update(interactive=True), gr.update(interactive=False)) + compose_update()
                # Cancelling the iterator alone does not release UI ownership:
                # the Stop/New/edit/Send callback invalidates it and restores
                # or takes over the controls. This also covers close-before-
                # callback ordering without leaving the composer disabled.

            admission = load_replay.click(begin_replay, [replay_select, replay_epoch, replay_guard], [replay_ticket] + replay_outputs,
                                          concurrency_limit=1, concurrency_id="replay-admission", trigger_mode="once", show_progress="hidden", api_name=False)
            # A cancelled admission can still produce a Gradio completion
            # notification. Only an actual ticket update may schedule loading.
            loading = replay_ticket.change(select_replay, [replay_ticket, replay_guard], replay_outputs,
                                           concurrency_limit=1, concurrency_id="gpu", show_progress="hidden", api_name=False)
            replay_events = [admission, loading]

        def admit_upload(payload: dict[str, Any] | None, epoch: str, guard: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
            # preprocess=False skips SilentVideo's FFmpeg work only. Gradio
            # still validates FileData and upload-folder access before admission.
            if epoch != guard.get("epoch") or payload is None:
                yield tuple(gr.skip() for _ in range(len(replay_outputs) + 1))
                return
            ticket = {"id": str(uuid.uuid4()), "epoch": epoch}
            guard["owner"] = ticket["id"]
            # Keep the validated path server-side. Client tickets never choose
            # a local path, and a newer upload immediately supersedes this one.
            guard["upload_payload"] = payload
            yield (ticket, "Preparing your clip…", gr.update(interactive=False), gr.update(interactive=True)) + compose_update(active=True)

        def normalize_upload(ticket: dict[str, Any] | None, guard: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
            if (not ticket or guard.get("owner") != ticket.get("id")
                    or guard.get("epoch") != ticket.get("epoch") or not guard.get("upload_payload")):
                yield tuple(gr.skip() for _ in replay_outputs)
                return
            payload = guard["upload_payload"]
            try:
                yield ("Preparing your clip…",) + tuple(gr.skip() for _ in replay_outputs[1:])
                clip = camera.preprocess(camera.data_model.model_validate(payload))
                if guard.get("owner") != ticket["id"] or guard.get("epoch") != ticket["epoch"]:
                    yield tuple(gr.skip() for _ in replay_outputs)
                    return
                guard["owner"] = None
                guard.pop("upload_payload", None)
                yield ("Clip ready. Share a few words before sending.", gr.update(interactive=True), gr.update(interactive=False),
                       gr.update(value=clip, interactive=True), gr.update(interactive=True)) + ((gr.update(interactive=True), gr.update(interactive=True)) if load_replay is not None else ())
            except Exception:
                if guard.get("owner") == ticket["id"]:
                    guard["owner"] = None
                    guard.pop("upload_payload", None)
                    yield ("The clip could not be prepared. Choose another short video; your message is kept.", gr.update(interactive=True), gr.update(interactive=False)) + compose_update()

        upload_admission = camera.upload(admit_upload, [camera, replay_epoch, replay_guard], [upload_ticket] + replay_outputs,
                                         preprocess=False, concurrency_id="upload-admission", concurrency_limit=1,
                                         trigger_mode="multiple", cancels=replay_events or None, show_progress="hidden", api_name=False)
        upload_loading = upload_ticket.change(normalize_upload, [upload_ticket, replay_guard], replay_outputs,
                                               concurrency_id="gpu", concurrency_limit=1, show_progress="hidden", api_name=False)
        replay_events += [upload_admission, upload_loading]

        def abandon_replay(guard: dict[str, Any]) -> tuple[Any, ...]:
            if not invalidate_replay(guard):
                return tuple(gr.skip() for _ in replay_outputs) + epoch_value(guard)
            return ("Clip loading stopped. Your draft is ready to edit.", gr.update(interactive=True), gr.update(interactive=False)) + compose_update() + epoch_value(guard)

        # User-only edits invalidate queued preparation. Programmatic output
        # values must not cancel themselves, so do not bind the change event.
        edit_events = [message.input, camera.clear]
        if load_replay is not None:
            edit_events.append(replay_select.input)
        for edit_event in edit_events:
            edit_event(abandon_replay, [replay_guard], replay_outputs + [replay_epoch], cancels=replay_events, queue=False, show_progress="hidden", api_name=False)

        inputs = [message, camera, conversation_state, session, turn, replay_guard, live_buffer, input_mode]
        send_event = send.click(run_turn, inputs, outputs, concurrency_limit=1, concurrency_id="gpu", trigger_mode="once", api_name=False,
                                cancels=replay_events or None, show_progress="full", show_progress_on=[chat])
        turn_events = [send_event, message.submit(run_turn, inputs, outputs, concurrency_limit=1,
            concurrency_id="gpu", trigger_mode="once", api_name=False, cancels=replay_events or None,
            show_progress="full", show_progress_on=[chat])]
        for sample_button in sample_buttons:
            def submit_sample(line, clip, history, session_id, turn_id, guard, buffer, mode):
                current = guard.setdefault("quest", quest.initial_quest())
                if line not in quest.options(current):
                    result = [gr.skip() for _ in range(base_output_count)]
                    result[4] = "Choose one of the new example replies, or write your own."
                    yield tuple(result) + game_updates(current, session_id)
                    return
                stream = run_turn(line, clip, history, session_id, turn_id, guard, buffer, mode)
                try:
                    for index, result in enumerate(stream):
                        if index == 0:
                            result = list(result)
                            result[9] = gr.update(value=line, interactive=False)
                        yield tuple(result)
                finally:
                    stream.close()
            turn_events.append(sample_button.click(submit_sample,
                [sample_button, camera, conversation_state, session, turn, replay_guard, live_buffer, input_mode], outputs,
                concurrency_limit=1, concurrency_id="gpu", trigger_mode="once", api_name=False,
                cancels=replay_events or None, show_progress="full", show_progress_on=[chat]))

        def request_cancel(session_id: str, turn_id: Any = None) -> bool:
            cancel = getattr(pipeline, "cancel", None)
            if not callable(cancel):
                return False
            return bool(cancel(session_id, str(turn_id))) if turn_id is not None else bool(cancel(session_id))

        def stop_current(session_id: str, turn_id: Any = None, guard: dict[str, Any] | None = None) -> tuple[Any, ...]:
            replay_pending = invalidate_replay(guard)
            if guard is not None:
                guard.pop("turn_owner", None)
            pending = request_cancel(session_id, turn_id)
            status = "Stopping after the current processing step. A new check-in may need to wait." if pending else "Replay loading stopped. Add a fresh clip or continue with words alone." if replay_pending else "No active response to stop."
            return (status, gr.update(interactive=True), gr.update(interactive=False)) + compose_update(clear=True) + epoch_value(guard) + game_updates(guard.get("quest", quest.initial_quest()) if guard else quest.initial_quest(), session_id)

        def new_conversation(session_id: str, turn_id: Any = None, guard: dict[str, Any] | None = None,
                             buffer: Any = None) -> tuple[Any, ...]:
            invalidate_replay(guard)
            next_epoch = str(uuid.uuid4())
            if guard is not None:
                guard["epoch"] = next_epoch
            pending = request_cancel(session_id, turn_id)
            if guard is not None:
                guard.pop("turn_owner", None)
                guard["quest"] = quest.initial_quest()
            new_session = str(uuid.uuid4())
            next_camera_control = gr.skip()
            if buffer is not None:
                buffer.reset(enabled=buffer.enabled, session_id=new_session)
                # Rotate the non-State token as well as the server session:
                # an old queued JPEG must not become this conversation's
                # first displayed estimate when State resolves at execution.
                next_camera_control = ("on:" if buffer.enabled else "off:") + str(uuid.uuid4())
                buffer.client_control = next_camera_control
            status = "A fresh conversation. The previous check-in is stopping after its current processing step." if pending else "A fresh conversation. The harbor gate awaits your next line."
            return ([], [], reply_cue_html(), {}, status, new_session, 0, gr.update(value="", interactive=True), gr.update(value=None, interactive=True), gr.update(interactive=True), gr.update(interactive=False)) + ((gr.update(interactive=True), gr.update(interactive=True)) if load_replay is not None else ()) + (next_camera_control, next_epoch) + game_updates(quest.initial_quest(), new_session)

        stop.click(
            stop_current,
            inputs=[session, turn, replay_guard],
            outputs=[activity, send, stop] + compose_controls + [replay_epoch] + game_outputs,
            cancels=turn_events + replay_events,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        leave_dialogue.click(stop_current, inputs=[session, turn, replay_guard],
            outputs=[activity, send, stop] + compose_controls + [replay_epoch] + game_outputs,
            cancels=turn_events + replay_events, queue=False, show_progress="hidden", api_name=False)
        new.click(
            new_conversation,
            inputs=[session, turn, replay_guard, live_buffer],
            outputs=[chat, conversation_state, emotion, output_state, activity, session, turn, message, camera, send, stop] + ([load_replay, replay_select] if load_replay is not None else []) + [live_control, replay_epoch] + game_outputs,
            cancels=turn_events + replay_events,
            queue=False,
            show_progress="hidden",
            api_name=False,
        )
        def refresh_status() -> tuple[dict[str, Any], str]:
            status = get_status()
            return status, status_message(status)

        refresh.click(refresh_status, outputs=[setup, activity], queue=False, show_progress="hidden", api_name=False)

        def camera_lifecycle(control: str, buffer: Any, session_id: str) -> None:
            # Repeated signals are harmless; a new track starts a fresh window.
            if getattr(buffer, "client_control", None) == control:
                return
            buffer.client_control = control
            buffer.reset(enabled=str(control).startswith("on:"), session_id=session_id)

        def observe_camera(frame: Any, buffer: Any, session_id: str, control: str) -> None:
            if not buffer.enabled or getattr(buffer, "client_control", None) != control:
                return
            observer = getattr(pipeline, "observe_live", None)
            if callable(observer):
                observer(frame, buffer, session_id)

        def live_status(buffer: Any, session_id: str) -> tuple[str, dict[str, Any]]:
            state = buffer.presentation_status(session_id)
            return live_emotion_html(state), state

        live_control.input(camera_lifecycle, [live_control, live_buffer, session], [], queue=False, api_name=False,
                           show_progress="hidden")
        # A webcam stream owns its queue slot for its lifetime. Keep it off
        # Send's queue; the pipeline's nonblocking GPU lock handles contention.
        live_camera.stream(observe_camera, [live_camera, live_buffer, session, live_control], [],
                           concurrency_id="live-camera", concurrency_limit=1,
                           stream_every=.2, time_limit=30, show_progress="hidden", api_name=False)
        live_refresh.tick(live_status, [live_buffer, session], [live_emotion, live_diagnostics],
                          queue=False, show_progress="hidden", api_name=False)
        live_tab.select(lambda: "camera", outputs=input_mode, queue=False, show_progress="hidden", api_name=False)
        upload_tab.select(lambda: "clip", outputs=input_mode, queue=False, show_progress="hidden", api_name=False)
        if replay_rows:
            replay_tab.select(lambda: "clip", outputs=input_mode, queue=False, show_progress="hidden", api_name=False)
    app.queue(default_concurrency_limit=1, max_size=8)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local check-in browser application.")
    parser.add_argument("--home", type=Path, help="Local model, checkpoint, and dataset directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    app = build_app(args.home)
    # Replay media can live outside the repository working directory. Grant
    # access to the exact offered clips, never the entire artifacts directory.
    replay_paths = [row["video_path"] for row in load_replay_rows(resolve_home(args.home))]
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=True,
        show_error=True,
        max_file_size="100mb",
        enable_monitoring=False,
        allowed_paths=replay_paths,
    )


if __name__ == "__main__":
    main()
