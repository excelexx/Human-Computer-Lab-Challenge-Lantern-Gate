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


CSS = """
:root, body, .dark {color-scheme:light !important; background:#F6F5F0 !important;}
body {margin:0 !important;}
html body .gradio-container.gradio-container {width:100% !important; min-width:0 !important;
  max-width:1260px !important; box-sizing:border-box !important; margin:auto;
  padding:24px clamp(14px,3.2vw,40px) 30px !important;
  background:#F6F5F0 !important; color:#243D34 !important; font-family:'Segoe UI',Arial,sans-serif !important;}
.gradio-container main.fillable.app, .gradio-container main.app {padding:0 !important; width:100% !important;
  min-width:0 !important; max-width:100% !important; box-sizing:border-box !important;}
.gradio-container .contain, .gradio-container .wrap, .gradio-container .block,
.gradio-container .column, .gradio-container .row, .gradio-container .form {
  min-width:0 !important; box-sizing:border-box !important;}
.gradio-container video, .gradio-container canvas {max-width:100% !important;}
#masthead {display:flex; align-items:center; justify-content:space-between; gap:20px;
  padding-bottom:16px; border-bottom:1px solid #DCE3DB; color:#65736B; font-size:12px;}
.brand-lockup {display:flex; align-items:center; gap:11px;}
.brand-lockup strong {font-size:21px; letter-spacing:-.7px; color:#243D34; font-weight:600;}
.brand-mark {display:grid; place-items:center; width:36px; height:36px; border:1px solid #A9BBAA;
  border-radius:50%; background:#EDF1E7;}
.privacy-note {display:flex; align-items:center; gap:8px;}
.privacy-note::before {content:''; height:6px; width:6px; background:#64836A; border-radius:50%;}
#invitation {display:flex; align-items:flex-end; justify-content:space-between; gap:36px;
  padding:21px 0 17px;}
#invitation h1 {font-family:Georgia,'Times New Roman',serif; font-size:clamp(36px,3.8vw,51px);
  line-height:1.12; letter-spacing:-1.8px; font-weight:400; color:#243D34; margin:0;}
#invitation p {font-size:15px; line-height:1.75; color:#65736B; max-width:35ch; margin:0 0 4px;}
#content-grid {gap:24px !important; align-items:stretch !important;}
#input-column {background:#FFFFFF; border:1px solid #DCE3DB; border-radius:18px;
  padding:23px; gap:15px;}
#conversation-column {background:#FFFFFF; border:1px solid #DCE3DB; border-radius:18px;
  padding:24px; gap:11px; box-shadow:0 10px 28px rgba(37,58,45,.035);}
.panel-heading h2 {font-size:18px; font-weight:600; letter-spacing:-.4px; color:#243D34; margin:0 0 5px;}
.panel-heading p {font-size:13px; line-height:1.6; color:#65736B; margin:0 0 5px;}
#input-column .block {box-shadow:none !important;}
#input-column .form {background:transparent !important; border:0 !important; box-shadow:none !important;}
#input-column .tab-nav {border-bottom:1px solid #DCE3DB; padding-bottom:3px;}
#input-column .tab-nav button {font-size:12px; color:#65736B !important; padding:7px 12px;}
#input-column .tab-nav button.selected {color:#315A47 !important; border-bottom-color:#315A47 !important;}
#camera-clip {background:#F2F5EF !important; border:1px solid #DCE3DB !important; border-radius:12px;}
#camera-clip .wrap, #camera-clip .upload-container {background:#F2F5EF !important; color:#65736B !important;}
#capture-note p {font-size:12px !important; color:#65736B !important; line-height:1.6 !important;}
#checkin-message {border:0 !important; background:transparent !important; padding:0 !important;}
#checkin-message .wrap, #checkin-message .container {background:transparent !important; box-shadow:none !important;}
#checkin-message textarea {background:#FAFBF8 !important; color:#243D34 !important;
  border:1px solid #D6DFD3 !important; border-radius:10px !important; line-height:1.65; padding:14px !important;}
#checkin-message textarea::placeholder {color:#65736B !important;}
#conversation {border:0 !important; background:#FFFFFF !important; border-radius:0;}
#conversation .message {font-size:15px; line-height:1.7; color:#243D34 !important;}
#conversation .user {background:#EAF0E5 !important; color:#243D34 !important; border:0 !important; border-radius:15px 15px 3px 15px;}
#conversation .bot {background:#F6F7F3 !important; color:#243D34 !important; border:0 !important; border-radius:15px 15px 15px 3px;}
.conversation-empty {text-align:center; max-width:34ch; margin:auto; color:#65736B;}
.conversation-empty .empty-shape {width:62px; height:70px; border:1px solid #C0CDBA;
  border-radius:48% 48% 43% 43%; margin:0 auto 23px; background:#EDF2E7; position:relative;}
.conversation-empty .empty-shape::after {content:''; position:absolute; width:24px; height:28px;
  border:1px solid #8FA38B; border-radius:65% 10% 65% 10%; top:20px; left:18px; transform:rotate(-12deg);}
.conversation-empty h3 {font-family:Georgia,'Times New Roman',serif; font-size:25px; font-weight:400;
  line-height:1.35; color:#3F5D4D; margin:0 0 13px;}
.conversation-empty p {font-size:13px; line-height:1.8; margin:0; color:#65736B;}
#send {background:#315A47 !important; color:#FFFFFF !important; border:1px solid #315A47 !important;
  font-weight:600; min-height:47px; border-radius:10px; box-shadow:none !important; font-size:14px;}
#send:hover {background:#264D3B !important;}
#send:disabled {opacity:.62;}
#stop {background:#FFFFFF !important; color:#5E7164 !important; border:1px solid #D1DCCF !important;
  border-radius:10px; min-height:47px; font-size:13px; box-shadow:none !important;}
#stop:disabled {color:#939D95 !important; border-color:#E1E7DF !important; opacity:.75;}
#new-conversation {background:transparent !important; color:#566F5E !important; border:0 !important;
  padding:6px 9px !important; box-shadow:none !important; font-size:12px !important; min-height:31px;}
#emotion-panel {border-top:1px solid #E4E9E0; padding:15px 0 2px; margin-top:2px;}
.emotion-line {display:flex; flex-wrap:wrap; gap:9px; align-items:center;}
.emotion-caption {font-size:12px; color:#6B776F;}
.emotion-pill {display:inline-block; border:1px solid #D8E1D2; background:#F1F5EC;
  color:#527047; border-radius:100px; padding:4px 11px; font-size:12px; font-weight:500;}
.emotion-note {font-size:11px; line-height:1.5; color:#65736B; margin-top:7px;}
#activity p {font-size:12px !important; color:#65736B !important; line-height:1.5 !important;}
#activity {min-height:23px;}
#diagnostics {border:1px solid #DCE3DB; border-radius:10px; background:#F6F5F0 !important; margin-top:12px;}
#diagnostics .label-wrap {color:#65736B !important;}
#care-note p {font-size:11px !important; line-height:1.7 !important; color:#65736B !important; max-width:100ch;}
#care-note {padding:5px 2px 0;}
.gradio-container button:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible {outline:3px solid #8CA286 !important; outline-offset:3px;}
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


def emotion_html(state: dict[str, Any] | None = None) -> str:
    state = state or {}
    emotion = state.get("emotion") or state.get("final_emotion")
    if isinstance(emotion, dict):
        emotion = emotion.get("label") or emotion.get("category")
    label = html.escape(str(emotion).replace("_", " ").capitalize()) if emotion else "Awaiting a check-in"
    note = "A tentative interpretation, not a statement of how you feel." if emotion else "Your words and clip will be considered together."
    if emotion and state.get("vision", {}).get("available") is False:
        note = "Based on your words; a usable face was not available. This interpretation can be wrong."
    return (
        '<div class="emotion-line"><span class="emotion-caption">Emotion signal</span>'
        f'<span class="emotion-pill">{label}</span></div><div class="emotion-note">{note}</div>'
    )


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


def build_app(home: str | Path | None = None, pipeline: Any = None) -> Any:
    """Build the UI without downloading models or starting a web server."""
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    data_home = resolve_home(home)
    if pipeline is None:
        from checkin.pipeline import CheckInPipeline

        pipeline = CheckInPipeline(data_home)

    def get_status() -> dict[str, Any]:
        try:
            return pipeline.status()
        except Exception as exc:
            return {"ready": False, "errors": [f"{type(exc).__name__}: {exc}"], "components": {}}

    initial_status = get_status()
    replay_rows = load_replay_rows(data_home)
    theme = gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
        font=["Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    )
    # Set both variants, since Gradio can inherit an OS/browser dark preference.
    # Keeping every surface on one intentional palette prevents mixed-theme text.
    palette = {
        "body_background_fill": "#F6F5F0", "body_text_color": "#243D34",
        "body_text_color_subdued": "#65736B", "background_fill_primary": "#FFFFFF",
        "background_fill_secondary": "#F2F5EF", "border_color_primary": "#DCE3DB",
        "border_color_accent": "#A9BBAA", "border_color_accent_subdued": "#DCE3DB",
        "color_accent_soft": "#EDF2E7", "link_text_color": "#315A47",
        "link_text_color_hover": "#243D34", "link_text_color_active": "#315A47",
        "link_text_color_visited": "#315A47", "block_background_fill": "#FFFFFF",
        "block_border_color": "#DCE3DB", "block_info_text_color": "#65736B",
        "block_label_background_fill": "#FFFFFF", "block_label_border_color": "#FFFFFF",
        "block_label_text_color": "#3F5D4D", "block_title_background_fill": "#FFFFFF",
        "block_title_text_color": "#243D34", "panel_background_fill": "#FFFFFF",
        "panel_border_color": "#DCE3DB", "accordion_text_color": "#65736B",
        "input_background_fill": "#FAFBF8", "input_background_fill_focus": "#FFFFFF",
        "input_background_fill_hover": "#FAFBF8", "input_border_color": "#D6DFD3",
        "input_border_color_focus": "#8CA286", "input_border_color_hover": "#A9BBAA",
        "input_placeholder_color": "#65736B", "button_primary_background_fill": "#315A47",
        "button_primary_background_fill_hover": "#264D3B", "button_primary_border_color": "#315A47",
        "button_primary_border_color_hover": "#264D3B", "button_primary_text_color": "#FFFFFF",
        "button_primary_text_color_hover": "#FFFFFF", "button_secondary_background_fill": "#FFFFFF",
        "button_secondary_background_fill_hover": "#F1F5EC", "button_secondary_border_color": "#D1DCCF",
        "button_secondary_border_color_hover": "#A9BBAA", "button_secondary_text_color": "#566F5E",
        "button_secondary_text_color_hover": "#315A47", "code_background_fill": "#F2F5EF",
        "loader_color": "#537263", "table_text_color": "#243D34", "table_border_color": "#DCE3DB",
        "table_even_background_fill": "#FFFFFF", "table_odd_background_fill": "#F6F8F3",
        "error_background_fill": "#FFF5F1", "error_text_color": "#864B38", "error_border_color": "#E3C9BE",
    }
    theme.set(**{key + suffix: value for key, value in palette.items() for suffix in ("", "_dark")},
              block_radius="12px", input_radius="10px", block_shadow="none", block_shadow_dark="none",
              button_primary_shadow="none", button_primary_shadow_dark="none", body_text_size="14px")

    with gr.Blocks(
        title="Check-in | A moment for your day",
        theme=theme,
        css=CSS,
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        session = gr.State(lambda: str(uuid.uuid4()))
        turn = gr.State(0)
        conversation_state = gr.State([])
        gr.HTML('<div id="masthead"><div class="brand-lockup"><span class="brand-mark" aria-hidden="true"><svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M6 18C6 10 10 5 18 5C18 13 14 18 6 18Z" stroke="#537263" stroke-width="1.35"/><path d="M6 18L14 10" stroke="#537263" stroke-width="1.35" stroke-linecap="round"/></svg></span><strong>Check-in</strong></div><span class="privacy-note">Private, on your computer</span></div>')
        gr.HTML('<section id="invitation"><h1>A little space<br>for your day.</h1><p>Start wherever you are. Share a few words and a short clip, and take a moment to reflect.</p></section>')

        with gr.Row(equal_height=False, elem_id="content-grid"):
            with gr.Column(scale=4, min_width=300, elem_id="input-column"):
                gr.HTML('<div class="panel-heading"><h2>Your check-in</h2><p>How did today feel?</p></div>')
                with gr.Tabs():
                    with gr.Tab("Camera"):
                        camera = gr.Video(
                            label="Your camera clip",
                            sources=["webcam", "upload"],
                            include_audio=False,
                            max_length=20,
                            height=180,
                            show_share_button=False,
                            elem_id="camera-clip",
                        )
                        gr.Markdown("A 3–5 second clip, with your face in view. No audio. You can also continue with words alone.", elem_id="capture-note")
                    if replay_rows:
                        with gr.Tab("MELD replay"):
                            replay_select = gr.Dropdown(
                                choices=[
                                    (f"{row.get('dialogue_id', '?')}/{row.get('utterance_id', '?')}: {row['text'][:70]}", str(index))
                                    for index, row in enumerate(replay_rows)
                                ],
                                label="Recorded utterance",
                            )
                            load_replay = gr.Button("Use this utterance")
                            gr.Markdown("Loads a test utterance into the same check-in pipeline. Its reference label is not sent to the model.")
                message = gr.Textbox(
                    label="What happened today?",
                    placeholder="There was a moment today that stayed with me…",
                    lines=3,
                    max_lines=8,
                    max_length=4000,
                    elem_id="checkin-message",
                )
                with gr.Row():
                    send = gr.Button("Send check-in", variant="primary", elem_id="send", scale=3)
                    stop = gr.Button("Stop", elem_id="stop", scale=1, interactive=False)
            with gr.Column(scale=6, min_width=320, elem_id="conversation-column"):
                gr.HTML('<div class="panel-heading"><h2>Your conversation</h2><p>A moment to feel heard.</p></div>')
                chat = gr.Chatbot(
                    label="Our conversation",
                    type="messages",
                    value=[],
                    height=356,
                    layout="bubble",
                    show_copy_button=True,
                    show_share_button=False,
                    show_label=False,
                    sanitize_html=True,
                    render_markdown=False,
                    placeholder='<div class="conversation-empty"><div class="empty-shape" aria-hidden="true"></div><h3>Start with one moment.</h3><p>Something small, something difficult,<br>or something worth celebrating.</p></div>',
                    elem_id="conversation",
                )
                emotion = gr.HTML(emotion_html(), elem_id="emotion-panel")
                activity = gr.Markdown(status_message(initial_status), elem_id="activity")
                new = gr.Button("New conversation", elem_id="new-conversation", size="sm")

        gr.Markdown("A supportive reflection companion. Its emotion signals can be wrong; you decide what fits your experience. This prototype does not provide diagnosis or treatment.", elem_id="care-note")
        with gr.Accordion("Diagnostics and setup", open=not bool(initial_status.get("ready")), elem_id="diagnostics"):
            gr.Markdown("The emotion state is produced by the classifier; the response generator uses that state with your message. Scores are not proof of a person's feelings.")
            output_state = gr.JSON(label="Latest structured state", value={})
            setup = gr.JSON(label="Local component status", value=initial_status)
            refresh = gr.Button("Refresh local status", size="sm")
            gr.Markdown("Camera clips are processed on this computer. Temporary browser uploads are eligible for cleanup after one hour. New conversation clears the displayed history; it does not immediately erase temporary files.")

        outputs = [chat, conversation_state, emotion, output_state, activity, turn, send, stop]

        def run_turn(text: str, clip: Any, history: Any, session_id: str, turn_id: int) -> Iterator[tuple[Any, ...]]:
            prior = clean_history(history)
            next_turn = int(turn_id or 0)
            if not text or not text.strip():
                yield prior, prior, gr.skip(), gr.skip(), "Write a message about your day before sending.", next_turn, gr.update(interactive=True), gr.update(interactive=False)
                return
            if len(text) > 4000:
                yield prior, prior, gr.skip(), gr.skip(), "Please shorten your message to 4,000 characters or fewer.", next_turn, gr.update(interactive=True), gr.update(interactive=False)
                return

            next_turn += 1
            messages = prior + [{"role": "user", "content": text.strip()}]
            state: dict[str, Any] = {}
            response = ""
            stream = None
            yield messages, messages, emotion_html(), state, "Considering your words and clip…", next_turn, gr.update(interactive=False), gr.update(interactive=True)
            try:
                stream = pipeline.stream(text.strip(), video_path(clip), prior, session_id, str(next_turn))
                for event in stream:
                    event_type = event.get("type")
                    if event_type in {"state", "done"}:
                        state = event.get("state") or state
                        if event_type == "done" and not response:
                            completed_response = state.get("response") or ""
                            response = str(completed_response.get("text") or "") if isinstance(completed_response, dict) else str(completed_response)
                    elif event_type == "text_delta":
                        response += str(event.get("text") or "")
                    elif event_type == "error":
                        raise RuntimeError(str(event.get("error") or "The local pipeline could not finish this turn."))
                    visible = messages + ([{"role": "assistant", "content": response}] if response else [])
                    progress = "Responding…" if response else "Emotion signal ready. Preparing a response…" if state else "Considering your check-in…"
                    yield visible, visible, emotion_html(state), state, progress, next_turn, gr.update(interactive=False), gr.update(interactive=True)

                visible = messages + ([{"role": "assistant", "content": response}] if response else [])
                cancelled = isinstance(state.get("response"), dict) and state["response"].get("status") == "cancelled"
                complete = "Stopped. Any partial response is shown above." if cancelled else "Ready for your next check-in." if response else "The turn finished without response text. See diagnostics for details."
                yield visible, visible, emotion_html(state), state, complete, next_turn, gr.update(interactive=True), gr.update(interactive=False)
            except Exception as exc:
                visible = messages + ([{"role": "assistant", "content": response}] if response else [])
                error_state = {**state, "error": f"{type(exc).__name__}: {exc}"}
                yield visible, visible, emotion_html(state), error_state, "This check-in could not finish. Open diagnostics for the error, then retry.", next_turn, gr.update(interactive=True), gr.update(interactive=False)
            finally:
                if stream is not None and hasattr(stream, "close"):
                    stream.close()

        inputs = [message, camera, conversation_state, session, turn]
        send_event = send.click(run_turn, inputs, outputs, concurrency_limit=1, concurrency_id="gpu", trigger_mode="once", api_name=False)

        def request_cancel(session_id: str) -> bool:
            cancel = getattr(pipeline, "cancel", None)
            return bool(cancel(session_id)) if callable(cancel) else False

        def stop_current(session_id: str) -> tuple[Any, ...]:
            pending = request_cancel(session_id)
            status = "Stopping after the current processing step. A new check-in may need to wait." if pending else "No active response to stop."
            return status, gr.update(interactive=True), gr.update(interactive=False)

        def new_conversation(session_id: str) -> tuple[Any, ...]:
            pending = request_cancel(session_id)
            status = "A fresh conversation. The previous check-in is stopping after its current processing step." if pending else "A fresh conversation. Share something from your day."
            return [], [], emotion_html(), {}, status, str(uuid.uuid4()), 0, "", None, gr.update(interactive=True), gr.update(interactive=False)

        stop.click(
            stop_current,
            inputs=[session],
            outputs=[activity, send, stop],
            cancels=[send_event],
            queue=False,
            api_name=False,
        )
        new.click(
            new_conversation,
            inputs=[session],
            outputs=[chat, conversation_state, emotion, output_state, activity, session, turn, message, camera, send, stop],
            cancels=[send_event],
            queue=False,
            api_name=False,
        )
        def refresh_status() -> tuple[dict[str, Any], str]:
            status = get_status()
            return status, status_message(status)

        refresh.click(refresh_status, outputs=[setup, activity], queue=False, api_name=False)
        if replay_rows:
            def select_replay(index: str | None) -> tuple[Any, Any]:
                if index is None:
                    return gr.skip(), gr.skip()
                row = replay_rows[int(index)]
                return row["text"], row["video_path"]

            load_replay.click(select_replay, [replay_select], [message, camera], queue=False, api_name=False)

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
