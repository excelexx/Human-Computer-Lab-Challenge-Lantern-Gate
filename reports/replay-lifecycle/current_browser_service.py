"""Bounded loopback-only synthetic scheduling probe; never loads ML models.

Run with the project venv and PYTHONPATH=outputs/checkin/src. This uses the
production build_app and event configuration. Only replay rows/value resolution
and the pipeline are fixtures. Select the single replay, click Use, then New
or Stop during the 2.5-second loading window. No service is started on import.
"""
import argparse
import time

import gradio as gr
import checkin.app as module


class FixturePipeline:
    def status(self):
        return {"ready": True, "errors": [], "components": {"fixture": "SYNTHETIC CPU SCHEDULING PROBE; no inference"}}

    def cancel(self, session_id, turn_id=None):
        return False

    def stream(self, *args):
        yield {"type": "error", "error": "Synthetic scheduling probe: inference is intentionally unavailable."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--delay", type=float, default=2.5)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800 or not .1 <= args.delay <= 10:
        parser.error("Use a bounded lifetime of 1–1800 seconds and delay of 0.1–10 seconds.")
    module.load_replay_rows = lambda home: [{"text": "OLD REPLAY", "video_path": None, "dialogue_id": "synthetic", "utterance_id": 0}]
    def delayed(row):
        print("SYNTHETIC replay preparation entered", flush=True)
        time.sleep(args.delay)
        print("SYNTHETIC replay preparation finished", flush=True)
        return row["text"], None
    module.replay_values = delayed
    app = module.build_app(pipeline=FixturePipeline())
    with app:
        gr.Markdown("**Synthetic CPU scheduling probe.** No emotion classification or language-model inference. The single replay returns OLD REPLAY after a controlled delay.")
    app.validate_queue_settings()
    try:
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
                   prevent_thread_lock=True, show_error=True, enable_monitoring=False)
        print(f"SYNTHETIC ONLY: http://127.0.0.1:{args.port}; bounded lifetime {args.seconds}s", flush=True)
        time.sleep(args.seconds)
    finally:
        app.close()


if __name__ == "__main__":
    main()
