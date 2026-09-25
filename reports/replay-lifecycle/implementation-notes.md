# Replay lifecycle evidence

The frozen baseline is `app-baseline-47ff75b.py`, commit
`47ff75b1286cafcb51322f1631078b4af152ad0b`, SHA-256
`5591ce132135ce39a3c3d1ebf53892f10baeff23b8b7aa20bad7c63ebcd8d2d4`.
`baseline_probe.py` reproduces the delayed replay result after reset through
Gradio's actual `process_api` and per-browser `SessionState`; its output is
`baseline-result.json`. This is synthetic scheduling, not inference or a browser
claim. Root separately owns the real browser baseline evidence.

The revised application uses two cancellable queued generators. Admission has
its own short queue, so another browser's active GPU turn does not block it.
It checks a hidden non-State epoch captured when clicked against that browser's
mutable guard. Admission publishes a unique ticket and disables the composer,
replay selector, and load button. A ticket change starts loading on the same
concurrency group as Send. A fixed non-State ticket prevents a queued loader
from adopting a new request's identity. Ticket change is used instead of `.then`
because a cancellation completion can still trigger `.then` in pinned Gradio.

New, Stop, Send, and direct composer/replay-selector/media input invalidate the
guard and cancel both events. Direct input listeners avoid broad `change` events
that would cancel programmatic replay values. Late work yields `gr.skip()` for
every output. Cancellation does not let an old iterator restore controls owned
by newer work. The controlling callback restores or takes over those controls,
including when Gradio closes the old iterator first.

Validation: 61 CPU tests passed across `test_replay_lifecycle.py`, `test_ui.py`,
`test_lifecycle.py`, and `test_pipeline.py`. Sixteen focused replay cases cover
launch queue validation, persistent same-browser guards, independent browser
guards, captured admission rejected after invalidation, queued and in-progress
loading, retained old iterators, restoration in both cancellation orders, and
success/failure paths. One pre-existing Starlette deprecation warning remains.
Current app SHA-256 at this check:
`caff825c1bdc7c75c7c73a7bc7252226b39203e60c51ada0f74d78f19d479622`.

`current_browser_service.py` uses the production `build_app` and event config
with a clearly synthetic pipeline, one OLD REPLAY/None row, and a 2.5-second
delay in replay value preparation. It defaults to loopback port 7863 and exits
after 600 seconds. It never loads models and was supplied without starting it.

Boundary: Python guards cannot retract data already dispatched to the browser.
Gradio's per-browser cancellation handles that transport boundary. These tests
prove the specified backend scheduling cases, not atomic delivery against
arbitrary packet reordering. Root's controlled browser verification is separate.
