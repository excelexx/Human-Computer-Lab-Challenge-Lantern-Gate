# Replay loading: preserving a fresh check-in

The old UI could replace a newly typed draft with a delayed MELD replay after **New conversation**. A controlled real-browser comparison reproduced this overwrite. The current UI preserved the fresh draft after both **New conversation** and **Stop**, while normal replay loading still completed.

This pass changes the replay interaction only. Models, classifier heads, generator prompt and sampler are unchanged. The complete learned-parameter allowance remains **4,464,745,375**, below six billion. Earlier model-selection, corruption and response studies remain frozen.

## Evidence and scope

The [plan](plan.json) was written after exploratory CPU discovery but before the browser comparison. Both browser fixtures used the actual Gradio application with one synthetic `OLD REPLAY` row, no video, no inference, and the same explicit 2.5-second delay. The old source is frozen from revision `47ff75b1286cafcb51322f1631078b4af152ad0b`.

| Browser sequence | Old source | Current source |
|---|---|---|
| Load replay, New, type a fresh draft, let loader finish | Draft replaced by OLD REPLAY | Fresh draft preserved and editable |
| Load replay, Stop, type a fresh draft, let loader finish | Not separately measured | Fresh draft preserved and editable |
| Load replay normally | Not separately measured | OLD REPLAY loaded; controls restored |

While loading, the current UI disables the composer, replay selector and Send, while keeping Stop available. Queued admission captures an immutable epoch and publishes a request ticket. The loader checks ownership before and after preparing values. Reset, Stop and user input invalidate the request; stale callbacks skip their outputs. The loader uses the same concurrency group as Send. See [implementation notes](implementation-notes.md) and [recorded browser observations](browser-observations.json).

The full CPU software suite passed **268 tests**, including 16 focused replay tests. One pre-existing dependency deprecation warning remains. The [test log](software-tests.txt) and [JUnit results](software-tests.xml) are included. Tests cover launch validity, stale admission, queued/running cancellation, independent guards, retained iterators and success/failure restoration. These are software checks, not 268 multimodal model evaluations.

## Actual multimodal smoke check

After restarting the production app, recorded MELD test utterance **4/2**, “It kicked! I think the baby kicked!”, loaded and ran through text plus all eight selected face frames. Its structured state reported `fusion`, available vision and disagreement (text: joy; vision: neutral). Local Qwen streamed a complete reply and the app cleared the used input.

The **anger** output is wrong against the **surprise** reference. Its reply was:

> It kicked! I think the baby kicked — that’s really exciting. How does that feel right now?

This presumes excitement and echoes the speaker's first-person wording. These are retained failures, not reasons to tune on this already exposed test example. No new accuracy or response-quality score is claimed. Classification/first-token/completion timings were 224/434/584 ms after a separate 9,694 ms cold model load; this one smoke check is not a new benchmark. The unchanged backend's prior benchmark remains [241/440/604 ms warm fusion p95](../reliability/REPORT.md), with 7,581 MiB peak total GPU use on the RTX 3080.

Both bounded synthetic services exited. The real app was left warm on loopback port 7860, with an empty conversation and Camera tab ready. Physical webcam capture was not exercised.

## Reproduction and limits

From the repository's configured Python environment, run `python -m pytest tests/test_replay_lifecycle.py`. The included `baseline_probe.py` reproduces the old scheduling issue without a browser. The baseline browser helper serves port 7862 for 150 seconds. The current helper accepts `--seconds 150`, serves port 7863, and imports the installed current application. Follow the sequences in the table manually; do not count synthetic fixture interactions as inference. Both fixtures intentionally replace model operations. Set up the project environment as documented in the main README first.

Only one browser was observed. Backend guards cannot retract updates already dispatched to the frontend; Gradio's cancellation still controls that transport boundary. The result does not prove atomic reset under arbitrary network reordering, cross-browser queue isolation, or hours of continuous stability. There is still no demonstrated visual accuracy advantage or clinical validation. Full source and artifact hashes are covered by the repository handoff inventory.
