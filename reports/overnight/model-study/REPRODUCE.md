# Reproduce the prepared text head

The repository now provides the preparation step directly:

```powershell
python -m checkin.model_study --home $Artifacts --output $Study --baseline-heads $OriginalHeads --candidate-output $Candidate prepare
```

This uses the already selected affine text weights, checks their hashes, converts them to the runtime checkpoint format, verifies numeric parity, and reports the fixed hybrid on development data. It does not refit a model, open the test cache, load an encoder/generator, use a GPU, or install the candidate.

## Original workspace example

Run from the `outputs/checkin` repository root with its Python environment active. These paths refer to the original completed study:

```powershell
$Artifacts = (Resolve-Path ..\..\work\runtime).Path
$Study = (Resolve-Path ..\..\work\overnight\model-study).Path
$OriginalHeads = Join-Path $Study "baseline-checkpoints"
$Candidate = Join-Path $Study "prepared-again"
python -m checkin.model_study --home $Artifacts --output $Study --baseline-heads $OriginalHeads --candidate-output $Candidate prepare
```

Choose a new candidate directory. Preparation refuses to overwrite an existing candidate and refuses destinations inside the production artifact tree or preserved-baseline directory. If `--candidate-output` is omitted, it uses `$Study/deployment`; that directory already contains the original candidate in this handoff, so a reproduction should supply a new destination.

The prepared directory receives:

- `text.pt`: runtime-compatible linear text checkpoint with labels, dimensions, selected configuration, cache hashes, feature identity, and folded-scaler provenance.
- `hybrid-definition.json`: exact checkpoint hash, preserved-baseline hashes, original and preparation source hashes, train/dev parity, development scores, and prospective parameter counts.
- `hybrid-dev-predictions.npz`: the fixed baseline-fusion/new-text prediction record.
- `preparation-started.json`: a collision guard and preparation provenance.

The hybrid always uses the frozen baseline fusion head for eligible visual clips and the new text head when vision is unavailable. The baseline vision head remains available for diagnostics. There is no search among additional hybrid configurations.

## Required inputs and changed production checkpoints

The study directory must contain its original `plan.json`, `selection.json`, and `frozen-models/text.npz`. `$Artifacts/cache/train.npz` and `dev.npz` must exactly match the recorded cache SHA256 and feature identity. `$OriginalHeads` must contain the three original `vision.pt`, `text.pt`, and `fusion.pt` files with the hashes frozen in the plan. The bundled study's `baseline-checkpoints` directory preserves those originals.

**The portable repository/report bundle does not include the original train/dev feature-cache NPZ files.** The prepared trained heads support immediate inference after installing the pinned encoders and generator, but reproducing the historical preparation and development checks additionally requires those original cache bytes. Re-extracting the same dataset with the same model names is not a guarantee of identical cache hashes across environments. If newly generated caches differ, keep their hashes honest: extract and train matching baseline heads in a separate artifacts directory, then start a new study plan/search against those new caches and baselines. Do not edit the historical plan, checkpoint metadata, or expected hashes to make different data pass the original freeze.

Use `--baseline-heads` when production has since been promoted to the new text head. The preparation command verifies the preserved files; it does not silently compare the candidate with changed production weights. The same option is available to `plan`, `search`, and `evaluate`; a new plan records the chosen baseline directory. Later phases verify the recorded hashes, including when an alternate directory containing the exact same baseline bytes is supplied.

Preparation may consume a completed schema-1 study produced by an earlier version of the study code. It preserves the original plan/selection/source hashes and records the preparation implementation's separate source hash. It does not revise the earlier freeze or claim that new code generated the historical selection. The exact original search source is archived at `source-snapshots/model_study_search_v1.py` and matches the source hash in the original plan.

## Starting a new reproduction of the search

Use a **new study directory** for a new declared search; never rewrite the existing plan to bypass a source or checkpoint mismatch. With cached features and preserved baseline heads already available:

```powershell
$NewStudy = Join-Path (Split-Path $Study) "model-study-reproduction"
python -m checkin.model_study --home $Artifacts --output $NewStudy --baseline-heads $OriginalHeads plan
python -m checkin.model_study --home $Artifacts --output $NewStudy --baseline-heads $OriginalHeads search
python -m checkin.model_study --home $Artifacts --output $NewStudy --baseline-heads $OriginalHeads prepare
```

`search` selects only from training-dialogue CV and development data. Search/evaluation continue to require the exact source revision recorded by their own plan. The separate `evaluate` phase remains single-use and refuses a second access in the same study directory. It is not needed to prepare the head. Any additional evaluation on the official test split must be described as reusing an already observed benchmark, not a fresh holdout; do not remove evaluation guards to tune repeatedly against it.

## Verified reproduction

The repository CLI was run against the existing frozen selection, using preserved baseline heads and a new `reproduced-deployment` destination. It produced exactly the same float32 weight tensors, unchanged train/dev labels, identical hybrid development metrics, and the same **4,464,745,375** prospective parameter count as the original candidate. It did not access test data or alter the original deployment artifact. Serialized checkpoint hashes differ because the new file records its own creation time and preparation provenance; its learned tensor values match exactly.

See `preparation-reproducibility-check.json` for this check. Forty focused CPU tests passed, covering numeric parity, legacy/new loader behavior, provenance mismatch rejection, explicit original-baseline use after production changes, no test access during preparation, collision protection, and CLI dispatch. No GPU tests or production activation were performed by this preparation task.
