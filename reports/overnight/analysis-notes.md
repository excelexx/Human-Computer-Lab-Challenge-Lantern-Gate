# Independent evidence review and one focused follow-up

Reviewed saved study schemas, original statistics, hybrid definition/results, and response-study outputs. No inference, fitting, threshold selection, or test-driven change was performed for this review. Robustness video extraction is still running, so its final figures are not assumed here.

## What the classifier evidence supports

The fixed hybrid replaces only missing-vision text fallback with the preselected linear head. Original fusion remains active for eligible vision. Full dev macro F1 rises from .38059 to .41855; weighted F1 .55155 to .57116; accuracy .56988 to .57890. Eligible predictions are unchanged by construction. The prospective runtime total falls to 4,464,745,375 parameters.

The reused-test macro improvement (.37845 to .40324) conceals a tradeoff: accuracy falls .60115 to .59042 and weighted F1 .58479 to .58374. The largest class gain is disgust, with 16/68 correct instead of 1/68; anger falls from 167/345 correct to 125/345. Neutral loses 38 correct examples and fear loses one. These are descriptive counts from an already observed benchmark, not an invitation to tune against it.

| Class | Dev F1 old → hybrid | Reused-test F1 old → hybrid |
|---|---:|---:|
| Neutral | .7201 → .7280 | .7736 → .7615 |
| Surprise | .5473 → .5769 | .4824 → .5059 |
| Fear | .1333 → .1622 | .1522 → .1188 |
| Sadness | .2967 → .3529 | .2708 → .3025 |
| Joy | .4968 → .4969 | .4955 → .5024 |
| Disgust | .0000 → .1200 | .0282 → .2302 |
| Anger | .4699 → .4929 | .4465 → .4013 |

The new text-only model still exceeds hybrid macro F1 on reused test (.41583 versus .40324). No improvement attributable to vision is established. The original operational-minus-text macro difference is essentially zero, with dialogue-bootstrap interval spanning zero. Its agreement/disagreement strata also have wide overlapping-zero intervals; disagreement alone does not justify a new gate. The full-split and common-visual report repeat the same eligible agreement/disagreement cohorts (278 agreement, 380 disagreement), so those are not independent findings. Dialogue grouping helps within-dialogue dependence but does not remove shared-speaker/episode dependence or reused-benchmark bias.

Sources: `model-study/hybrid-definition.json`, `model-study/hybrid-reused-test-results.json`, `model-study/SUMMARY.md`, `baseline-statistics.json`. Hybrid was explicitly defined after prior reused-test results were known; prospective evaluation is still missing.

## Response evidence and its connection to multimodal behavior

The 150-request controlled prompt study supports narrower grounding on selected development examples, especially an ambiguous fragment and the actual dev guitar utterance. It also shows harmful visual-state influence: joy encourages congratulations for explicitly unwanted responsibility; sadness can be treated as a reported feeling when the text says no such thing. Other unresolved categories are fabricated shared grief, assumed welcome/positivity, omitted mixed feelings, and implied unsupported contact options.

The separate 54-request production-sampler replication retained these limitations and introduced a normal-reaction suggestion in a bodily-experience reply. All 18 deterministic reserved replies passed the narrow regex checks despite qualitative defects. Regex passes are not quality or clinical scores. Two of six identical fixed-seed/temperature-zero controls differed, so any exact text change across visual conditions is insufficient evidence of beneficial causal adaptation. The production replication contains only the selected prompt: it cannot estimate comparative improvement over the production baseline.

## Recommended single follow-up: replay availability transitions

After robustness extraction finishes, compare the preserved baseline and fixed hybrid on the same immutable saved dev feature rows for all eight conditions. Load records through the existing artifact/hash/protocol validator, bind both models to their checkpoint metadata and frozen hashes, and first reproduce saved baseline probabilities within a declared numeric tolerance. Failed extraction rows remain explicit errors, not missing-vision fallback observations.

For each condition report paired baseline/hybrid macro and weighted F1, per-class supports, coverage, fallback rate, label changes and distribution movement. Split results by the original-to-transformed availability transition: eligible→eligible, eligible→fallback, fallback→eligible, fallback→fallback. Require all hybrid outputs still eligible for fusion to be exactly unchanged. Any improvement or regression must then be traced to the changed fallback head, including the black-frame control. Also compare condition predictions with each model's own original-video outputs on the same valid cohort.

This answers a focused integration question: **does degraded vision switch the system to the new text head correctly, and what class tradeoffs does that introduce?** It needs only saved features and small CPU heads. It uses dev data already involved in selection, class-stratified samples rather than natural prevalence, and altered videos whose emotion evidence may have changed; report it as post-hoc stress sensitivity, never independent accuracy or a new model-selection exercise. Do not tune a gate or prompt from these results. Root has authorized this exact follow-up; implementation proceeds separately in `robustness_compare.py` without changing the active extraction code.

## Completed replay result

The follow-up completed on 95 utterances × eight conditions, with all 760 rows valid and no extraction failures. Preserved baseline predictions reproduced with maximum absolute probability error 1.79e-7 against the predeclared tolerance 2e-6; every label matched. All 649 rows with eligible vision were **exactly unchanged** under the candidate. The 111 fallback condition rows are repeated measurements of these same utterances, not 111 independent situations. CPU replay took 1.17 seconds, excluding the earlier extraction.

| Condition | Fallback / 95 | Old macro F1 | Hybrid macro F1 | Changed candidate labels |
|---|---:|---:|---:|---:|
| Original | 0 | .33072 | .33072 | 0 |
| Silent stream copy | 0 | .33072 | .33072 | 0 |
| H.264 reencode | 0 | .32957 | .32957 | 0 |
| Half resolution | 7 | .31342 | .31020 | 4 |
| Half brightness | 2 | .31662 | .32424 | 1 |
| Gaussian blur | 5 | .25425 | .28184 | 3 |
| Horizontal flip | 2 | .32531 | .32671 | 2 |
| Black control | 95 | .31177 | .28958 | 31 |

Black-control accuracy loses one correct example out of 95, and half-resolution macro F1 also regresses. Blur gains three correct examples; brightness gains one. These mixed outcomes should accompany the full-dev fallback gain, not be hidden by it. Original eligibility and transformed eligibility are identical for both model versions; all differences are caused by fallback-head predictions. Silent stream copy matches original predictions, while reencoding changes one label even though availability remains 95/95. This supports retaining the lossless mute path.

Full comparison and per-class/transition tables: `robustness-hybrid-comparison.json`. Source: `outputs/checkin/src/checkin/robustness_compare.py`; focused replay plus extraction-contract tests: **26 passed in 4.25s**. No thresholds, prompts, or models were retuned from this result.
