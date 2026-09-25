# CPU model study and prepared text replacement

Completed 25 September 2026. **The production checkpoints were not changed.** The study found a stronger text classifier, but did not establish an accuracy benefit from adding vision. A replacement text checkpoint is prepared for separate review; the original vision and fusion heads remain the proposed multimodal path.

## Design and provenance

The written `plan.json` preceded fitting. Its fixed search covered standardized, L2-regularized multinomial logistic classifiers for text, vision, and early fusion: C = 0.01, 0.1, 1, 10, each unweighted or inverse-square-root class-frequency weighted. Weight mass was normalized to a mean of one. Standardization and class frequencies used each training fold only.

Three stratified folds grouped all utterances of a MELD dialogue together. The 24 configurations required 72 fits; the three winning configurations were then fitted on all corresponding training rows. The search took 242 seconds on CPU, with PyTorch and BLAS limited to two threads. No encoder, GPU inference, or language generator ran. Environment versions are recorded in `environment.json`.

For each stage, selection used mean seven-class macro F1 across folds, with all validation rows for text and eligible visual rows for vision/fusion. Selected out-of-fold text and vision probabilities were mixed with fixed visual weights 0, .1, .2, .3, .5. Configuration search reused these folds, so the selected CV score is not an unbiased nested-CV estimate. The existing MLP heads were not scored on their own training rows as if those were held-out folds.

`selection.json` was frozen before the study opened the test cache. The benchmark had already been observed in the original project; all subsequent test figures are **reused-benchmark evidence, not a fresh holdout**. There was no automatic promotion or test-driven configuration change.

## Cross-validation and original study results

All three linear stages selected C=.01 and square-root-balanced weighting. All selected CV and full-training fits converged. Some rejected high-C text/fusion fits reached the fixed iteration limit; their warnings remain recorded.

| Model and CV population | Fold 1 | Fold 2 | Fold 3 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Text, all utterances | .3888 | .4046 | .4100 | .4011 ± .0110 |
| Text, eligible visual utterances | .3690 | .4238 | .4294 | .4074 ± .0334 |
| Vision, eligible | .1902 | .1783 | .1744 | .1810 ± .0083 |
| Early fusion, eligible | .3286 | .3491 | .3528 | .3435 ± .0131 |
| 90% text / 10% vision probability fusion, eligible | .3738 | .4195 | .4400 | .4111 ± .0339 |

The three eligible validation folds contain 866/869/806 utterances, including only 16/22/31 disgust cases and 36/22/30 fear cases. Full supports and dialogue counts are in `cv-support.json`. Late fusion's small mean CV gain over its text control was positive in two folds and negative in one.

| Model | Full dev macro F1 | Eligible dev macro F1 | Full reused-test macro F1 | Eligible reused-test macro F1 |
|---|---:|---:|---:|---:|
| Original text MLP | .3688 | .3713 | .3787 | .3750 |
| Original operational fusion/text fallback | .3806 | .4017 | .3784 | .3671 |
| New linear text | .3954 | .3375 | .4158 | .4270 |
| Selected 90/10 late fusion with text fallback | .3946 | .3345 | .4138 | .4208 |

The frozen text control gains .03713 full-test macro F1 over the old text head; a paired dialogue-bootstrap interval is [.01385, .06058]. This descriptive interval was computed from the saved frozen prediction artifact, with no test-cache reopening or model selection. The selected late-fusion model is worse than its own text control by .00203 overall and .00617 on eligible clips; both corresponding intervals cross zero. The stronger representation-to-label classifier, rather than visual information, explains the main improvement.

## Prepared deployment candidate

After the original study, a **single fixed hybrid follow-up** was defined: retain the original vision head and original fusion head, replacing only the text head used when vision is unavailable. This follow-up was explicitly defined after the prior reused-test results were known. Its definition and development scores were frozen in `hybrid-definition.json` before its benchmark pass; no alternate hybrids were tried.

`deployment/text.pt` contains the frozen C=.01 weighted linear text head, with training-only scaler statistics folded into its weight matrix and bias. It is not installed. It carries stage, labels, feature identity, train/dev cache SHA256 values, training count 9,989, dev count 1,109, selected configuration, and source artifact provenance. Its SHA256 is `8c9eb533e4c9419935a1085a9c8799025fa760542ba010f563236f1e0578ca63`.

| Fixed hybrid comparison | Original operational | Prepared hybrid |
|---|---:|---:|
| Full dev macro F1 | .38059 | **.41855** |
| Full reused-test macro F1 | .37845 | **.40324** |
| Full reused-test weighted F1 | **.58479** | .58374 |
| Eligible reused-test macro F1 | .36710 | .36710 |

The paired dialogue-bootstrap full-test macro difference is +.02479, with 95% percentile interval [.00158, .04749], based on 1,000 resamples of 280 dialogues. This is conditional, reused-benchmark evidence; it does not establish prospective generalization. Weighted F1 is slightly lower. Eligible outputs are exactly unchanged: all improvement comes from better text fallback. The new text head alone still scores higher than the hybrid, so no claim of visual added accuracy is supported.

Float32 deployment changes no predicted label on any of the 13,708 cached train/dev/test utterances relative to the frozen float64 model; maximum probability error is below 9.5e-7. The loader supports `architecture="linear"`, defaults old checkpoints to the existing MLP, rejects unknown architectures, and strictly validates weight shapes. Thirty-four targeted numeric, checkpoint, pipeline, and contract tests passed. No physical webcam, generator, or new end-to-end latency measurement was part of this CPU study.

## Prospective parameter count and rollback material

| Required component | Parameters |
|---|---:|
| Qwen3-4B-Instruct-2507 | 4,022,468,096 |
| DeBERTa-v3-large | 434,012,160 |
| EmotiEffLib EfficientNet-B2 including original FER head | 7,710,857 |
| YuNet conservative allowance | 53,121 |
| Existing vision MLP | 181,255 |
| Prepared linear text head | 7,175 |
| Existing fusion MLP | 312,711 |
| **Prospective complete inference path** | **4,464,745,375** |

The replacement reduces the total by 124,928 parameters and leaves 1,535,254,625 below the cap. No extra scaler is required at inference. Keeping the old text model simultaneously required at runtime would instead total 4,464,877,478. The original three checkpoint bytes are preserved with verified hashes under `baseline-checkpoints/`.

Detailed artifacts: `selection.json`, `reused-test-results.json`, `text-control-intervals.json`, `hybrid-definition.json`, `hybrid-reused-test-results.json`, and the corresponding prediction NPZ files. The production head files, model manifests, and reports remain untouched. Any activation, package updates, runtime parameter-label refresh, and further robustness review are separate steps.
