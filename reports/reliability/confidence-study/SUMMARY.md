# Development-only confidence study

This is a diagnostic study, not independent generalization evidence: these development examples already influenced head selection or confirmation. No confidence threshold, deployable calibration, or production change was selected.

The protocol was frozen before computing results. Five dialogue-group folds fit one bounded positive temperature per operational route on the other four folds only. NLL uses natural logs; multiclass Brier sums errors across seven classes; ECE uses ten fixed equal-width bins.

| Population | N | Accuracy | Raw NLL | Cross-fit NLL | Raw Brier | Cross-fit Brier | Raw ECE | Cross-fit ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1109 | 0.5789 | 1.2138 | 1.2137 | 0.5710 | 0.5711 | 0.0169 | 0.0188 |
| fusion | 316 | 0.6171 | 1.1823 | 1.1762 | 0.5516 | 0.5489 | 0.0836 | 0.0854 |
| text_fallback | 793 | 0.5637 | 1.2264 | 1.2286 | 0.5787 | 0.5799 | 0.0310 | 0.0352 |

Changed argmax labels: **0**. Temperatures change probability magnitudes, not predicted categories.

## Temperatures and fold support

| Fold | Route | Fit rows | Held rows | Temperature | Held NLL before | Held NLL after |
|---|---|---:|---:|---:|---:|---:|
| 1 | fusion | 258 | 58 | 0.8321 | 1.2923 | 1.3052 |
| 1 | text_fallback | 645 | 148 | 1.0131 | 1.4595 | 1.4558 |
| 2 | fusion | 267 | 49 | 0.8746 | 1.1247 | 1.1037 |
| 2 | text_fallback | 673 | 120 | 1.1238 | 1.0242 | 1.0476 |
| 3 | fusion | 244 | 72 | 0.8766 | 1.1514 | 1.1335 |
| 3 | text_fallback | 635 | 158 | 1.0744 | 1.1935 | 1.1910 |
| 4 | fusion | 236 | 80 | 0.8892 | 1.0474 | 1.0274 |
| 4 | text_fallback | 604 | 189 | 1.0697 | 1.2288 | 1.2256 |
| 5 | fusion | 259 | 57 | 0.8233 | 1.3484 | 1.3703 |
| 5 | text_fallback | 615 | 178 | 1.0950 | 1.1954 | 1.1982 |

## Fixed coverage diagnostics

Each population keeps its highest-confidence ceil(fraction × N) rows, with ID-based tie-breaking. These fixed slices were not optimized into deployment thresholds. Class supports and reliability-bin counts for every slice are retained in results.json.

| Population | Confidence method | Requested coverage | Retained rows | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|---:|---:|
| all | raw | 100% | 1109 | 0.5789 | 1.2138 | 0.0169 |
| all | raw | 80% | 888 | 0.6351 | 1.1221 | 0.0130 |
| all | raw | 50% | 555 | 0.7261 | 0.9290 | 0.0122 |
| all | cross-fit temperature | 100% | 1109 | 0.5789 | 1.2137 | 0.0188 |
| all | cross-fit temperature | 80% | 888 | 0.6408 | 1.1141 | 0.0213 |
| all | cross-fit temperature | 50% | 555 | 0.7189 | 0.9279 | 0.0101 |
| fusion | raw | 100% | 316 | 0.6171 | 1.1823 | 0.0836 |
| fusion | raw | 80% | 253 | 0.6719 | 1.0601 | 0.0794 |
| fusion | raw | 50% | 158 | 0.7532 | 0.8872 | 0.0512 |
| fusion | cross-fit temperature | 100% | 316 | 0.6171 | 1.1762 | 0.0854 |
| fusion | cross-fit temperature | 80% | 253 | 0.6680 | 1.0601 | 0.0840 |
| fusion | cross-fit temperature | 50% | 158 | 0.7405 | 0.9035 | 0.0501 |
| text_fallback | raw | 100% | 793 | 0.5637 | 1.2264 | 0.0310 |
| text_fallback | raw | 80% | 635 | 0.6268 | 1.1352 | 0.0327 |
| text_fallback | raw | 50% | 397 | 0.7179 | 0.9324 | 0.0295 |
| text_fallback | cross-fit temperature | 100% | 793 | 0.5637 | 1.2286 | 0.0352 |
| text_fallback | cross-fit temperature | 80% | 635 | 0.6283 | 1.1388 | 0.0394 |
| text_fallback | cross-fit temperature | 50% | 397 | 0.7179 | 0.9391 | 0.0259 |

Cross-fitting protects the temperature fit from its held dialogue labels; it does not remove earlier head-selection bias. Differences are descriptive, ECE is sensitive to binning and small supports, and high-confidence slices may favor common classes. There was no test-cache access, no GPU/encoder/generator inference, no final full-dev temperature fit, and no changes to runtime parameters.

Reproduce with `python -m checkin.confidence_study --home <artifacts> --output <new-study-directory> plan`, then the same command ending in `run`. Exact train/dev cache bytes and active checkpoint hashes must match the frozen protocol. Existing protocols and runs are never overwritten.
