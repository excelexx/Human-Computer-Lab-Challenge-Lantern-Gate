# Confidence reliability check

**Keep the deployed scores unchanged.** This second pass verified the confidence study and found no convincing overall benefit from temperature scaling. No calibration, confidence threshold, or model change was deployed.

The study used **1,109 development utterances across 114 dialogues**. Those examples already influenced model selection or confirmation. Five-fold dialogue cross-fitting keeps temperature fitting separate from its held dialogues, but does **not** make these results independent generalization evidence.

| Population | Utterances | Accuracy, unchanged | NLL: raw → cross-fit | Brier: raw → cross-fit | ECE: raw → cross-fit |
|---|---:|---:|---:|---:|---:|
| All operational predictions | 1,109 | 57.89% | 1.2138 → 1.2137 | .5710 → .5711 | .0169 → .0188 |
| Fusion route | 316 | 61.71% | 1.1823 → 1.1762 | .5516 → .5489 | .0836 → .0854 |
| Text fallback | 793 | 56.37% | 1.2264 → 1.2286 | .5787 → .5799 | .0310 → .0352 |

Lower NLL, Brier, and ECE are better. NLL uses natural logarithms; Brier sums error across seven classes; ECE uses ten fixed confidence bins. All **1,109 predicted categories stay unchanged**. The pooled ECE hides meaningful differences between routes, so it should not be treated as uniformly reliable confidence.

Keeping the highest-confidence half raises pooled accuracy to **72.61%**, but changes the population. The retained fusion subset includes **zero of its three disgust examples**. This selective-coverage result does not justify choosing an automatic threshold, especially for rare classes.

The [independent verification](verification.json) recomputed scores, reliability-bin counts, fixed-coverage rankings, class supports, and fold statistics from saved logits. It checked disjoint dialogue folds, positive bounded temperatures, unchanged categories, matching active-head hashes, and the frozen protocol/source references. It did not rerun model inference or access original feature caches.

The [evidence inventory](artifact-integrity.json) records **367,995 bytes** copied exactly from the completed study. No original feature caches, media, or model binaries are included. Evidence consists of the [protocol](protocol.json), [full results](results.json), [fold and coverage tables](SUMMARY.md), [interpretation notes](NOTES.md), [per-example outputs](dev-confidence.npz), recorded run start, and frozen source snapshot.

From the repository root, the compact check can be repeated without GPU inference:

```powershell
python reports/reliability/confidence-study/verify_evidence.py --heads trained-heads
```
