# Interpretation and handoff

**Keep the deployed confidence behavior unchanged.** This bounded diagnostic did not show a convincing overall benefit from route-specific temperature scaling. Its purpose was to measure the current scores, not select new production thresholds or calibration parameters.

The protocol was frozen before results, binding both feature-cache hashes, the common train/dev feature identity, all three active head hashes, the study source, five dialogue-group folds, temperature bounds [.25, 4], ten fixed confidence bins, and coverage fractions 1/.8/.5. Evaluation used only the 1,109 development examples across 114 dialogues. The development data already influenced the heads' selection or confirmation, so neither raw nor cross-fit results are independent generalization evidence.

- Operational NLL changes by only **−0.00015 nats** with cross-fitting. Brier changes by **+0.000064** and ECE by **+0.00189**, both slightly worse. No significance claim is supported.
- Pooling routes conceals differences. Fusion accuracy is .6171 with mean confidence .5381 and ECE .0836. Text fallback accuracy is .5637 with mean confidence .5820 and ECE .0310. The pooled ECE of .0169 should not be read as uniformly reliable confidence across routes or classes.
- Positive fitted temperatures preserve every argmax label. Fusion temperatures range .8233–.8892; fallback temperatures range 1.0131–1.1238. Every temperature and fitting/held support appears in `results.json`; only the other four dialogue folds contribute to a given fit.
- Fixed high-confidence subsets are easier, but less representative. At raw 50% coverage, operational accuracy rises from .5789 to .7261 while retaining 555 examples. The fusion subset retains **none of its three disgust examples**. This is not evidence for an automatic abstention threshold.

`results.json` contains raw text-only diagnostics, operational and route-specific scores, full ten-bin tables, class supports, all fixed-coverage slices, and every fold result. `dev-confidence.npz` preserves IDs, routes, labels, logits, fold assignments, and held-fold temperatures. `source_snapshot.py` matches the frozen source hash. `SUMMARY.md` presents the main tables and reproduction commands.

Ten focused CPU tests passed: metric definitions and numerical stability, fixed-coverage rounding/ties, positive bounded temperatures, held-dialogue and route isolation, no test-cache access, unchanged active checkpoints, and refusal to overwrite a protocol or repeat a run. No GPU, encoder, language-model, or live-interaction inference occurred. No final full-dev temperature was fitted; no threshold or calibration was deployed; no runtime parameter counts or prior study artifacts changed.
