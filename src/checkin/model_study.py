"""CPU-only, preregistered linear-model study; never changes production heads.

Run ``python -m checkin.model_study --home ... --output ... plan``, then
``... search`` and ``... evaluate``. Evaluation consumes the previously observed
official test benchmark once, after an immutable selection artifact is written.
The ``prepare`` phase converts the frozen text winner into a deployable checkpoint
and reports the fixed baseline-fusion/new-text hybrid on development data only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from .evaluate import classification_metrics
from .downloads import model_manifest
from .models import DIMENSIONS, LinearEmotionHead, load_head
from .settings import LABELS
from .train import atomic_checkpoint, atomic_json, file_sha256, load_cache, predict_probabilities


CS = (0.01, 0.1, 1.0, 10.0)
WEIGHTS = ("unweighted", "sqrt_balanced")
LATE_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5)
SEED = 42
FOLDS = 3
MAX_ITER = 500


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def groups_from_ids(ids):
    """Official IDs encode split:dialogue:utterance; never split a dialogue."""
    groups = []
    for identity in ids:
        parts = str(identity).split(":")
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            raise ValueError(f"Malformed MELD identity: {identity}")
        groups.append(":".join(parts[:2]))
    return np.asarray(groups)


def features(cache, stage):
    if stage in {"vision", "text"}:
        return cache[stage]
    if stage == "fusion":
        return np.concatenate((cache["vision"], cache["text"], cache["quality"]), axis=1)
    raise ValueError(stage)


def eligibility(cache):
    return cache["quality"][:, 2] > 0


def baseline_directory(home, plan=None, override=None):
    return Path(override or (plan or {}).get("baseline_heads") or Path(home) / "checkpoints").resolve()


def verify_baseline_heads(home, plan, baseline_heads=None):
    directory = baseline_directory(home, plan, baseline_heads)
    for stage, digest in plan["baseline_head_sha256"].items():
        path = directory / f"{stage}.pt"
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"The frozen baseline {stage} checkpoint is missing or changed. "
                             "Pass --baseline-heads with the preserved original checkpoint directory.")
    return directory


def weights_for(labels, mode):
    if mode == "unweighted":
        return None
    if mode != "sqrt_balanced":
        raise ValueError(mode)
    counts = np.bincount(labels, minlength=7)
    class_weights = np.zeros(7, dtype=np.float64)
    class_weights[counts > 0] = 1.0 / np.sqrt(counts[counts > 0])
    weights = class_weights[labels]
    return weights / weights.mean()  # Keep average objective mass/C comparable.


def fit_linear(x, y, config):
    """Scaler statistics and class frequencies are fitted only to these rows."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x).astype(np.float64)
    classifier = LogisticRegression(
        C=config["C"], solver="lbfgs", penalty="l2", max_iter=MAX_ITER,
        tol=1e-4, random_state=SEED,
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaled, y, sample_weight=weights_for(y, config["weighting"]))
    if classifier.classes_.tolist() != list(range(7)):
        raise ValueError("A training fold lacks a MELD category; no silent class remapping is allowed.")
    # Fold data-estimated scaler statistics into the learned affine classifier.
    # Inference requires just the resulting matrix and bias, with no extra state.
    coefficient = classifier.coef_ / scaler.scale_[None, :]
    intercept = classifier.intercept_ - coefficient @ scaler.mean_
    artifact = {"coefficient": coefficient, "intercept": intercept}
    expected = classifier.predict_proba(scaler.transform(x[:32]).astype(np.float64))
    if not np.allclose(predict_linear(artifact, x[:32]), expected, atol=1e-6, rtol=1e-6):
        raise AssertionError("Folding standardization changed predictions.")
    return artifact, {
        "fit_seconds": time.perf_counter() - started,
        "iterations": int(classifier.n_iter_.max()),
        "converged": not any(issubclass(item.category, ConvergenceWarning) for item in observed),
        "warnings": [str(item.message) for item in observed],
        "learned_inference_parameters": int(coefficient.size + intercept.size),
    }


def predict_linear(model, x):
    result = softmax(np.asarray(x, dtype=np.float64) @ model["coefficient"].T + model["intercept"], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite classifier probabilities.")
    return result


def mix_probabilities(text, vision, available, weight):
    if not 0 <= weight <= 1 or text.shape != vision.shape or available.shape != (len(text),):
        raise ValueError("Invalid probability-fusion inputs.")
    result = text.copy()
    result[available] = (1.0 - weight) * text[available] + weight * vision[available]
    return result


def quick_metrics(truth, probability):
    return classification_metrics(truth, probability.argmax(axis=1))


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def make_plan(home, output, baseline_heads=None):
    """Write the entire search space and decision rules before running a fit."""
    train, dev = load_cache(home, "train"), load_cache(home, "dev")
    if train["feature_identity"] != dev["feature_identity"]:
        raise ValueError("Train/dev feature identities differ.")
    output.mkdir(parents=True, exist_ok=True)
    baseline_heads = baseline_directory(home, override=baseline_heads)
    plan = {
        "schema_version": 1, "created_at": utc_now(),
        "purpose": "Bounded CPU-only follow-up; production models remain unchanged.",
        "study_source_sha256": file_sha256(Path(__file__)),
        "home": str(home.resolve()), "output": str(output.resolve()),
        "baseline_heads": str(baseline_heads),
        "feature_identity": train["feature_identity"],
        "cache_sha256": {"train": train["sha256"], "dev": dev["sha256"]},
        "baseline_head_sha256": {stage: file_sha256(baseline_heads / f"{stage}.pt") for stage in DIMENSIONS},
        "training_rows": len(train["ids"]), "development_rows": len(dev["ids"]),
        "cv": {"type": "StratifiedGroupKFold", "group": "MELD dialogue ID", "folds": FOLDS, "seed": SEED,
               "variance": "Report each disjoint-dialogue fold and sample standard deviation; three folds are not three independent experiments."},
        "linear": {"stages": list(DIMENSIONS), "C": list(CS), "weighting": list(WEIGHTS),
                   "solver": "lbfgs", "max_iter": MAX_ITER, "tolerance": 1e-4,
                   "standardization": "Fit mean/scale to each training fold only; fold statistics into final affine weights.",
                   "text_training": "all training rows", "vision_training": "visually eligible training rows",
                   "fusion_training": "all training rows; original normalized vision/text plus three quality values",
                   "weighting_definition": "Inverse square root training class frequency, per-row weights normalized to mean one."},
        "late_fusion": {"vision_probability_weights": list(LATE_WEIGHTS), "missing_vision": "exact text fallback"},
        "selection_rules": [
            "Select one configuration per linear stage by highest mean dialogue-CV seven-class macro F1: all rows for text, eligible rows for vision/fusion.",
            "Break exact configuration ties by lower C, then unweighted loss; do not use official test data.",
            "Select a late-fusion weight by highest mean eligible dialogue-CV macro F1; exact ties choose less visual weight. Also retain the best strictly positive weight as a multimodal candidate.",
            "Fit each selected stage to all corresponding train rows, then inspect dev once. Do not train on dev.",
            "Early fusion and best positive-weight late fusion qualify for selection only if mean eligible CV improvement over the selected linear text model is positive and at least two of three folds improve.",
            "Among qualifying multimodal families choose highest eligible dev macro F1, then full operational dev macro F1. If neither qualifies, select the higher-CV family for diagnosis and mark it unsupported for promotion.",
            "Write selection.json with fitted weight hashes, all decisions, and timestamp BEFORE opening the test cache.",
            "Evaluate frozen text/vision/fusion controls and the single selected multimodal candidate together once on the reused official test benchmark. This is not a fresh holdout because the initial project already observed this test set.",
            "Do not promote a candidate automatically. Robust-improvement reporting requires agreement across CV, dev, and a dialogue-bootstrap interval for paired reused-test differences; reused-test results cannot remove prior selection bias."
        ],
        "uncertainty": {"paired_dialogue_bootstrap_replicates": 1000, "seed": 314159,
                        "interval": "2.5 and 97.5 percentiles; conditional on these frozen models and this previously observed benchmark"},
        "test_access": "No test cache, labels, predictions, or previous test metrics are used by search. Evaluation is a separate single-use phase.",
        "resource_limit": "CPU only; torch and BLAS capped at two threads; no encoders or language generator loaded.",
        "production_promotion": "forbidden; baseline checkpoints are read-only",
    }
    target = output / "plan.json"
    if target.exists():
        raise FileExistsError("A plan already exists. Reuse it with search; use a new output directory for a new declared study.")
    atomic_json(target, plan)
    return plan


def read_plan(home, output, baseline_heads=None):
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan["study_source_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Study source changed after preregistration; create a new explicitly documented plan.")
    if plan["home"] != str(home.resolve()):
        raise ValueError("Plan refers to another artifact home.")
    verify_baseline_heads(home, plan, baseline_heads)
    return plan


def validate_caches(plan, *caches):
    for cache in caches:
        if cache["feature_identity"] != plan["feature_identity"]:
            raise ValueError("Feature identity changed.")
        if cache["split"] in plan["cache_sha256"] and cache["sha256"] != plan["cache_sha256"][cache["split"]]:
            raise ValueError("Train/dev feature bytes changed.")


def baseline_probabilities(home, cache, plan, baseline_heads=None):
    directory = verify_baseline_heads(home, plan, baseline_heads)
    probabilities = {}
    for stage in DIMENSIONS:
        model, payload = load_head(directory / f"{stage}.pt", device="cpu")
        if payload.get("feature_identity") != plan["feature_identity"] or payload.get("cache_sha256") != plan["cache_sha256"]:
            raise ValueError("Baseline checkpoint provenance differs from the study features.")
        probabilities[stage] = predict_probabilities(model, features(cache, stage), device="cpu")
    probabilities["operational"] = probabilities["fusion"].copy()
    probabilities["operational"][~eligibility(cache)] = probabilities["text"][~eligibility(cache)]
    return probabilities


def baseline_parameter_inventory(home, plan, baseline_heads=None):
    """Count the frozen baseline actually used, independently of current heads."""
    directory = verify_baseline_heads(home, plan, baseline_heads)
    head_counts = {}
    for stage, dimension in DIMENSIONS.items():
        model, payload = load_head(directory / f"{stage}.pt", device="cpu")
        if (payload.get("stage") != stage or payload.get("input_dim") != dimension
                or payload.get("labels") != LABELS
                or payload.get("feature_identity") != plan["feature_identity"]
                or payload.get("cache_sha256") != plan["cache_sha256"]):
            raise ValueError("Baseline checkpoint contract/provenance differs from the study.")
        head_counts[stage] = sum(parameter.numel() for parameter in model.parameters())
    head_names = {f"MELD {stage} head" for stage in DIMENSIONS}
    pretrained = {component["name"]: component["parameters"] for component in model_manifest()["components"]
                  if component["name"] not in head_names}
    return {"baseline_heads": head_counts, "pretrained_components": pretrained,
            "baseline_complete_total": sum(pretrained.values()) + sum(head_counts.values()),
            "count_scope": "Verified frozen study baseline heads plus pinned project pretrained architecture counts; may differ from current production.",
            "baseline_head_sha256": plan["baseline_head_sha256"]}


def comparisons(cache, probabilities):
    available = eligibility(cache)
    return {
        "all": {name: quick_metrics(cache["labels"], values) for name, values in probabilities.items() if name != "vision"},
        "eligible": {name: quick_metrics(cache["labels"][available], values[available]) for name, values in probabilities.items()},
    }


def fold_summary(cache, probability, assignment, eligible_only):
    scores = []
    for fold in range(FOLDS):
        mask = assignment == fold
        if eligible_only:
            mask &= eligibility(cache)
        scores.append(quick_metrics(cache["labels"][mask], probability[mask])["macro_f1"])
    return {"fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores)),
            "std_macro_f1": float(np.std(scores, ddof=1))}


def search(home, output, baseline_heads=None):
    plan = read_plan(home, output, baseline_heads)
    if (output / "selection.json").exists():
        raise FileExistsError("Selection is already frozen. Do not rerun search in this study directory.")
    train, dev = load_cache(home, "train"), load_cache(home, "dev")
    validate_caches(plan, train, dev)
    groups = groups_from_ids(train["ids"])
    folds = list(StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(train["text"], train["labels"], groups))
    assignment = np.full(len(groups), -1, dtype=np.int64)
    for fold, (fit_rows, validation_rows) in enumerate(folds):
        if set(groups[fit_rows]) & set(groups[validation_rows]):
            raise AssertionError("Dialogue leakage in cross-validation.")
        assignment[validation_rows] = fold
    save_arrays(output / "folds.npz", ids=train["ids"], fold=assignment, groups=groups)
    stage_winners, selected_oof, candidates = {}, {}, []
    all_started = time.perf_counter()
    for stage in ("text", "vision", "fusion"):
        x = features(train, stage)
        stage_candidates = []
        for c in CS:
            for weighting in WEIGHTS:
                name = f"{stage}-C{c:g}-{weighting}"
                config = {"stage": stage, "C": c, "weighting": weighting}
                path = output / "cv" / f"{name}.json"
                probability_path = path.with_suffix(".npz")
                if path.exists() and probability_path.exists():
                    report = json.loads(path.read_text(encoding="utf-8"))
                    probability = load_arrays(probability_path)["probability"]
                else:
                    probability = np.full((len(train["ids"]), 7), 1 / 7, dtype=np.float64)
                    fit_reports = []
                    for fold, (fit_rows, validation_rows) in enumerate(folds):
                        if stage == "vision":
                            fit_rows = fit_rows[eligibility(train)[fit_rows]]
                            validation_rows = validation_rows[eligibility(train)[validation_rows]]
                        model, fitted = fit_linear(x[fit_rows], train["labels"][fit_rows], config)
                        probability[validation_rows] = predict_linear(model, x[validation_rows])
                        fit_reports.append({"fold": fold, "train_count": len(fit_rows), "validation_count": len(validation_rows), **fitted})
                        print(f"{name} fold {fold + 1}/{FOLDS}: {fitted['fit_seconds']:.1f}s; converged={fitted['converged']}", flush=True)
                    report = {"name": name, "config": config, "fits": fit_reports,
                              "selection": fold_summary(train, probability, assignment, stage != "text"),
                              "eligible": fold_summary(train, probability, assignment, True),
                              "all_fits_converged": all(item["converged"] for item in fit_reports)}
                    save_arrays(probability_path, probability=probability)
                    atomic_json(path, report)
                candidates.append(report)
                stage_candidates.append(report)
        ranked = sorted(stage_candidates, key=lambda item: (-item["selection"]["mean_macro_f1"], item["config"]["C"], item["config"]["weighting"] != "unweighted"))
        winner = ranked[0]
        stage_winners[stage] = winner
        selected_oof[stage] = load_arrays(output / "cv" / f"{winner['name']}.npz")["probability"]
        print(f"Selected {stage}: {winner['name']} CV={winner['selection']['mean_macro_f1']:.4f}", flush=True)
        atomic_json(output / "cv-progress.json", {"selected_stages": stage_winners, "completed_candidates": candidates})

    late_candidates = []
    for weight in LATE_WEIGHTS:
        probability = mix_probabilities(selected_oof["text"], selected_oof["vision"], eligibility(train), weight)
        late_candidates.append({"vision_weight": weight, **fold_summary(train, probability, assignment, True)})
    late_best = max(late_candidates, key=lambda item: (item["mean_macro_f1"], -item["vision_weight"]))
    late_positive = max((item for item in late_candidates if item["vision_weight"] > 0),
                        key=lambda item: (item["mean_macro_f1"], -item["vision_weight"]))
    full_models, dev_probabilities, fit_reports, hashes = {}, {}, {}, {}
    for stage, winner in stage_winners.items():
        mask = eligibility(train) if stage == "vision" else np.ones(len(train["ids"]), dtype=bool)
        model, fitted = fit_linear(features(train, stage)[mask], train["labels"][mask], winner["config"])
        full_models[stage], fit_reports[stage] = model, fitted
        model_path = output / "frozen-models" / f"{stage}.npz"
        save_arrays(model_path, **model)
        hashes[stage] = file_sha256(model_path)
        dev_probabilities[stage] = predict_linear(model, features(dev, stage))
    dev_probabilities["early_operational"] = dev_probabilities["fusion"].copy()
    dev_probabilities["early_operational"][~eligibility(dev)] = dev_probabilities["text"][~eligibility(dev)]
    for item in late_candidates:
        dev_probabilities[f"late_{item['vision_weight']:g}"] = mix_probabilities(dev_probabilities["text"], dev_probabilities["vision"], eligibility(dev), item["vision_weight"])
    baseline = baseline_probabilities(home, dev, plan, baseline_heads)
    dev_metrics = comparisons(dev, dev_probabilities)
    baseline_metrics = comparisons(dev, baseline)
    text_folds = np.asarray(stage_winners["text"]["eligible"]["fold_macro_f1"])
    families = []
    for family, summary, key in (
        ("early", stage_winners["fusion"]["eligible"], "early_operational"),
        ("late", late_positive, f"late_{late_positive['vision_weight']:g}"),
    ):
        gains = np.asarray(summary["fold_macro_f1"]) - text_folds
        families.append({"family": family, "probability_key": key, "cv": summary,
                         "cv_paired_gains_vs_linear_text": gains.tolist(),
                         "qualifies_on_cv": bool(gains.mean() > 0 and (gains > 0).sum() >= 2),
                         "dev_eligible_macro_f1": dev_metrics["eligible"][key]["macro_f1"],
                         "dev_operational_macro_f1": dev_metrics["all"][key]["macro_f1"]})
    qualified = [item for item in families if item["qualifies_on_cv"]]
    chosen = max(qualified, key=lambda item: (item["dev_eligible_macro_f1"], item["dev_operational_macro_f1"])) if qualified else max(families, key=lambda item: item["cv"]["mean_macro_f1"])
    selection = {
        "schema_version": 1, "frozen_at": utc_now(), "plan_sha256": file_sha256(output / "plan.json"),
        "feature_identity": plan["feature_identity"], "cache_sha256": plan["cache_sha256"],
        "fold_assignment_sha256": file_sha256(output / "folds.npz"),
        "linear_winners": stage_winners, "late_candidates_cv": late_candidates,
        "late_best_including_zero": late_best, "late_best_positive": late_positive,
        "families": families, "chosen_multimodal": chosen,
        "supports_multimodal_on_cv": bool(qualified), "full_train_fit_reports": fit_reports,
        "frozen_model_sha256": hashes, "dev": dev_metrics, "baseline_dev": baseline_metrics,
        "test_accessed_by_search": False, "automatic_promotion": False,
        "selection_warnings": ["Cross-validation configuration selection can make selected OOF scores optimistic; this is not nested CV.",
                               "Only three disjoint-dialogue folds; report fold variation, not seed-independent certainty.",
                               "The development subset is small, especially for disgust and fear."] +
                              (["At least one selected fit did not converge; examine iteration limits before any promotion."] if any(not fit_reports[s]["converged"] or not stage_winners[s]["all_fits_converged"] for s in DIMENSIONS) else []),
        "study_seconds_before_test": time.perf_counter() - all_started,
    }
    save_arrays(output / "dev-predictions.npz", ids=dev["ids"], labels=dev["labels"], available=eligibility(dev),
                **{f"linear_{key}": value for key, value in dev_probabilities.items()},
                **{f"baseline_{key}": value for key, value in baseline.items()})
    atomic_json(output / "selection.json", selection)
    print(f"Frozen family: {chosen['family']}; CV supported={bool(qualified)}; dev eligible F1={chosen['dev_eligible_macro_f1']:.4f}", flush=True)
    return selection


def macro_from_confusion(matrix):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0).mean(axis=-1)


def paired_dialogue_interval(truth, candidate, reference, groups, replicates=1000, seed=314159):
    """Paired cluster bootstrap preserves all utterances in a sampled dialogue."""
    unique, inverse = np.unique(groups, return_inverse=True)
    matrices = []
    for prediction in (candidate, reference):
        count = np.bincount(inverse * 49 + truth * 7 + prediction, minlength=len(unique) * 49)
        matrices.append(count.reshape(len(unique), 7, 7))
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        weights = np.bincount(random.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        a = np.tensordot(weights, matrices[0], axes=1)
        b = np.tensordot(weights, matrices[1], axes=1)
        values.append(float(macro_from_confusion(a) - macro_from_confusion(b)))
    return {"paired_delta_macro_f1": float(macro_from_confusion(matrices[0].sum(axis=0)) - macro_from_confusion(matrices[1].sum(axis=0))),
            "percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "dialogue_count": len(unique), "replicates": replicates, "seed": seed,
            "scope": "Conditional paired dialogue bootstrap; reused benchmark and selection bias remain."}


def evaluate_once(home, output, baseline_heads=None):
    plan = read_plan(home, output, baseline_heads)
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["plan_sha256"] != file_sha256(output / "plan.json"):
        raise ValueError("Frozen selection and declared plan differ.")
    models = {}
    for stage, digest in selection["frozen_model_sha256"].items():
        path = output / "frozen-models" / f"{stage}.npz"
        if file_sha256(path) != digest:
            raise ValueError("Frozen model bytes changed after selection.")
        models[stage] = load_arrays(path)
    lock_path = output / "test-evaluation-started.json"
    # Exclusive file creation prevents accidental repeated peeking/tuning cycles.
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump({"started_at": utc_now(), "selection_sha256": file_sha256(selection_path),
                   "benchmark_status": "previously observed official test; NOT a fresh holdout"}, handle, indent=2)
    test = load_cache(home, "test")  # First test access in this study.
    validate_caches(plan, test)
    probabilities = {stage: predict_linear(models[stage], features(test, stage)) for stage in DIMENSIONS}
    available = eligibility(test)
    probabilities["early_operational"] = probabilities["fusion"].copy()
    probabilities["early_operational"][~available] = probabilities["text"][~available]
    chosen = selection["chosen_multimodal"]
    if chosen["family"] == "early":
        probabilities["selected"] = probabilities["early_operational"]
    else:
        probabilities["selected"] = mix_probabilities(probabilities["text"], probabilities["vision"], available,
                                                       selection["late_best_positive"]["vision_weight"])
    baseline = baseline_probabilities(home, test, plan, baseline_heads)
    groups = groups_from_ids(test["ids"])
    intervals = {}
    for population, mask in (("all", np.ones(len(test["ids"]), dtype=bool)), ("eligible", available)):
        intervals[population] = {}
        for reference_name, reference in (("baseline_text", baseline["text"]), ("baseline_operational", baseline["operational"]), ("linear_text", probabilities["text"])):
            intervals[population][reference_name] = paired_dialogue_interval(test["labels"][mask], probabilities["selected"][mask].argmax(axis=1), reference[mask].argmax(axis=1), groups[mask])
    parameters = {stage: int(model["coefficient"].size + model["intercept"].size) for stage, model in models.items()}
    baseline_inventory = baseline_parameter_inventory(home, plan, baseline_heads)
    report = {
        "schema_version": 1, "completed_at": utc_now(),
        "benchmark_status": "Reused official MELD test benchmark, already observed during the original project; not a fresh holdout.",
        "selection_sha256": file_sha256(selection_path), "test_cache_sha256": test["sha256"],
        "feature_identity": test["feature_identity"], "count": len(test["ids"]),
        "visual_eligible_count": int(available.sum()), "selected_family": chosen,
        "candidate": comparisons(test, probabilities), "baseline": comparisons(test, baseline),
        "paired_dialogue_intervals": intervals,
        "parameters": {"linear_affine_heads": parameters, "all_linear_heads_total": sum(parameters.values()),
                       "fixed_late_probability_weight_adds_learned_parameters": 0,
                       **baseline_inventory,
                       "conservative_total_if_all_study_heads_were_additionally_required": baseline_inventory["baseline_complete_total"] + sum(parameters.values()),
                       "inference_standardizer_parameters": 0,
                       "standardizer_explanation": "Training means/scales are algebraically folded into affine weights and are not needed at inference."},
        "automatic_promotion": False,
        "limitations": ["Frozen feature study cannot repair face attribution, missing modalities, pretrained representations, or domain shift.",
                        "Cross-validation reuses folds to choose a small prespecified grid; no nested-CV unbiased estimate is claimed.",
                        "Test intervals are descriptive conditional comparisons on a reused benchmark, not proof of prospective generalization.",
                        "All inference probabilities are uncalibrated.",
                        "No production checkpoint, encoder, preprocessing class, or runtime decision changed."],
    }
    save_arrays(output / "reused-test-predictions.npz", ids=test["ids"], labels=test["labels"], available=available,
                **{f"candidate_{key}": value for key, value in probabilities.items()},
                **{f"baseline_{key}": value for key, value in baseline.items()})
    atomic_json(output / "reused-test-results.json", report)
    read_plan(home, output, baseline_heads)  # Assert baseline remained unchanged throughout.
    return report


def read_frozen_selection(home, output, baseline_heads=None):
    """Consume frozen artifacts without pretending to rerun their source version.

    Preparation supports schema-1 completed studies from an earlier source
    revision. Their plan/source hashes remain intact; preparation records its own
    source hash separately. Search/evaluate still require their exact plan source.
    """
    plan_path, selection_path = output / "plan.json", output / "selection.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or selection.get("schema_version") != 1:
        raise ValueError("Preparation requires a supported schema-1 frozen study and selection.")
    if selection.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("Frozen selection and study plan differ.")
    if selection.get("feature_identity") != plan.get("feature_identity") or selection.get("cache_sha256") != plan.get("cache_sha256"):
        raise ValueError("Frozen selection feature/cache provenance differs from the plan.")
    directory = verify_baseline_heads(home, plan, baseline_heads)
    path = output / "frozen-models" / "text.npz"
    if file_sha256(path) != selection["frozen_model_sha256"]["text"]:
        raise ValueError("Frozen text model bytes changed after selection.")
    linear = load_arrays(path)
    if (set(linear) != {"coefficient", "intercept"}
            or linear["coefficient"].shape != (7, DIMENSIONS["text"])
            or linear["intercept"].shape != (7,)
            or not all(np.isfinite(value).all() for value in linear.values())):
        raise ValueError("Frozen text artifact must contain a finite 7-by-1024 affine classifier.")
    config = selection["linear_winners"]["text"]["config"]
    if config.get("stage") != "text" or config.get("C") not in plan["linear"]["C"] or config.get("weighting") not in plan["linear"]["weighting"]:
        raise ValueError("Selected text configuration is outside the declared study plan.")
    return plan, selection, linear, directory


def prepare_hybrid(home, output, baseline_heads=None, candidate_output=None):
    """Prepare a new text head; never install it, retrain it, or open test data."""
    home, output = Path(home).resolve(), Path(output).resolve()
    destination = Path(candidate_output or output / "deployment").resolve()
    directory = baseline_directory(home, override=baseline_heads)
    if destination == home or home in destination.parents:
        raise ValueError("Candidate output must be outside the production artifact tree.")
    plan, selection, linear, directory = read_frozen_selection(home, output, baseline_heads)
    if destination == directory or directory in destination.parents:
        raise ValueError("Candidate output must be separate from preserved baseline heads.")
    targets = ("text.pt", "hybrid-definition.json", "hybrid-dev-predictions.npz", "preparation-started.json")
    if any((destination / name).exists() for name in targets):
        raise FileExistsError("Candidate preparation artifacts already exist; nothing was overwritten. "
                              "Use --candidate-output with a new empty destination to reproduce preparation.")
    train, dev = load_cache(home, "train"), load_cache(home, "dev")
    validate_caches(plan, train, dev)
    selected = selection["linear_winners"]["text"]
    state = {"net.weight": torch.from_numpy(linear["coefficient"].astype(np.float32)),
             "net.bias": torch.from_numpy(linear["intercept"].astype(np.float32))}
    model = LinearEmotionHead(DIMENSIONS["text"]).eval()
    model.load_state_dict(state)
    parity, predictions = {}, {}
    for cache in (train, dev):
        reference = predict_linear(linear, cache["text"])
        actual = predict_probabilities(model, cache["text"], device="cpu")
        maximum_error = float(np.abs(reference - actual).max())
        label_changes = int((reference.argmax(axis=1) != actual.argmax(axis=1)).sum())
        if maximum_error > 1e-5 or label_changes:
            raise ValueError("Float32 deployment failed the frozen model probability/label parity check.")
        parity[cache["split"]] = {"count": len(cache["ids"]), "maximum_probability_error": maximum_error,
                                  "changed_labels": label_changes}
        predictions[cache["split"]] = actual
    baseline = baseline_probabilities(home, dev, plan, directory)
    available = eligibility(dev)
    hybrid = predictions["dev"].copy()
    hybrid[available] = baseline["fusion"][available]
    dev_metrics = {}
    for population, mask in (("all", np.ones(len(dev["ids"]), dtype=bool)), ("eligible", available), ("unavailable", ~available)):
        dev_metrics[population] = {
            name: quick_metrics(dev["labels"][mask], probability[mask])
            for name, probability in (("baseline_text", baseline["text"]), ("baseline_operational", baseline["operational"]),
                                      ("new_text", predictions["dev"]), ("hybrid", hybrid))}
    config = selected["config"]
    provenance = {
        "study_plan_sha256": file_sha256(output / "plan.json"),
        "model_study_selection_sha256": file_sha256(output / "selection.json"),
        "original_study_source_sha256": plan["study_source_sha256"],
        "preparation_source_sha256": file_sha256(Path(__file__)),
        "source_frozen_affine_sha256": selection["frozen_model_sha256"]["text"],
        "preparation_compatibility": "Schema-1 frozen selection consumed without rerunning or modifying the original search.",
    }
    payload = {
        "schema_version": 2, "architecture": "linear", "stage": "text", "input_dim": DIMENSIONS["text"],
        "state_dict": state, "labels": list(LABELS), "seed": plan["cv"]["seed"],
        "train_count": len(train["ids"]), "dev_count": len(dev["ids"]),
        "cache_sha256": plan["cache_sha256"], "feature_identity": plan["feature_identity"],
        "selected_config": config, "selected_loss": config["weighting"] + "_logistic",
        "best_dev_macro_f1": dev_metrics["all"]["new_text"]["macro_f1"], "best_epoch": None, "history": [],
        "selection_population": "Prespecified training-dialogue CV; development confirmation without refitting",
        "hyperparameters": {"C": config["C"], "weighting": config["weighting"], "solver": plan["linear"]["solver"],
                            "max_iter": plan["linear"]["max_iter"], "tolerance": plan["linear"]["tolerance"]},
        "folded_scaler_provenance": {
            "fit_population": f"Official training text features only, {len(train['ids'])} utterances",
            "formula": "saved_weight = coefficient / training_scale; saved_bias = intercept - saved_weight @ training_mean",
            "runtime_scaler_required": False, "runtime_scaler_parameters": 0,
            "source_weight_dtype": str(linear["coefficient"].dtype), "deployment_weight_dtype": "float32"},
        "created_at": utc_now(), "production_activation": "Prepared only; a separate promotion decision is required.",
        **provenance,
    }
    # Refuse collisions before writing and lock this one preparation against a
    # second process. A failed attempt remains visible rather than being replaced.
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "preparation-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at": utc_now(), **provenance}, handle, indent=2)
    target = destination / "text.pt"
    atomic_checkpoint(target, payload)
    restored, _ = load_head(target, device="cpu")
    if not np.array_equal(predict_probabilities(restored, dev["text"], device="cpu"), predictions["dev"]):
        raise ValueError("Serialized deployment checkpoint did not preserve its development predictions.")
    baseline_counts = {}
    for stage in DIMENSIONS:
        head, _ = load_head(directory / f"{stage}.pt", device="cpu")
        baseline_counts[stage] = sum(parameter.numel() for parameter in head.parameters())
    counts = {**baseline_counts, "text": sum(parameter.numel() for parameter in restored.parameters())}
    pretrained = {"Qwen3-4B-Instruct-2507": 4022468096, "DeBERTa-v3-large": 434012160,
                  "EmotiEffLib enet_b2_7 including original FER head": 7710857,
                  "YuNet conservative initializer allowance": 53121}
    total, old_total = sum(pretrained.values()) + sum(counts.values()), sum(pretrained.values()) + sum(baseline_counts.values())
    report = {
        "schema_version": 1, "frozen_at": utc_now(), **provenance,
        "definition": "Use the frozen baseline fusion head when vision is eligible, otherwise the prepared linear text head; retain the baseline vision head for diagnostics.",
        "rationale": "Use the one frozen text configuration selected by training-dialogue CV; report dev confirmation without trying alternate hybrids.",
        "benchmark_status": "Preparation uses train/dev only. Any later official test evaluation is a reused benchmark, never a fresh holdout.",
        "prepared_text_sha256": file_sha256(target), "prepared_text_path": str(target),
        "baseline_heads": str(directory), "baseline_head_sha256": plan["baseline_head_sha256"],
        "original_study_home": plan["home"], "cache_home": str(home),
        "train_dev_cache_sha256": plan["cache_sha256"], "feature_identity": plan["feature_identity"],
        "float32_deployment_parity": parity, "dev": dev_metrics,
        "parameters": {"pretrained": pretrained, "runtime_heads": counts, "runtime_heads_total": sum(counts.values()),
                       "complete_total": total, "old_complete_total": old_total, "change_vs_old": total - old_total,
                       "remaining_below_six_billion": 6000000000 - total, "folded_scaler_extra_parameters": 0,
                       "total_if_old_text_were_also_required_at_runtime": total + baseline_counts["text"],
                       "basis": "Pinned project pretrained architecture counts plus actual loaded head counts; preparation does not reload large encoders/generator."},
        "test_accessed": False, "production_activated": False,
    }
    verify_baseline_heads(home, plan, directory)
    save_arrays(destination / "hybrid-dev-predictions.npz", ids=dev["ids"], labels=dev["labels"], available=available,
                hybrid=hybrid, new_text=predictions["dev"], baseline_operational=baseline["operational"])
    atomic_json(destination / "hybrid-definition.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-heads", type=Path, help="Preserved original head directory; hashes must match the frozen study baseline.")
    parser.add_argument("--candidate-output", type=Path, help="New candidate directory for prepare; default: OUTPUT/deployment. Never the runtime checkpoint directory.")
    parser.add_argument("phase", choices=("plan", "search", "evaluate", "prepare"))
    args = parser.parse_args(argv)
    home, output = args.home.resolve(), args.output.resolve()
    if output == home or home in output.parents:
        raise ValueError("Study output must be outside the production artifact tree.")
    if args.candidate_output is not None and args.phase != "prepare":
        parser.error("--candidate-output applies only to the prepare phase.")
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase == "prepare":
            result = prepare_hybrid(home, output, args.baseline_heads, args.candidate_output)
        else:
            result = {"plan": make_plan, "search": search, "evaluate": evaluate_once}[args.phase](home, output, args.baseline_heads)
    print(json.dumps({"phase": args.phase, "output": str(output), "completed": True,
                      "created_at": result.get("created_at", result.get("completed_at", result.get("frozen_at")))}, indent=2))


if __name__ == "__main__":
    main()
