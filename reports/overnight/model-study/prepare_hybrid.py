"""Prepare one declared hybrid; never activate it or change baseline artifacts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from checkin.evaluate import classification_metrics
from checkin.model_study import (baseline_probabilities, groups_from_ids, load_arrays,
                                 paired_dialogue_interval, predict_linear, read_plan, save_arrays)
from checkin.models import DIMENSIONS, LinearEmotionHead, load_head
from checkin.settings import LABELS
from checkin.train import atomic_checkpoint, atomic_json, file_sha256, load_cache, predict_probabilities


HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "plan.json").read_text(encoding="utf-8"))
HOME = Path(PLAN["home"])


def score(labels, probability):
    return classification_metrics(labels, probability.argmax(axis=1))


def prepare():
    if (HERE / "hybrid-definition.json").exists():
        raise FileExistsError("Hybrid definition is already frozen.")
    read_plan(HOME, HERE)
    selection = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    frozen_path = HERE / "frozen-models/text.npz"
    assert file_sha256(frozen_path) == selection["frozen_model_sha256"]["text"]
    linear = load_arrays(frozen_path)
    state = {"net.weight": torch.from_numpy(linear["coefficient"].astype(np.float32)),
             "net.bias": torch.from_numpy(linear["intercept"].astype(np.float32))}
    train, dev = load_cache(HOME, "train"), load_cache(HOME, "dev")
    for cache in (train, dev):
        assert cache["sha256"] == PLAN["cache_sha256"][cache["split"]]
        assert cache["feature_identity"] == PLAN["feature_identity"]
    selected = selection["linear_winners"]["text"]
    payload = {
        "schema_version": 2, "architecture": "linear", "stage": "text", "input_dim": 1024,
        "state_dict": state, "labels": list(LABELS), "seed": 42,
        "train_count": len(train["ids"]), "dev_count": len(dev["ids"]),
        "cache_sha256": PLAN["cache_sha256"], "feature_identity": PLAN["feature_identity"],
        "selected_config": selected["config"], "selected_loss": "sqrt_balanced_logistic",
        "best_dev_macro_f1": selection["dev"]["all"]["text"]["macro_f1"],
        "best_epoch": None, "history": [],
        "selection_population": "Prespecified three-fold training-dialogue CV; full development confirmation",
        "hyperparameters": {"C": selected["config"]["C"], "weighting": selected["config"]["weighting"],
                            "solver": "lbfgs", "max_iter": 500, "tolerance": 1e-4},
        "folded_scaler_provenance": {
            "fit_population": "Official training text features only, 9989 utterances",
            "formula": "saved_weight = coefficient / training_scale; saved_bias = intercept - saved_weight @ training_mean",
            "runtime_scaler_required": False, "runtime_scaler_parameters": 0,
            "source_weight_dtype": "float64", "deployment_weight_dtype": "float32",
        },
        "model_study_selection_sha256": file_sha256(HERE / "selection.json"),
        "source_frozen_affine_sha256": file_sha256(frozen_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_activation": "Prepared only; requires an explicit separate promotion decision.",
    }
    target = HERE / "deployment/text.pt"
    if target.exists():
        raise FileExistsError(target)
    atomic_checkpoint(target, payload)
    model, restored = load_head(target, device="cpu")
    parity = {}
    for cache in (train, dev):
        reference = predict_linear(linear, cache["text"])
        actual = predict_probabilities(model, cache["text"], device="cpu")
        maximum_error = float(np.abs(reference - actual).max())
        label_changes = int((reference.argmax(axis=1) != actual.argmax(axis=1)).sum())
        if maximum_error > 1e-5 or label_changes:
            raise AssertionError(f"Float32 deployment parity failed: {maximum_error=}, {label_changes=}")
        parity[cache["split"]] = {"count": len(cache["ids"]), "maximum_probability_error": maximum_error,
                                  "changed_labels": label_changes}
    baseline = baseline_probabilities(HOME, dev, PLAN)
    probability = predict_probabilities(model, dev["text"], device="cpu")
    available = dev["quality"][:, 2] > 0
    hybrid = probability.copy()
    hybrid[available] = baseline["fusion"][available]
    dev_metrics = {
        "all": {"baseline_text": score(dev["labels"], baseline["text"]),
                "baseline_operational": score(dev["labels"], baseline["operational"]),
                "new_text": score(dev["labels"], probability), "hybrid": score(dev["labels"], hybrid)},
        "eligible": {"baseline_operational": score(dev["labels"][available], baseline["operational"][available]),
                     "hybrid": score(dev["labels"][available], hybrid[available])},
        "unavailable": {"baseline_text": score(dev["labels"][~available], baseline["text"][~available]),
                        "new_text": score(dev["labels"][~available], probability[~available])},
    }
    head_counts = {}
    for stage in DIMENSIONS:
        path = target if stage == "text" else HOME / "checkpoints" / f"{stage}.pt"
        head, _ = load_head(path, device="cpu")
        head_counts[stage] = sum(parameter.numel() for parameter in head.parameters())
    pretrained = {"Qwen3-4B-Instruct-2507": 4022468096, "DeBERTa-v3-large": 434012160,
                  "EmotiEffLib enet_b2_7 including original FER head": 7710857,
                  "YuNet conservative initializer allowance": 53121}
    total = sum(pretrained.values()) + sum(head_counts.values())
    definition = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "definition": "Use the original production fusion head whenever vision is eligible; otherwise use the one frozen new linear text head. Original vision head remains unchanged for diagnostics.",
        "rationale_before_this_hybrid_benchmark": "The text configuration was selected by training-dialogue CV and improved full development macro F1. The original visual/fusion heads are retained because the all-linear multimodal study did not establish visual added value.",
        "selection_warning": "This hybrid was requested after the first reused-test study was observed. It is a single fixed follow-up, not a fresh holdout and not selected among additional hybrid configurations.",
        "prepared_text_sha256": file_sha256(target), "prepared_text_path": str(target),
        "baseline_head_sha256": PLAN["baseline_head_sha256"],
        "train_dev_cache_sha256": PLAN["cache_sha256"], "feature_identity": PLAN["feature_identity"],
        "loader_source_sha256": file_sha256(Path(__import__("checkin.models", fromlist=["__file__"]).__file__)),
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "float32_deployment_parity": parity, "dev": dev_metrics,
        "parameters": {"pretrained": pretrained, "runtime_heads": head_counts,
                       "runtime_heads_total": sum(head_counts.values()), "complete_total": total,
                       "old_complete_total": 4464870303, "change_vs_old": total - 4464870303,
                       "remaining_below_six_billion": 6000000000 - total,
                       "folded_scaler_extra_parameters": 0,
                       "total_if_old_text_were_also_required_at_runtime": total + 132103},
        "test_protocol": "One subsequent pass using this prepared artifact on the reused official test cache. Paired dialogue bootstrap, 1000 replicates/seed314159, versus original operational model and original text. No test-based fitting or alternate-hybrid selection.",
        "production_activated": False,
    }
    save_arrays(HERE / "hybrid-dev-predictions.npz", ids=dev["ids"], labels=dev["labels"], available=available,
                hybrid=hybrid, new_text=probability, baseline_operational=baseline["operational"])
    atomic_json(HERE / "hybrid-definition.json", definition)
    print(json.dumps({"prepared": str(target), "sha256": definition["prepared_text_sha256"],
                      "hybrid_dev_macro_f1": dev_metrics["all"]["hybrid"]["macro_f1"],
                      "old_dev_macro_f1": dev_metrics["all"]["baseline_operational"]["macro_f1"],
                      "parameters": total, "activated": False}, indent=2))


def evaluate():
    read_plan(HOME, HERE)
    path = HERE / "hybrid-definition.json"
    definition = json.loads(path.read_text(encoding="utf-8"))
    target = HERE / "deployment/text.pt"
    if file_sha256(target) != definition["prepared_text_sha256"]:
        raise ValueError("Prepared model changed after hybrid freeze.")
    with (HERE / "hybrid-test-evaluation-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"definition_sha256": file_sha256(path), "started_at": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
    cache = load_cache(HOME, "test")
    if cache["feature_identity"] != definition["feature_identity"]:
        raise ValueError("Wrong feature identity.")
    model, _ = load_head(target, device="cpu")
    new_text = predict_probabilities(model, cache["text"], device="cpu")
    baseline = baseline_probabilities(HOME, cache, PLAN)
    available = cache["quality"][:, 2] > 0
    hybrid = new_text.copy()
    hybrid[available] = baseline["fusion"][available]
    groups = groups_from_ids(cache["ids"])
    metrics, intervals = {}, {}
    for population, mask in (("all", np.ones(len(cache["ids"]), dtype=bool)), ("eligible", available), ("unavailable", ~available)):
        metrics[population] = {name: score(cache["labels"][mask], probability[mask])
                               for name, probability in (("hybrid", hybrid), ("baseline_operational", baseline["operational"]),
                                                         ("baseline_text", baseline["text"]), ("new_text", new_text))}
        intervals[population] = {name: paired_dialogue_interval(cache["labels"][mask], hybrid[mask].argmax(axis=1),
                                                                 reference[mask].argmax(axis=1), groups[mask])
                                  for name, reference in (("baseline_operational", baseline["operational"]), ("baseline_text", baseline["text"]))}
    original = load_arrays(HERE / "frozen-models/text.npz")
    float64 = predict_linear(original, cache["text"])
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(), "hybrid_definition_sha256": file_sha256(path),
        "prepared_text_sha256": file_sha256(target), "test_cache_sha256": cache["sha256"],
        "benchmark_status": "Reused official test benchmark, already observed before this fixed follow-up; not fresh holdout evidence.",
        "metrics": metrics, "paired_dialogue_intervals": intervals,
        "float32_test_parity": {"maximum_probability_error": float(np.abs(new_text - float64).max()),
                                "changed_labels": int((new_text.argmax(axis=1) != float64.argmax(axis=1)).sum())},
        "eligible_hybrid_equals_original_fusion": bool(np.array_equal(hybrid[available], baseline["fusion"][available])),
        "parameters": definition["parameters"], "production_activated": False,
    }
    save_arrays(HERE / "hybrid-reused-test-predictions.npz", ids=cache["ids"], labels=cache["labels"], available=available,
                hybrid=hybrid, new_text=new_text, baseline_operational=baseline["operational"], baseline_text=baseline["text"])
    atomic_json(HERE / "hybrid-reused-test-results.json", report)
    read_plan(HOME, HERE)
    print(json.dumps({"test_macro_f1": {name: values["macro_f1"] for name, values in metrics["all"].items()},
                      "interval": intervals["all"]["baseline_operational"], "activated": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "evaluate"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        {"prepare": prepare, "evaluate": evaluate}[args.phase]()
