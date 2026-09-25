import numpy as np
import pytest
from checkin.backtest_statistics import calibration, clustered_difference, scores_from_confusion


def test_perfect_seven_class_predictions_have_zero_calibration_error():
    result = calibration(np.arange(7), np.eye(7))
    assert result["top_label_ece_equal_width"] == 0
    assert result["multiclass_brier_sum"] == 0
    assert result["bins"][-1]["n"] == 7
    assert scores_from_confusion(np.eye(7))["macro_f1"] == 1


def test_paired_identical_models_have_zero_difference_in_every_bootstrap():
    truth = np.tile(np.arange(7), 3)
    groups = np.repeat(np.arange(7), 3)
    result = clustered_difference(truth, truth, truth, groups, draws=100)
    assert result["dialogues"] == 7
    assert all(item["observed"] == 0 and item["percentile_95_interval"] == [0, 0]
               for item in result["differences"].values())


def test_overconfident_wrong_predictions_are_not_reported_as_calibrated():
    result = calibration([0], np.eye(7)[[1]])
    assert result["top_label_ece_equal_width"] == 1
    assert result["multiclass_brier_sum"] == 2
    with pytest.raises(ValueError):
        calibration([0], np.ones((1, 7)))


@pytest.mark.parametrize("labels", [[-1], [7], [.5], [[0]], [True]])
def test_labels_cannot_silently_index_another_class(labels):
    with pytest.raises(ValueError):
        calibration(labels, np.eye(7)[[0]])
    with pytest.raises(ValueError):
        clustered_difference(labels, [0], [0], ["g"], draws=100)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_calibration_requires_meaningful_bin_count(bins):
    with pytest.raises(ValueError):
        calibration([0], np.eye(7)[[1]], bins=bins)


def test_all_fallback_report_has_explicit_empty_visual_cohort(tmp_path):
    import json
    from checkin.backtest_statistics import run
    row = {"id": "dev:0:0", "true_label_id": 0, "operational_probabilities": [1,0,0,0,0,0,0],
           "text_probabilities": [1,0,0,0,0,0,0], "vision_available": False, "modality_disagreement": False}
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(row), encoding="utf-8")
    report = run(source, tmp_path / "report.json", draws=100)
    assert report["results"]["common_visual"]["n"] == 0
    assert report["split"] == "dev" and "no test access" in report["test_status"]
    source.write_text(json.dumps(row) + "\n" + json.dumps({**row, "id": "test:0:0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="one official split"):
        run(source, tmp_path / "report.json", draws=100)
