"""Dataset and learned-head contracts, with no checkpoints or GPU required."""

import json

import numpy as np
import pytest
import torch

from checkin.data import VideoProcessor, box_iou, l2_normalize, read_manifest
from checkin.models import DIMENSIONS, EmotionHead, load_head
from checkin.settings import LABELS, LABEL_TO_ID


def test_meld_label_mapping_is_complete_and_invertible():
    assert len(LABELS) == 7
    assert set(LABELS) == {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
    assert [LABEL_TO_ID[label] for label in LABELS] == list(range(7))


def test_missing_media_is_explicit_without_running_a_detector(tmp_path):
    processor = VideoProcessor.__new__(VideoProcessor)
    for missing in (None, str(tmp_path / "absent.mp4")):
        faces, quality = processor.process(missing)
        assert faces == []
        assert quality["available"] is False
        assert quality["reason"] == "missing_video"
        assert quality["selected_frames"] == 0
        assert "emotion" not in quality


def test_no_faces_are_unavailable_not_neutral(tmp_path, monkeypatch):
    import checkin.data as data

    class NoFaceDetector:
        def setInputSize(self, size):
            pass

        def detect(self, frame):
            return None, None

    class CameraFixture:
        released = False

        def get(self, prop):
            return 24 if prop == data.cv2.CAP_PROP_FPS else 48

        def isOpened(self):
            return True

        def set(self, prop, value):
            pass

        def read(self):
            return True, np.zeros((120, 160, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    fixture_capture = CameraFixture()
    monkeypatch.setattr(data.cv2, "VideoCapture", lambda _: fixture_capture)
    path = tmp_path / "fixture.mp4"
    path.touch()
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.frames = 8
    processor.min_valid = 6
    processor.detector = NoFaceDetector()
    faces, quality = processor.process(path)
    assert faces == []
    assert quality["available"] is False
    assert quality["reason"] == "insufficient_face_frames"
    assert quality["sampled_frames"] == 8
    assert quality["selected_frames"] == 0
    assert fixture_capture.released


def test_manifest_filter_preserves_official_split_identity(tmp_path):
    (tmp_path / "manifests").mkdir()
    records = [{"id": f"{split}:0:0", "split": split, "dialogue_id": 0, "utterance_id": 0} for split in ("train", "dev", "test")]
    (tmp_path / "manifests/meld.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    assert len(read_manifest(tmp_path)) == 3
    assert read_manifest(tmp_path, "test") == [records[2]]


@pytest.mark.parametrize("kind", ["vision", "text", "fusion"])
def test_head_parameters_count_all_learned_values_and_output_seven_labels(kind):
    dimension = DIMENSIONS[kind]
    head = EmotionHead(dimension).eval()
    expected = (dimension + 1) * 128 + (128 + 1) * 7
    assert sum(parameter.numel() for parameter in head.parameters()) == expected
    head.requires_grad_(False)
    assert sum(parameter.numel() for parameter in head.parameters()) == expected
    logits = head(torch.zeros(2, dimension))
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_head_checkpoint_roundtrip_preserves_predictions(tmp_path):
    head = EmotionHead(DIMENSIONS["vision"]).eval()
    sample = torch.ones(1, DIMENSIONS["vision"])
    path = tmp_path / "head.pt"
    torch.save({"input_dim": DIMENSIONS["vision"], "state_dict": head.state_dict(), "labels": LABELS}, path)
    restored, payload = load_head(path)
    assert payload["labels"] == LABELS
    torch.testing.assert_close(restored(sample), head(sample))


def test_zero_features_remain_finite_and_normalized_features_have_unit_norm():
    result = l2_normalize([[0, 0, 0], [3, 4, 0]])
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0], np.zeros(3))
    assert np.linalg.norm(result[1]) == pytest.approx(1)


def test_box_iou_is_symmetric_and_distinguishes_disjoint_tracks():
    first, second = [0, 0, 10, 10], [5, 5, 10, 10]
    assert box_iou(first, first) == pytest.approx(1)
    assert box_iou(first, second) == pytest.approx(box_iou(second, first))
    assert box_iou(first, [20, 20, 1, 1]) == 0
