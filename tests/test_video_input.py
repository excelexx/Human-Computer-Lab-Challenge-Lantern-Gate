"""CPU/FFmpeg regressions for browser media fidelity; no model downloads."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from checkin.video_input import SilentVideo, prepare_silent_video, probe_video


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *map(str, args)], check=True,
                   capture_output=True, timeout=30)


@pytest.fixture
def movie(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the media regression")
    path = tmp_path / "uploaded clip & (day).mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=1.5",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path)
    return path


def decoded_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    hashes = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    finally:
        cap.release()
    assert hashes, "The real decoder must read at least one frame"
    return hashes


def test_upload_audio_is_removed_without_changing_decoded_pixels(movie, tmp_path):
    before = movie.read_bytes()
    result = prepare_silent_video(movie, tmp_path / "cache")
    assert result.mode == "stream_copy"
    assert decoded_frames(movie) == decoded_frames(result.path)
    assert not any(s["codec_type"] == "audio" for s in probe_video(Path(result.path))["streams"])
    assert movie.read_bytes() == before
    assert prepare_silent_video(movie, tmp_path / "cache").path == result.path
    assert prepare_silent_video(result.path, tmp_path / "cache").mode == "unchanged"


def test_replay_postprocess_and_upload_preprocess_share_lossless_path(movie, tmp_path):
    from gradio.data_classes import FileData
    from gradio.components.video import VideoData
    component = SilentVideo(include_audio=False, max_length=20, sources=["webcam", "upload"])
    component.GRADIO_CACHE = str(tmp_path / "gradio-cache")
    upload = component.preprocess(VideoData(video=FileData(path=str(movie))))
    replay = component.postprocess(str(movie))
    submitted_replay = component.preprocess(replay)
    assert component.get_block_name() == "video"
    assert upload == submitted_replay
    assert decoded_frames(movie) == decoded_frames(upload)
    assert upload in component.temp_files
    assert str(movie) not in component.temp_files
    assert component.preprocess(None) is None
    assert component.postprocess(None) is None


def test_replaced_upload_with_same_filename_never_reuses_stale_cache(movie, tmp_path):
    old = prepare_silent_video(movie, tmp_path / "cache")
    movie.unlink()
    ffmpeg("-f", "lavfi", "-i", "color=c=green:size=128x96:rate=12:duration=1.5",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", movie)
    fresh = prepare_silent_video(movie, tmp_path / "cache")
    assert fresh.path != old.path
    assert decoded_frames(fresh.path) == decoded_frames(movie)
    assert decoded_frames(fresh.path) != decoded_frames(old.path)


def test_unsupported_codec_has_explicit_conversion_mode(movie, tmp_path):
    avi = tmp_path / "old codec.avi"
    ffmpeg("-i", movie, "-an", "-c:v", "mpeg4", avi)
    result = prepare_silent_video(avi, tmp_path / "cache")
    assert result.mode == "transcoded"
    info = probe_video(Path(result.path))
    assert info["video"]["codec_name"] == "h264"
    assert info["video"]["pix_fmt"] == "yuv420p"
    assert not any(s["codec_type"] == "audio" for s in info["streams"])


@pytest.mark.parametrize("codec", ["libvpx", "libvpx-vp9"])
def test_browser_webm_codecs_are_copied_without_reencoding(movie, tmp_path, codec):
    source = tmp_path / "browser recording.webm"
    ffmpeg("-i", movie, "-c:v", codec, "-c:a", "libopus", source)
    result = prepare_silent_video(source, tmp_path / "cache")
    assert result.mode == "stream_copy"
    assert Path(result.path).suffix == ".webm"
    assert decoded_frames(source) == decoded_frames(result.path)
    assert not any(s["codec_type"] == "audio" for s in probe_video(Path(result.path))["streams"])


def test_h264_in_other_container_is_remuxed_without_changing_pixels(movie, tmp_path):
    source = tmp_path / "camera recording.mov"
    ffmpeg("-i", movie, "-c", "copy", source)
    result = prepare_silent_video(source, tmp_path / "cache")
    assert result.mode == "stream_copy"
    assert Path(result.path).suffix == ".mp4"
    assert decoded_frames(source) == decoded_frames(result.path)


def test_compatibility_conversion_warns_user(movie, tmp_path, monkeypatch):
    import checkin.video_input as module
    source = tmp_path / "unsupported.avi"
    ffmpeg("-i", movie, "-an", "-c:v", "mpeg4", source)
    messages = []
    monkeypatch.setattr(module.gr, "Warning", messages.append)
    component = SilentVideo(include_audio=False, max_length=20)
    component.GRADIO_CACHE = str(tmp_path / "cache")
    component.postprocess(source)
    assert len(messages) == 1
    assert "image pixels may differ" in messages[0]


def test_overlength_upload_is_rejected_before_conversion(movie, tmp_path):
    with pytest.raises(ValueError, match="too long"):
        prepare_silent_video(movie, tmp_path / "cache", max_length=1)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "-1", "N/A", None])
def test_invalid_duration_is_not_allowed_to_bypass_length_limit(tmp_path, monkeypatch, duration):
    import checkin.video_input as module
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"test")
    monkeypatch.setattr(module, "_run", lambda *a, **kw: json.dumps({
        "streams": [{"codec_type": "video", "codec_name": "h264"}], "format": {"duration": duration}}))
    with pytest.raises(ValueError, match="finite duration"):
        prepare_silent_video(path, tmp_path / "cache")


def test_missing_and_oversized_uploads_fail_locally(tmp_path):
    with pytest.raises(ValueError, match="no longer available"):
        prepare_silent_video(tmp_path / "missing.mp4", tmp_path / "cache")
    path = tmp_path / "oversized.mp4"
    with path.open("wb") as file:
        file.truncate(100 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="100 MiB"):
        prepare_silent_video(path, tmp_path / "cache")


def test_real_meld_frames_survive_browser_audio_removal(tmp_path):
    home = Path(os.environ.get("CHECKIN_VIDEO_TEST_HOME", Path(__file__).resolve().parents[3] / "work/runtime"))
    manifest = home / "manifests/meld.jsonl"
    if not manifest.is_file() or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Optional real MELD regression needs the prepared local runtime and FFmpeg")
    rows = (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines())
    row = next(r for r in rows if r["split"] == "test" and r["dialogue_id"] == 4 and r["utterance_id"] == 2)
    source = Path(row["video_path"])
    if not source.is_absolute():
        source = home / source
    source_info = probe_video(source)
    assert any(s["codec_type"] == "audio" for s in source_info["streams"])
    result = prepare_silent_video(source, tmp_path / "cache")
    before, after = decoded_frames(source), decoded_frames(result.path)
    prepared_info = probe_video(Path(result.path))
    assert result.mode == "stream_copy"
    assert before == after
    assert source_info["video"]["r_frame_rate"] == prepared_info["video"]["r_frame_rate"]
    assert not any(s["codec_type"] == "audio" for s in prepared_info["streams"])
    report = {"input_id": "test:4:2", "source": str(source), "mode": result.mode,
              "source_duration_seconds": source_info["duration"], "muted_duration_seconds": prepared_info["duration"],
              "decoded_frame_count": len(before), "all_decoded_frames_identical": before == after,
              "audio_absent": True, "decoder": "OpenCV BGR frame SHA256; every decoded frame compared",
              "scope": "CPU media preparation only; no model, webcam, or browser latency claim"}
    if path := os.environ.get("CHECKIN_VIDEO_TEST_REPORT"):
        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
