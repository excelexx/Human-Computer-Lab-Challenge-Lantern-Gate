"""Real CPU media regressions for browser WebM without a Duration element."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from checkin.video_input import prepare_silent_video, probe_video


def run(*args):
    return subprocess.run(list(map(str, args)), check=True, capture_output=True,
                          text=True, timeout=30).stdout


def frame_hashes(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    result = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result.append(hashlib.sha256(frame.tobytes()).hexdigest())
    finally:
        capture.release()
    assert result
    return result


def live_webm(tmp_path, codec="libvpx", audio=False):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the media regression")
    path = tmp_path / "browser live recording.webm"
    args = ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
            "testsrc2=size=128x96:rate=12:duration=1.5"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1.5"]
    args += ["-c:v", codec, "-threads", "1"]
    args += ["-c:a", "libopus", "-shortest"] if audio else ["-an"]
    run(*args, "-f", "webm", "-live", "1", path)
    metadata = json.loads(run("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path))
    assert "duration" not in metadata["format"]
    assert "duration" not in metadata["streams"][0]
    return path


@pytest.mark.parametrize("codec", ["libvpx", "libvpx-vp9"])
@pytest.mark.parametrize("audio", [False, True])
def test_missing_browser_webm_duration_is_normalized_without_changing_frames(tmp_path, codec, audio):
    source = live_webm(tmp_path, codec, audio)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="finite duration"):
        probe_video(source)  # Strict public validation never accepts unknown duration.
    result = prepare_silent_video(source, tmp_path / "cache")
    assert result.mode == "stream_copy"
    assert Path(result.path) != source
    assert 1.4 < result.duration < 1.7
    normalized = probe_video(Path(result.path))
    assert normalized["duration"] == result.duration
    assert not any(stream["codec_type"] == "audio" for stream in normalized["streams"])
    assert len(frame_hashes(source)) == 18
    assert frame_hashes(source) == frame_hashes(result.path)
    assert source.read_bytes() == before
    assert prepare_silent_video(source, tmp_path / "cache").path == result.path
    assert prepare_silent_video(result.path, tmp_path / "cache").mode == "unchanged"


def test_unknown_duration_does_not_bypass_limit_or_publish_truncated_video(tmp_path):
    source = live_webm(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="too long"):
        prepare_silent_video(source, tmp_path / "cache", max_length=1)
    assert source.read_bytes() == before
    assert not list((tmp_path / "cache").rglob("*.webm"))


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "-1"])
def test_explicit_invalid_webm_duration_is_not_treated_as_missing(tmp_path, monkeypatch, duration):
    import checkin.video_input as module
    source = tmp_path / "invalid.webm"
    source.write_bytes(b"not accepted")
    monkeypatch.setattr(module, "_run", lambda *args, **kwargs: json.dumps({
        "format": {"format_name": "matroska,webm", "duration": duration},
        "streams": [{"codec_type": "video", "codec_name": "vp8"}]}))
    with pytest.raises(ValueError, match="finite duration"):
        prepare_silent_video(source, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
