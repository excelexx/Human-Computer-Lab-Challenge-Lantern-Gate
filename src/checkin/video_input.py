"""Silent local video input without Gradio's implicit audio-removal re-encode.

H.264/yuv420p is copied to MP4; VP8/VP9 is copied to WebM. Other codecs or
non-browser H.264 pixel formats are explicitly converted to H.264/yuv420p.
That compatibility fallback changes decoded pixels and is reported to the UI.
Neither branch changes the shared model preprocessing or uses audio features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid

import gradio as gr
from gradio.data_classes import FileData
from gradio.components.video import VideoData


MAX_VIDEO_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class SilentClip:
    path: str
    mode: str
    duration: float


def _run(command: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Video preparation needs working FFmpeg/ffprobe and a readable short clip.") from exc
    if result.returncode:
        raise ValueError("The clip could not be prepared. Try another short video file.")
    return result.stdout


def probe_video(path: Path, max_length: float = 20, *, allow_missing_webm_duration: bool = False) -> dict:
    """Validate media, optionally admitting an unfinished WebM duration for remux.

    Browser MediaRecorder can omit WebM's Duration element. Such a file still
    has timestamped packets, but must never be served or accepted as bounded
    until a full stream-copy remux has written and verified its duration.
    """
    if not math.isfinite(max_length) or max_length <= 0:
        raise ValueError("The video length limit must be finite and positive.")
    if not path.is_file():
        raise ValueError("The selected clip is no longer available. Please select it again.")
    if not 0 < path.stat().st_size <= MAX_VIDEO_BYTES:
        raise ValueError("Choose a nonempty video file no larger than 100 MiB.")
    try:
        info = json.loads(_run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
        video = next(stream for stream in info["streams"] if stream.get("codec_type") == "video")
        raw_duration = info.get("format", {}).get("duration") or video.get("duration")
        missing_webm_duration = (
            allow_missing_webm_duration
            and raw_duration in {None, "N/A"}
            and "webm" in info.get("format", {}).get("format_name", "").split(",")
            and video.get("codec_name") in {"vp8", "vp9"}
        )
        duration = None if missing_webm_duration else float(raw_duration or "nan")
    except (ValueError, KeyError, StopIteration, TypeError) as exc:
        raise ValueError("Choose a video with a readable, finite duration.") from exc
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Choose a video with a readable, finite duration.")
    if duration is not None and duration > max_length:
        raise ValueError(f"Video is too long; choose a clip of at most {max_length:g} seconds.")
    return {"streams": info["streams"], "video": video, "duration": duration}


def prepare_silent_video(source: str | Path, cache_dir: str | Path, max_length: float = 20) -> SilentClip:
    """Return a silent browser-compatible local clip, preserving pixels when possible.

    Cache identities use the file bytes, not its name, so replacing an upload
    cannot accidentally reuse a previous clip. Temporary outputs are private to
    this call; a completed cache entry is published atomically.
    """
    source = Path(source).resolve()
    info = probe_video(source, max_length, allow_missing_webm_duration=True)
    codec = info["video"].get("codec_name")
    pixels = info["video"].get("pix_fmt")
    copy = codec in {"vp8", "vp9"} or (codec == "h264" and pixels in {"yuv420p", "yuvj420p"})
    suffix = ".webm" if codec in {"vp8", "vp9"} else ".mp4"
    audio = any(stream.get("codec_type") == "audio" for stream in info["streams"])
    if copy and not audio and source.suffix.lower() == suffix and info["duration"] is not None:
        return SilentClip(str(source), "unchanged", info["duration"])
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    cache = Path(cache_dir).resolve() / "checkin-silent-v1" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("clip" + suffix)
    mode = "stream_copy" if copy else "transcoded"
    if target.is_file():
        prepared = probe_video(target, max_length)
        if not any(stream.get("codec_type") == "audio" for stream in prepared["streams"]):
            return SilentClip(str(target), mode, prepared["duration"])
    temporary = cache / (uuid.uuid4().hex + suffix)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn"]
    # Do not use -t to hide an unknown overlong duration by truncation. The
    # original 100 MiB limit and 30-second process deadline bound the remux;
    # strict probing below rejects the full result if it exceeds max_length.
    command += ["-c:v", "copy"] if copy else ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if suffix == ".mp4":
        command += ["-movflags", "+faststart"]
    try:
        _run(command + [str(temporary)])
        prepared = probe_video(temporary, max_length)
        if any(stream.get("codec_type") == "audio" for stream in prepared["streams"]):
            raise ValueError("Audio removal failed. Please choose another clip.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return SilentClip(str(target), mode, prepared["duration"])


class SilentVideo(gr.Video):
    """Use Gradio's normal Video frontend with deterministic silent preparation."""

    is_template = True

    def _silent(self, path: str | Path) -> str:
        try:
            result = prepare_silent_video(path, self.GRADIO_CACHE, self.max_length or 20)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        if result.mode == "transcoded":
            gr.Warning("This clip needed a video format conversion for the browser. Its image pixels may differ from the original; use an H.264 MP4 or VP8/VP9 WebM for an unchanged video stream.")
        if Path(result.path).is_relative_to(Path(self.GRADIO_CACHE).resolve()):
            self.temp_files.add(result.path)
        return result.path

    def preprocess(self, payload: VideoData | None) -> str | None:
        return self._silent(payload.video.path) if payload is not None else None

    def _format_video(self, video: str | Path | None) -> FileData | None:
        # Replay values travel through postprocess before they are submitted.
        # Preparing them here also avoids serving their original audio track.
        if video is None:
            return None
        path = self._silent(video)
        return FileData(path=path, orig_name=Path(path).name)
