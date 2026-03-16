from __future__ import annotations

import bisect
import random
from pathlib import Path

from .models import TimelineClip

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
BEAT_SNAP_TOLERANCE_SECONDS = 0.15
MIN_CLIP_DURATION_SECONDS = 0.35


def resolve_music_track(music_path: Path | None) -> Path | None:
    if music_path is None or not music_path.exists():
        return None
    if music_path.is_file() and music_path.suffix.lower() in AUDIO_EXTENSIONS:
        return music_path
    if not music_path.is_dir():
        return None

    tracks = [
        path
        for path in sorted(music_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not tracks:
        return None
    return random.choice(tracks)


def detect_music_beats(music_path: Path, log: callable | None = None) -> list[float]:
    try:
        import librosa
    except Exception:
        if log:
            log("Beat sync fallback: librosa not installed, skipping music beat analysis.")
        return []

    try:
        signal, sample_rate = librosa.load(str(music_path), sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()
    except Exception as exc:
        if log:
            log(f"Beat sync fallback: music analysis failed ({exc}).")
        return []

    beats = sorted(float(t) for t in beat_times if float(t) > 0.0)
    if log:
        log(f"Beat sync source: music ({music_path.name}) | beats={len(beats)} | tempo={float(tempo):.1f} BPM")
    return beats


def detect_voiceover_pauses(voiceover_path: Path, log: callable | None = None) -> list[float]:
    try:
        from pydub import AudioSegment, silence
    except Exception:
        if log:
            log("Beat sync fallback: pydub not available for pause detection.")
        return []

    try:
        audio = AudioSegment.from_file(str(voiceover_path))
    except Exception as exc:
        if log:
            log(f"Beat sync fallback: could not read voiceover for pause detection ({exc}).")
        return []

    silence_threshold = audio.dBFS - 18 if audio.dBFS != float("-inf") else -42
    silence_ranges = silence.detect_silence(
        audio,
        min_silence_len=220,
        silence_thresh=silence_threshold,
        seek_step=10,
    )
    pause_points = [((start_ms + end_ms) / 2000.0) for start_ms, end_ms in silence_ranges if (end_ms - start_ms) >= 220]
    pauses = sorted(float(t) for t in pause_points if float(t) > 0.0)
    if log:
        log(f"Beat sync source: voiceover pauses | pauses={len(pauses)}")
    return pauses


def _nearest_point(points: list[float], target: float) -> float | None:
    if not points:
        return None
    idx = bisect.bisect_left(points, target)
    candidates: list[float] = []
    if idx < len(points):
        candidates.append(points[idx])
    if idx > 0:
        candidates.append(points[idx - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item - target))


def snap_timeline_to_rhythm(
    timeline: list[TimelineClip],
    rhythm_points: list[float],
    tolerance_seconds: float = BEAT_SNAP_TOLERANCE_SECONDS,
    min_clip_duration_seconds: float = MIN_CLIP_DURATION_SECONDS,
    log: callable | None = None,
) -> list[TimelineClip]:
    if len(timeline) < 2 or not rhythm_points:
        return timeline

    total_duration = float(timeline[-1].timeline_end)
    boundaries = [float(clip.timeline_end) for clip in timeline[:-1]]
    snapped = list(boundaries)
    snap_count = 0

    for idx, boundary in enumerate(boundaries):
        nearest = _nearest_point(rhythm_points, boundary)
        if nearest is None or abs(nearest - boundary) > tolerance_seconds:
            continue

        prev_boundary = 0.0 if idx == 0 else snapped[idx - 1]
        next_boundary = total_duration if idx == len(boundaries) - 1 else snapped[idx + 1]
        if nearest <= prev_boundary + min_clip_duration_seconds:
            continue
        if nearest >= next_boundary - min_clip_duration_seconds:
            continue

        snapped[idx] = float(nearest)
        snap_count += 1

    if snap_count == 0:
        if log:
            log("Beat sync: no cut points were close enough to snap.")
        return timeline

    rebuilt: list[TimelineClip] = []
    cursor = 0.0
    for idx, clip in enumerate(timeline):
        end = total_duration if idx == len(timeline) - 1 else snapped[idx]
        rebuilt.append(
            TimelineClip(
                source_path=clip.source_path,
                timeline_start=cursor,
                timeline_end=end,
                transition_after=clip.transition_after,
                transition_seconds=clip.transition_seconds,
                transition_type=clip.transition_type,
                emotion=clip.emotion,
                plan_idx=clip.plan_idx,
            )
        )
        cursor = end

    if log:
        log(f"Beat sync: snapped {snap_count} cut points within ±{tolerance_seconds:.2f}s")
    return rebuilt


def apply_rhythm_sync(
    timeline: list[TimelineClip],
    voiceover_path: Path,
    music_path: Path | None,
    log: callable | None = None,
) -> list[TimelineClip]:
    rhythm_points: list[float] = []
    if music_path is not None:
        rhythm_points = detect_music_beats(music_path, log=log)
    if not rhythm_points:
        rhythm_points = detect_voiceover_pauses(voiceover_path, log=log)
    return snap_timeline_to_rhythm(timeline, rhythm_points=rhythm_points, log=log)