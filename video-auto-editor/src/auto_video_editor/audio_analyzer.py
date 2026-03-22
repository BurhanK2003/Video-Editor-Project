from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .models import TranscriptSegment, WordToken

_AUDIO_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "there", "its", "it's", "you", "your", "we",
    "our", "they", "their", "from", "into", "over", "just", "like", "very", "really", "then", "than",
}


@dataclass
class AudioAnalysis:
    transcript_segments: list[TranscriptSegment]
    keywords: list[str]
    rms_times: list[float]
    rms_values: list[float]
    beat_times: list[float]


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        try:
            from moviepy.editor import AudioFileClip
        except Exception:
            from moviepy import AudioFileClip
    except Exception:
        return 0.0

    with AudioFileClip(str(audio_path)) as clip:
        return float(clip.duration or 0.0)


def transcribe_voiceover_whisper(voiceover_path: Path, model_size: str, log: callable | None = None) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        duration = _audio_duration_seconds(voiceover_path)
        if log:
            log("faster-whisper unavailable in audio_analyzer; using fallback transcript segment.")
        return [TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")]

    try:
        device = "cpu"
        compute_type = "int8"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except Exception:
            pass

        if log:
            log(f"Whisper transcription: model={model_size}, device={device}, compute={compute_type}")

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        raw_segments, _ = model.transcribe(
            str(voiceover_path),
            beam_size=4,
            vad_filter=True,
            word_timestamps=True,
        )

        segments: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            words: list[WordToken] = []
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if not token:
                    continue
                words.append(
                    WordToken(
                        start=float(w.start),
                        end=float(w.end),
                        text=token,
                    )
                )
            segments.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    words=words or None,
                )
            )

        if segments:
            return segments

        duration = _audio_duration_seconds(voiceover_path)
        return [TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")]
    except Exception as exc:
        duration = _audio_duration_seconds(voiceover_path)
        if log:
            log(f"Whisper transcription failed ({exc}); using fallback segment.")
        return [TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")]


def extract_keywords_from_transcript(segments: list[TranscriptSegment], max_keywords: int = 24) -> list[str]:
    text = " ".join((s.text or "") for s in segments)
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    counts: dict[str, int] = {}
    for token in tokens:
        if len(token) < 3 or token in _AUDIO_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1

    ranked = sorted(counts.items(), key=lambda it: (-it[1], -len(it[0]), it[0]))
    return [k for k, _v in ranked[:max_keywords]]


def compute_rms_energy(audio_path: Path, hop_seconds: float = 0.1) -> tuple[list[float], list[float]]:
    try:
        import librosa
        import numpy as np
    except Exception:
        return [], []

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        hop_length = max(1, int(sr * max(0.02, float(hop_seconds))))
        frame_length = max(hop_length * 2, int(sr * 0.2))
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

        if len(rms) == 0:
            return [], []

        low = float(rms.min())
        high = float(rms.max())
        spread = max(1e-6, high - low)
        norm = [float((v - low) / spread) for v in rms]
        return [float(t) for t in times], norm
    except Exception:
        return [], []


def detect_music_beats(audio_path: Path) -> list[float]:
    try:
        import librosa
    except Exception:
        return []

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return [float(t) for t in beat_times if float(t) > 0.0]
    except Exception:
        return []


def energy_at_time(rms_times: list[float], rms_values: list[float], timestamp: float) -> float:
    if not rms_times or not rms_values:
        return 0.0
    if timestamp <= rms_times[0]:
        return float(rms_values[0])
    if timestamp >= rms_times[-1]:
        return float(rms_values[-1])

    # Linear scan is fine for short voiceovers; keeps logic straightforward.
    for idx in range(1, len(rms_times)):
        left_t = rms_times[idx - 1]
        right_t = rms_times[idx]
        if left_t <= timestamp <= right_t:
            alpha = (timestamp - left_t) / max(1e-6, right_t - left_t)
            return float(rms_values[idx - 1] * (1.0 - alpha) + rms_values[idx] * alpha)
    return 0.0


def analyze_audio(voiceover_path: Path, music_path: Path | None, whisper_model: str, log: callable | None = None) -> AudioAnalysis:
    segments = transcribe_voiceover_whisper(voiceover_path, whisper_model, log=log)
    keywords = extract_keywords_from_transcript(segments)
    rms_times, rms_values = compute_rms_energy(voiceover_path, hop_seconds=0.1)

    beat_source = music_path if music_path and music_path.exists() else voiceover_path
    beat_times = detect_music_beats(beat_source)

    if log:
        log(
            "Audio analysis: "
            f"segments={len(segments)}, keywords={len(keywords)}, "
            f"rms_points={len(rms_values)}, beats={len(beat_times)}"
        )

    return AudioAnalysis(
        transcript_segments=segments,
        keywords=keywords,
        rms_times=rms_times,
        rms_values=rms_values,
        beat_times=beat_times,
    )
