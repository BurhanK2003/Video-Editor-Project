from __future__ import annotations

from pathlib import Path

from moviepy.editor import AudioFileClip

from .models import TranscriptSegment


def _audio_duration_seconds(audio_path: Path) -> float:
    with AudioFileClip(str(audio_path)) as clip:
        return float(clip.duration)


def transcribe_voiceover(
    voiceover_path: Path,
    model_size: str,
    log: callable,
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        duration = _audio_duration_seconds(voiceover_path)
        log("faster-whisper not available. Falling back to a single segment.")
        return [
            TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")
        ]

    try:
        log(f"Loading faster-whisper model: {model_size}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        raw_segments, _ = model.transcribe(
            str(voiceover_path),
            beam_size=3,
            vad_filter=True,
        )
        segments = [
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=(seg.text or "").strip(),
            )
            for seg in raw_segments
            if (seg.text or "").strip()
        ]
        if segments:
            return segments

        duration = _audio_duration_seconds(voiceover_path)
        log("No transcript produced. Falling back to a single segment.")
        return [
            TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")
        ]
    except Exception as exc:
        duration = _audio_duration_seconds(voiceover_path)
        log(f"Transcription failed ({exc}). Falling back to a single segment.")
        return [
            TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")
        ]
