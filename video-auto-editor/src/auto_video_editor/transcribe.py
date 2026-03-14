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
        log("faster-whisper not available. Trying openai-whisper.")
    else:
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
            log("faster-whisper produced no segments. Trying openai-whisper.")
        except Exception as exc:
            log(f"faster-whisper failed ({exc}). Trying openai-whisper.")

    try:
        import whisper

        log(f"Loading openai-whisper model: {model_size}")
        model = whisper.load_model(model_size)
        result = model.transcribe(str(voiceover_path), task="transcribe")
        segments: list[TranscriptSegment] = []
        for seg in result.get("segments", []):
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text,
                )
            )
        if segments:
            return segments
        log("openai-whisper produced no segments. Falling back to a single segment.")
    except Exception as exc:
        log(f"openai-whisper failed ({exc}). Falling back to a single segment.")

    duration = _audio_duration_seconds(voiceover_path)
    return [
        TranscriptSegment(start=0.0, end=duration, text="Voiceover segment")
    ]
