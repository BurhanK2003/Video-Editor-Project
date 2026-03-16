"""
Free text-to-speech for script-to-video.

Priority chain (all free):
  1. kokoro-onnx   — best local quality, no internet, no API key
                     pip install kokoro-onnx soundfile
  2. edge-tts      — Microsoft Edge neural TTS, internet required, no API key,
                     300+ voices, near-Azure quality
                     pip install edge-tts
  3. pyttsx3       — fully offline, lower quality, last resort
                     pip install pyttsx3

Usage
-----
    from auto_video_editor.tts import generate_voiceover
    out_path = generate_voiceover("Your script here.", Path("output/vo.wav"), log=print)

Returns the output path on success, raises RuntimeError if all backends fail.
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path


VOICE_KOKORO = "af_heart"   # kokoro-onnx default voice (American English female)
VOICE_EDGE   = "en-US-AriaNeural"  # edge-tts default — warm, natural


# ---------------------------------------------------------------------------
# Backend: kokoro-onnx (local, no internet)
# ---------------------------------------------------------------------------

def _try_kokoro(text: str, output_path: Path, log: callable) -> bool:
    try:
        from kokoro_onnx import Kokoro  # type: ignore
        import soundfile as sf           # type: ignore
        import numpy as np
    except ImportError:
        return False

    try:
        log("TTS: using kokoro-onnx (local, no internet required)")
        kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
        samples, sample_rate = kokoro.create(text, voice=VOICE_KOKORO, speed=1.0, lang="en-us")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), np.array(samples), sample_rate)
        log(f"TTS: saved to {output_path}")
        return True
    except Exception as exc:
        log(f"TTS kokoro-onnx failed ({exc})")
        return False


# ---------------------------------------------------------------------------
# Backend: edge-tts (internet required, no API key)
# ---------------------------------------------------------------------------

def _try_edge_tts(text: str, output_path: Path, voice: str, log: callable) -> bool:
    try:
        import edge_tts  # type: ignore
    except ImportError:
        return False

    try:
        log(f"TTS: using edge-tts (voice: {voice})")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async def _run() -> None:
            communicator = edge_tts.Communicate(text, voice)
            await communicator.save(str(output_path))

        asyncio.run(_run())
        log(f"TTS: saved to {output_path}")
        return True
    except Exception as exc:
        log(f"TTS edge-tts failed ({exc})")
        return False


# ---------------------------------------------------------------------------
# Backend: pyttsx3 (fully offline, lower quality)
# ---------------------------------------------------------------------------

def _try_pyttsx3(text: str, output_path: Path, log: callable) -> bool:
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        return False

    try:
        log("TTS: using pyttsx3 (local, low quality fallback)")
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        log(f"TTS: saved to {output_path}")
        return True
    except Exception as exc:
        log(f"TTS pyttsx3 failed ({exc})")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_edge_voices() -> list[str]:
    """Return available edge-tts voice names (requires internet on first call)."""
    try:
        import edge_tts  # type: ignore

        async def _get() -> list[str]:
            voices = await edge_tts.list_voices()
            return [v["ShortName"] for v in voices]

        return asyncio.run(_get())
    except Exception:
        return []


def generate_voiceover(
    text: str,
    output_path: Path,
    voice: str = VOICE_EDGE,
    log: callable = print,
) -> Path:
    """
    Convert *text* to speech and write to *output_path*.

    Tries kokoro-onnx → edge-tts → pyttsx3 in order.
    Raises RuntimeError if every backend fails.
    """
    cleaned = _clean_script(text)
    if not cleaned:
        raise ValueError("Script text is empty after cleaning.")

    # kokoro-onnx always outputs .wav; edge-tts can do .mp3/.wav
    if output_path.suffix.lower() not in {".wav", ".mp3"}:
        output_path = output_path.with_suffix(".wav")

    if _try_kokoro(cleaned, output_path, log):
        return output_path
    if _try_edge_tts(cleaned, output_path, voice or VOICE_EDGE, log):
        return output_path
    if _try_pyttsx3(cleaned, output_path, log):
        return output_path

    raise RuntimeError(
        "All TTS backends failed. Install at least one:\n"
        "  pip install kokoro-onnx soundfile\n"
        "  pip install edge-tts\n"
        "  pip install pyttsx3"
    )


def _clean_script(text: str) -> str:
    """Remove markdown, stage directions, and excess whitespace from a script."""
    text = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", text)   # bold/italic markdown
    text = re.sub(r"#{1,6}\s*", "", text)               # headings
    text = re.sub(r"\[.*?\]", "", text)                  # stage directions [pause]
    text = re.sub(r"\(.*?\)", "", text)                  # parentheticals
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
