from __future__ import annotations

import os
import shutil
import importlib


def ensure_pillow_compatibility() -> None:
    """Backfill Pillow constants removed in newer releases but still used by MoviePy."""
    try:
        from PIL import Image
    except Exception:
        return

    if hasattr(Image, "ANTIALIAS"):
        return

    resampling = getattr(Image, "Resampling", None)
    if resampling is not None and hasattr(resampling, "LANCZOS"):
        Image.ANTIALIAS = resampling.LANCZOS


def ensure_ffmpeg_for_moviepy() -> str | None:
    """Resolve an ffmpeg binary path for MoviePy if PATH does not contain ffmpeg."""
    existing = shutil.which("ffmpeg")
    if existing:
        os.environ.setdefault("FFMPEG_BINARY", existing)
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", existing)
        return existing

    try:
        module = importlib.import_module("imageio_ffmpeg")
        resolved = module.get_ffmpeg_exe()
    except Exception:
        return None

    if resolved:
        os.environ["FFMPEG_BINARY"] = resolved
        os.environ["IMAGEIO_FFMPEG_EXE"] = resolved
    return resolved
