from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.editor import ImageClip, VideoClip
except Exception:
    from moviepy import ImageClip, VideoClip

from .models import PlannedSegment


def _load_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    project_root = Path(__file__).resolve().parents[2]
    bundled = (
        project_root / "assets" / "fonts" / "Montserrat-Bold.ttf",
        project_root / "assets" / "fonts" / "Anton-Regular.ttf",
    )
    for path in bundled:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                pass

    fallback = ("arialbd.ttf", "segoeuib.ttf", "calibrib.ttf") if bold else ("arial.ttf", "segoeui.ttf", "calibri.ttf")
    for name in fallback:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return ""
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if current and (bbox[2] - bbox[0]) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines[:3])


def _extract_hook_text(subtitle_plan: list[PlannedSegment], override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    return ""


def _ease_out_elastic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    c4 = (2 * math.pi) / 3
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    return pow(2, -10 * x) * math.sin((x * 10 - 0.75) * c4) + 1


def _ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1 - pow(1 - x, 3)


def _hook_text_clip(width: int, height: int, text: str, duration: float) -> ImageClip | None:
    if not text.strip() or duration <= 0:
        return None

    canvas_w = int(width * 0.88)
    canvas_h = max(180, int(height * 0.22))
    image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((0, 0, canvas_w, canvas_h), radius=24, fill=(10, 10, 12, 185))

    font = _load_font(max(34, int(height * 0.06)), bold=True)
    wrapped = _wrap_text(draw, text.upper(), font, int(canvas_w * 0.9))
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=8)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (canvas_w - tw) // 2
    ty = (canvas_h - th) // 2
    draw.multiline_text(
        (tx, ty),
        wrapped,
        font=font,
        fill=(255, 255, 255, 255),
        align="center",
        spacing=8,
        stroke_width=2,
        stroke_fill=(0, 0, 0, 240),
    )

    clip = ImageClip(np.array(image)).set_start(0.0).set_duration(duration)
    slam_seconds = min(0.42, duration)

    def _scale_at(t: float) -> float:
        if t <= 0:
            return 0.01
        progress = min(1.0, t / max(1e-4, slam_seconds))
        return max(0.01, _ease_out_elastic(progress))

    clip = clip.resize(_scale_at).set_position(("center", int(height * 0.08)))
    return clip


def _stat_badge_clip(width: int, height: int, text: str, duration: float) -> ImageClip | None:
    if not text.strip() or duration <= 0:
        return None

    font = _load_font(max(24, int(height * 0.03)), bold=True)
    probe = Image.new("RGBA", (400, 120), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(probe)
    bbox = pdraw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = 28
    pad_y = 18
    badge_w = tw + pad_x * 2
    badge_h = th + pad_y * 2
    badge = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle((0, 0, badge_w, badge_h), radius=badge_h // 2, fill=(255, 112, 67, 240))
    draw.text((pad_x, pad_y), text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 210))

    start = min(1.0, max(0.2, duration * 0.12))
    clip = ImageClip(np.array(badge)).set_start(start).set_duration(max(0.2, duration - start))

    x_end = int(width * 0.06)
    x_start = -badge_w - 20
    y_pos = int(height * 0.34)
    fly_in_seconds = 0.75

    def _pos(t: float) -> tuple[int, int]:
        local = max(0.0, t - start)
        progress = min(1.0, local / max(1e-4, fly_in_seconds))
        eased = _ease_out_elastic(progress)
        x = int(x_start + (x_end - x_start) * eased)
        return (x, y_pos)

    return clip.set_position(_pos)


def _cta_card_clip(width: int, height: int, text: str, final_duration: float) -> ImageClip | None:
    if final_duration <= 0.3:
        return None
    cta = (text or "").strip()
    if not cta:
        return None

    card_w = int(width * 0.78)
    card_h = max(120, int(height * 0.12))
    image = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, card_w, card_h), radius=26, fill=(24, 26, 34, 235))
    draw.rounded_rectangle((8, 8, card_w - 8, card_h - 8), radius=22, outline=(255, 255, 255, 70), width=2)

    font = _load_font(max(30, int(height * 0.036)), bold=True)
    bbox = draw.textbbox((0, 0), cta, font=font)
    tx = (card_w - (bbox[2] - bbox[0])) // 2
    ty = (card_h - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), cta, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 210))

    start = max(0.0, final_duration - 2.0)
    clip = ImageClip(np.array(image)).set_start(start).set_duration(final_duration - start)

    y_end = int(height * 0.70)
    y_start = height + card_h + 20
    x_pos = (width - card_w) // 2
    slide_seconds = 0.45

    def _pos(t: float) -> tuple[int, int]:
        local = max(0.0, t - start)
        progress = min(1.0, local / max(1e-4, slide_seconds))
        eased = _ease_out_cubic(progress)
        y = int(y_start + (y_end - y_start) * eased)
        return (x_pos, y)

    return clip.set_position(_pos)


def _logo_clip(width: int, height: int, final_duration: float, logo_path: Path | None) -> ImageClip | None:
    if logo_path is None or not logo_path.exists() or not logo_path.is_file() or final_duration <= 0.6:
        return None

    try:
        logo = Image.open(str(logo_path)).convert("RGBA")
    except Exception:
        return None

    max_w = int(width * 0.16)
    scale = min(1.0, max_w / max(1, logo.width))
    new_w = max(1, int(round(logo.width * scale)))
    new_h = max(1, int(round(logo.height * scale)))
    logo = logo.resize((new_w, new_h), Image.LANCZOS)

    start = 0.5
    clip = ImageClip(np.array(logo)).set_start(start).set_duration(max(0.05, final_duration - start))
    clip = clip.set_opacity(0.70).crossfadein(0.35)
    return clip.set_position((int(width - new_w - width * 0.03), int(height * 0.03)))


def _progress_bar_clips(width: int, final_duration: float) -> list[ImageClip | VideoClip]:
    if final_duration <= 0:
        return []

    bar_h = 8
    bg = Image.new("RGBA", (width, bar_h), (255, 255, 255, 60))
    fill = Image.new("RGBA", (width, bar_h), (255, 255, 255, 235))

    bg_clip = ImageClip(np.array(bg)).set_start(0).set_duration(final_duration).set_position((0, 0))
    fill_clip = ImageClip(np.array(fill)).set_start(0).set_duration(final_duration).set_position((0, 0))

    def _resize_at(t: float):
        progress = max(0.0, min(1.0, t / max(1e-4, final_duration)))
        return (max(2, int(width * progress)), bar_h)

    fill_clip = fill_clip.resize(_resize_at)
    return [bg_clip, fill_clip]


def create_motion_graphics_overlays(
    subtitle_plan: list[PlannedSegment],
    width: int,
    height: int,
    final_duration: float,
    hook_text_override: str | None = None,
    stat_badge_text: str | None = None,
    cta_text: str | None = None,
    logo_path: Path | None = None,
    enable_progress_bar: bool = True,
) -> list[VideoClip]:
    overlays: list[VideoClip] = []

    hook_text = _extract_hook_text(subtitle_plan, hook_text_override)
    hook_clip = _hook_text_clip(width=width, height=height, text=hook_text, duration=min(1.5, final_duration))
    if hook_clip is not None:
        overlays.append(hook_clip)

    badge_text = (stat_badge_text or "").strip()
    badge_clip = _stat_badge_clip(width=width, height=height, text=badge_text, duration=final_duration)
    if badge_clip is not None:
        overlays.append(badge_clip)

    cta_clip = _cta_card_clip(width=width, height=height, text=(cta_text or ""), final_duration=final_duration)
    if cta_clip is not None:
        overlays.append(cta_clip)

    logo = _logo_clip(width=width, height=height, final_duration=final_duration, logo_path=logo_path)
    if logo is not None:
        overlays.append(logo)

    if enable_progress_bar:
        overlays.extend(_progress_bar_clips(width=width, final_duration=final_duration))

    return overlays
