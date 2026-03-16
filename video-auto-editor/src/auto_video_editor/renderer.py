from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.audio.fx.all import audio_loop
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, VideoClip, VideoFileClip, concatenate_videoclips

from .models import PlannedSegment, TimelineClip, WordToken

TRANSITION_STYLES = ("none", "crossfade", "zoom", "fade_black")
SEGMENT_TRANSITIONS = ("jump_cut", "zoom_in", "whip", "fade")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MICRO_ZOOM_EVERY_SECONDS = 2.35
MICRO_ZOOM_PULSE_SECONDS = 0.46
MICRO_ZOOM_STRENGTH = 0.055

CAPTION_STYLE_PRESETS = {
    "beast": {
        "base_fill": (20, 20, 20, 190),
        "emphasis_fill": (30, 24, 10, 208),
        "accent": (255, 196, 0, 220),
        "emphasis_accent": (255, 223, 92, 235),
        "text": (255, 255, 255, 255),
        "highlight": (255, 244, 179, 255),
        "shadow": (0, 0, 0, 90),
        "position_ratio": 0.54,
        "pop_strength": 0.05,
        "max_lines": 6,
    },
    "clean": {
        "base_fill": (14, 17, 24, 172),
        "emphasis_fill": (25, 38, 58, 188),
        "accent": (100, 197, 255, 210),
        "emphasis_accent": (146, 219, 255, 228),
        "text": (248, 251, 255, 255),
        "highlight": (199, 237, 255, 255),
        "shadow": (0, 0, 0, 70),
        "position_ratio": 0.58,
        "pop_strength": 0.03,
        "max_lines": 5,
    },
    "kinetic": {
        "base_fill": (15, 13, 19, 184),
        "emphasis_fill": (38, 15, 16, 205),
        "accent": (255, 107, 107, 218),
        "emphasis_accent": (255, 158, 158, 232),
        "text": (255, 255, 255, 255),
        "highlight": (255, 214, 163, 255),
        "shadow": (0, 0, 0, 100),
        "position_ratio": 0.52,
        "pop_strength": 0.07,
        "max_lines": 5,
    },
}

EMPHASIS_HINT_WORDS = {
    # Ordinals — structural anchors in listicle shorts
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    # Intensity / negation
    "never", "always", "only", "must", "warning",
    "secret", "new", "hidden", "unknown",
    # Superlatives / scale
    "biggest", "smallest", "fastest", "slowest",
    "best", "worst", "most", "least", "every", "last",
    # Hook / viral trigger words
    "amazing", "incredible", "shocking", "surprising",
    "insane", "wild", "crazy", "impossible", "actually",
    # Stakes
    "free", "win", "lose", "danger", "death", "survive",
    "million", "billion", "thousand", "percent",
    # Challenge / MrBeast vocabulary
    "challenge", "beat", "earn", "spend", "left",
}


def _fit_clip_to_canvas(clip: VideoFileClip, width: int, height: int) -> VideoFileClip:
    """Scale to fill target frame and center-crop to preserve aspect ratio."""
    target_ratio = width / height
    source_ratio = clip.w / clip.h

    if source_ratio > target_ratio:
        resized = clip.resize(height=height)
    else:
        resized = clip.resize(width=width)

    return resized.crop(x_center=resized.w / 2, y_center=resized.h / 2, width=width, height=height)


def _apply_micro_zooms(
    clip: VideoFileClip,
    pulse_every: float = MICRO_ZOOM_EVERY_SECONDS,
    pulse_seconds: float = MICRO_ZOOM_PULSE_SECONDS,
    strength: float = MICRO_ZOOM_STRENGTH,
) -> VideoFileClip:
    """Apply periodic micro-zoom pulses to keep visuals feeling active.

    Pulses are subtle and brief to avoid seasickness while still improving retention.
    """
    cycle = max(1.6, float(pulse_every))
    width, height = int(clip.w), int(clip.h)

    def scale_at_time(t: float) -> float:
        phase = float(t) % cycle
        if phase > pulse_seconds:
            return 1.0
        pulse_pos = phase / max(0.05, pulse_seconds)
        envelope = math.sin(math.pi * pulse_pos)
        return 1.0 + max(0.0, float(strength)) * envelope

    # Use clip.fl() with direct PIL scaling to bypass moviepy's clip.resize(callable)
    # path, which reverses dimensions before passing to PIL and relies on Image.ANTIALIAS.
    def _zoom_frame(gf, t: float, cw: int = width, ch: int = height) -> np.ndarray:
        scale = scale_at_time(t)
        if abs(scale - 1.0) < 1e-4:
            return gf(t)
        new_w = max(cw, int(round(cw * scale)))
        new_h = max(ch, int(round(ch * scale)))
        resized = Image.fromarray(gf(t)).resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(resized)
        y0 = (new_h - ch) // 2
        x0 = (new_w - cw) // 2
        return arr[y0 : y0 + ch, x0 : x0 + cw]

    return clip.fl(_zoom_frame, keep_duration=True)


def _pick_music_track(music_folder: Path | None) -> Path | None:
    if not music_folder or not music_folder.exists():
        return None
    tracks = [
        p
        for p in sorted(music_folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not tracks:
        return None
    return random.choice(tracks)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int = 3,
) -> str:
    words = text.strip().split()
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
    return "\n".join(lines[: max(1, int(max_lines))])


def _load_caption_font(height: int, scale: float = 1.0) -> ImageFont.ImageFont:
    size = max(24, int(height * 0.03 * max(0.6, float(scale))))
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_caption_font_bold(height: int, scale: float = 1.0) -> ImageFont.ImageFont:
    size = max(26, int(height * 0.032 * max(0.6, float(scale))))
    for name in ("arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return _load_caption_font(height, scale=scale)


def _apply_ken_burns(clip: VideoFileClip, zoom_ratio: float = 0.06) -> VideoFileClip:
    """Slowly zoom into the clip over its duration (Ken Burns effect).

    The clip is centre-cropped so the output dimensions never change.
    *zoom_ratio* is the fractional size increase at the end (0.06 = 6%).
    """
    w, h = int(clip.w), int(clip.h)

    def zoom_frame(gf, t: float):
        progress = min(t / max(clip.duration, 1e-3), 1.0)
        scale = 1.0 + zoom_ratio * progress
        frame = gf(t)
        new_w = max(w, int(round(w * scale)))
        new_h = max(h, int(round(h * scale)))
        resized = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(resized)
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return arr[y0 : y0 + h, x0 : x0 + w]

    return clip.fl(zoom_frame)


def _build_trimmed_voiceover(
    voiceover_path: Path,
    subtitle_plan: list[PlannedSegment],
    log: callable,
) -> tuple[AudioClip, list[PlannedSegment]]:
    """Trim long pauses between transcript segments and remap subtitle times."""
    source = AudioFileClip(str(voiceover_path))
    if not subtitle_plan:
        return source, subtitle_plan

    sorted_plan = sorted(subtitle_plan, key=lambda s: float(s.start))
    audio_parts: list[AudioClip] = []
    remapped: list[PlannedSegment] = []

    max_gap_keep = 0.22
    cursor = 0.0
    prev_end = 0.0
    trimmed_gap_total = 0.0

    for seg in sorted_plan:
        seg_start = max(prev_end, float(seg.start))
        seg_end = min(float(seg.end), float(source.duration))
        if seg_end <= seg_start:
            continue

        gap = max(0.0, seg_start - prev_end)
        keep_gap = min(gap, max_gap_keep)
        if keep_gap > 0.0:
            audio_parts.append(source.subclip(prev_end, prev_end + keep_gap))
            cursor += keep_gap
        trimmed_gap_total += max(0.0, gap - keep_gap)

        part = source.subclip(seg_start, seg_end)
        audio_parts.append(part)

        new_start = cursor
        new_end = new_start + (seg_end - seg_start)
        remapped.append(
            PlannedSegment(
                start=new_start,
                end=new_end,
                text=seg.text,
                duration=max(0.45, new_end - new_start),
                transition_after=seg.transition_after,
                transition_seconds=seg.transition_seconds,
                emphasis=seg.emphasis,
                highlight_phrase=seg.highlight_phrase,
                emphasis_words=seg.emphasis_words,
                word_tokens=[
                    WordToken(
                        start=new_start + max(0.0, float(w.start) - seg_start),
                        end=min(new_end, new_start + max(0.0, float(w.end) - seg_start)),
                        text=w.text,
                    )
                    for w in (seg.word_tokens or [])
                ]
                or None,
                visual_query=seg.visual_query,
                emotion=seg.emotion,
                pacing=seg.pacing,
                transition_type=seg.transition_type,
                clip_length_seconds=seg.clip_length_seconds,
            )
        )
        cursor = new_end
        prev_end = seg_end

    # Keep a tiny tail so the final spoken word does not sound clipped.
    tail = min(0.12, max(0.0, float(source.duration) - prev_end))
    if tail > 0:
        audio_parts.append(source.subclip(prev_end, prev_end + tail))

    if not audio_parts:
        return source, subtitle_plan

    trimmed_audio = concatenate_audioclips(audio_parts)
    log(f"Trimmed silence between sentences: {trimmed_gap_total:.2f}s removed")
    # Do not close `source` here; concatenated subclips still read frames from it.
    return trimmed_audio, remapped


def _compose_with_adaptive_transitions(
    clips: list[VideoFileClip],
    transition_plan: list[TimelineClip],
    style: str,
    default_dur: float,
    size: tuple[int, int],
) -> tuple[CompositeVideoClip, int]:
    """Build a timeline with per-cut transition decisions from the plan."""
    if not clips:
        raise ValueError("No clips to compose")

    if style not in TRANSITION_STYLES:
        style = "crossfade"

    prepared = clips
    if style == "zoom":
        prepared = [_apply_ken_burns(c) for c in clips]

    timeline: list[VideoFileClip] = []
    cursor = 0.0
    applied = 0

    for i, clip in enumerate(prepared):
        start = cursor
        transition_enabled = False
        trans_dur = max(0.1, min(float(default_dur), 0.8))

        if i > 0:
            prev = transition_plan[i - 1] if i - 1 < len(transition_plan) else None
            if prev is not None:
                transition_enabled = bool(prev.transition_after) and style != "none"
                trans_dur = max(0.1, min(float(prev.transition_seconds), 0.8))
                transition_kind = prev.transition_type if prev.transition_type in SEGMENT_TRANSITIONS else "jump_cut"
            else:
                transition_kind = "jump_cut"
        else:
            transition_kind = "jump_cut"

        if transition_enabled:
            if transition_kind == "jump_cut":
                pass
            elif transition_kind == "zoom_in":
                clip = _apply_ken_burns(clip, zoom_ratio=0.09).crossfadein(max(0.10, trans_dur))
                start = max(0.0, cursor - max(0.10, trans_dur))
                applied += 1
            elif transition_kind == "whip":
                whip_dur = max(0.08, min(trans_dur, 0.16))
                clip = clip.crossfadein(whip_dur)
                start = max(0.0, cursor - whip_dur)
                applied += 1
            elif transition_kind == "fade":
                fade_dur = max(0.10, trans_dur)
                clip = clip.fadein(fade_dur)
                if timeline:
                    timeline[-1] = timeline[-1].fadeout(fade_dur)
                applied += 1

        timeline.append(clip.set_start(start))
        cursor = max(cursor, start + clip.duration)

    return CompositeVideoClip(timeline, size=size).set_duration(cursor), applied


def _window_energy(clip: AudioClip, center: float, window: float = 0.16, samples: int = 11) -> float:
    """Estimate speech intensity around a timestamp using RMS energy."""
    if clip is None or not hasattr(clip, "get_frame"):
        return 0.0
    if clip.duration <= 0:
        return 0.0

    half = window / 2.0
    start = max(0.0, center - half)
    end = min(float(clip.duration), center + half)
    if end <= start:
        return 0.0

    energies: list[float] = []
    for i in range(samples):
        t = start + (end - start) * (i / max(1, samples - 1))
        try:
            frame = np.array(clip.get_frame(t), dtype=np.float32)
        except Exception:
            return 0.0
        if frame.ndim > 0:
            frame = np.mean(frame)
        val = float(abs(frame))
        energies.append(val * val)
    return math.sqrt(float(np.mean(energies))) if energies else 0.0


def _sync_plan_to_voice_energy(
    subtitle_plan: list[PlannedSegment],
    voiceover_clip: AudioClip,
    log: callable,
) -> list[PlannedSegment]:
    """Beat-sync transitions and emphasis from measured voice intensity."""
    if not subtitle_plan:
        return subtitle_plan

    seg_energy: list[float] = []
    boundary_energy: list[float] = []

    for idx, seg in enumerate(subtitle_plan):
        center = (float(seg.start) + float(seg.end)) / 2.0
        seg_energy.append(_window_energy(voiceover_clip, center=center, window=0.20, samples=13))
        if idx < len(subtitle_plan) - 1:
            boundary_t = float(seg.end)
            boundary_energy.append(_window_energy(voiceover_clip, center=boundary_t, window=0.14, samples=9))

    baseline = max(1e-4, float(np.median(seg_energy)) if seg_energy else 1e-4)

    updated: list[PlannedSegment] = []
    forced_emphasis = 0
    transitions_disabled = 0

    for idx, seg in enumerate(subtitle_plan):
        local_energy = seg_energy[idx] if idx < len(seg_energy) else baseline
        energy_ratio = local_energy / baseline

        emphasis = seg.emphasis or energy_ratio >= 1.28
        if emphasis and not seg.emphasis:
            forced_emphasis += 1

        transition_after = seg.transition_after
        transition_seconds = max(0.10, min(float(seg.transition_seconds), 0.80))
        if idx < len(boundary_energy):
            b_ratio = boundary_energy[idx] / baseline
            if b_ratio < 0.70:
                transition_after = False
                transitions_disabled += 1
            elif b_ratio >= 1.35:
                transition_after = True
                transition_seconds = max(0.10, transition_seconds * 0.72)
            elif b_ratio >= 1.10:
                transition_seconds = max(0.10, transition_seconds * 0.85)

        updated.append(
            PlannedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                duration=seg.duration,
                transition_after=transition_after,
                transition_seconds=transition_seconds,
                emphasis=emphasis,
                highlight_phrase=seg.highlight_phrase,
                emphasis_words=seg.emphasis_words,
                word_tokens=seg.word_tokens,
                visual_query=seg.visual_query,
                emotion=seg.emotion,
                pacing=seg.pacing,
                transition_type=seg.transition_type,
                clip_length_seconds=seg.clip_length_seconds,
            )
        )

    log(
        "Beat sync: "
        f"{forced_emphasis} extra emphasis points, "
        f"{transitions_disabled} low-energy cuts switched to hard cuts"
    )
    return updated


def _derive_emphasis_words(segment: PlannedSegment) -> list[str]:
    words: list[str] = []
    for item in segment.emphasis_words or []:
        token = (item or "").strip().lower()
        if token:
            words.append(token)

    tokens = re.findall(r"\b\w+\b", segment.text or "")
    for tok in tokens:
        lower = tok.lower()
        if lower in EMPHASIS_HINT_WORDS:
            words.append(lower)
        if tok.isupper() and len(tok) >= 3:
            words.append(lower)
        if any(ch.isdigit() for ch in tok):
            words.append(lower)

    if segment.highlight_phrase:
        for tok in re.findall(r"\b\w+\b", segment.highlight_phrase):
            words.append(tok.lower())

    unique: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        unique.append(word)
    return unique[:4]


def _render_caption_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box_width: int,
    emphasis_words: list[str],
    text_fill: tuple[int, int, int, int],
    highlight_fill: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    lines = text.split("\n")
    line_boxes = [draw.textbbox((0, 0), line or " ", font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_boxes]
    line_spacing = 6
    text_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)

    y = 0
    width_used = 0
    for idx, line in enumerate(lines):
        words = line.split()
        if not words:
            y += line_heights[idx] + line_spacing
            continue

        space_bbox = draw.textbbox((0, 0), " ", font=font)
        space_w = max(1, space_bbox[2] - space_bbox[0])

        widths = [draw.textbbox((0, 0), word, font=font)[2] for word in words]
        row_w = sum(widths) + space_w * max(0, len(words) - 1)
        width_used = max(width_used, row_w)

        x = max(0, (box_width - row_w) // 2)
        for i, word in enumerate(words):
            fill = text_fill
            normalized = re.sub(r"[^A-Za-z0-9']+", "", word).lower()
            if normalized in emphasis_words:
                fill = highlight_fill

            draw.text(
                (x, y),
                word,
                font=font,
                fill=fill,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 230),
            )
            x += widths[i] + (space_w if i < len(words) - 1 else 0)
        y += line_heights[idx] + line_spacing

    return width_used, text_h, line_spacing


def _frame_saliency_center_y(frame: np.ndarray) -> float:
    """Return normalized Y center of visual saliency as a proxy for subject position."""
    if frame is None or frame.size == 0:
        return 0.5

    gray = frame.mean(axis=2).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    sal = gx + gy
    total = float(sal.sum())
    if total <= 1e-6:
        return 0.5

    ys = np.arange(sal.shape[0], dtype=np.float32).reshape(-1, 1)
    cy = float((sal * ys).sum() / total)
    return max(0.0, min(1.0, cy / max(1.0, sal.shape[0] - 1)))


def _adaptive_caption_position_ratio(
    base_video: VideoFileClip | CompositeVideoClip | None,
    segment_start: float,
    segment_end: float,
    default_ratio: float,
    cache: dict[int, float],
) -> float:
    """Place captions away from likely subject region using frame saliency."""
    if base_video is None or not hasattr(base_video, "get_frame"):
        return default_ratio

    mid = max(0.0, (float(segment_start) + float(segment_end)) / 2.0)
    key = int(round(mid * 2.0))  # cache per 0.5s bucket
    if key in cache:
        return cache[key]

    try:
        frame = np.array(base_video.get_frame(mid))
    except Exception:
        return default_ratio

    focus_y = _frame_saliency_center_y(frame)
    chosen = float(default_ratio)
    if focus_y <= 0.40:
        chosen = 0.64
    elif focus_y >= 0.60:
        chosen = 0.42
    else:
        gray = frame.mean(axis=2).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        sal = gx + gy
        h = sal.shape[0]
        top_band = sal[int(0.10 * h) : int(0.35 * h), :]
        bottom_band = sal[int(0.65 * h) : int(0.90 * h), :]
        top_energy = float(top_band.mean()) if top_band.size else 0.0
        bottom_energy = float(bottom_band.mean()) if bottom_band.size else 0.0
        chosen = 0.42 if top_energy <= bottom_energy else 0.64

    chosen = max(0.35, min(0.80, chosen))
    cache[key] = chosen
    return chosen


def _word_windows_for_segment(segment: PlannedSegment) -> list[tuple[str, float, float]]:
    tokens = [w for w in (segment.word_tokens or []) if (w.text or "").strip()]
    if not tokens:
        words = [w for w in (segment.text or "").split() if w.strip()]
        if not words:
            return []
        dur = max(0.2, float(segment.end) - float(segment.start))
        step = dur / len(words)
        windows: list[tuple[str, float, float]] = []
        for i, word in enumerate(words):
            ws = i * step
            we = dur if i == len(words) - 1 else (i + 1) * step
            windows.append((word, ws, we))
        return windows

    seg_start = float(segment.start)
    seg_dur = max(0.2, float(segment.end) - seg_start)
    windows = []
    for token in tokens:
        ws = max(0.0, float(token.start) - seg_start)
        we = max(ws + 0.04, float(token.end) - seg_start)
        windows.append((token.text, min(ws, seg_dur), min(we, seg_dur)))
    return windows


def _active_word_index(t_local: float, windows: list[tuple[str, float, float]]) -> int:
    if not windows:
        return -1
    for i, (_, ws, we) in enumerate(windows):
        if ws <= t_local <= we:
            return i
    if t_local < windows[0][1]:
        return 0
    return len(windows) - 1


def _make_segment_caption_fns(
    width: int,
    box_height: int,
    rect_x0: int,
    rect_y0: int,
    rect_x1: int,
    rect_y1: int,
    rect_w: int,
    rect_h: int,
    pad_x: int,
    safe_text_h: int,
    words_by_line: list,
    word_windows: list,
    font: ImageFont.ImageFont,
    emphasis_words: list,
    preset: dict,
    segment_emphasis: bool,
    enable_karaoke_highlight: bool,
):
    """
    Factory that captures all per-segment variables by value and returns
    (make_caption_frame, make_caption_mask) callables free of loop-closure bugs.
    """
    line_spacing = 6

    def make_caption_rgba_frame(t: float) -> np.ndarray:
        image = Image.new("RGBA", (width, box_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        draw.rounded_rectangle(
            (rect_x0 + 5, rect_y0 + 5, rect_x1 + 5, rect_y1 + 5),
            radius=14,
            fill=preset["shadow"],
        )
        base_fill = preset["base_fill"] if not segment_emphasis else preset["emphasis_fill"]
        draw.rounded_rectangle((rect_x0, rect_y0, rect_x1, rect_y1), radius=14, fill=base_fill)

        accent_h = max(6, int(rect_h * 0.12))
        accent_color = preset["accent"] if not segment_emphasis else preset["emphasis_accent"]
        draw.rounded_rectangle((rect_x0, rect_y0, rect_x1, rect_y0 + accent_h), radius=14, fill=accent_color)

        text_canvas = Image.new("RGBA", (max(1, rect_w - pad_x * 2), max(1, safe_text_h + 8)), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_canvas)

        active_idx = _active_word_index(t, word_windows) if enable_karaoke_highlight else -1
        word_cursor = 0
        y = 0
        for line_words in words_by_line:
            if not line_words:
                y += line_spacing + 8
                continue

            widths = [text_draw.textbbox((0, 0), w, font=font)[2] for w in line_words]
            space_w = max(1, text_draw.textbbox((0, 0), " ", font=font)[2])
            row_w = sum(widths) + space_w * max(0, len(line_words) - 1)
            x = max(0, (text_canvas.width - row_w) // 2)
            line_h = max((text_draw.textbbox((0, 0), w, font=font)[3] for w in line_words), default=20)

            for i, word in enumerate(line_words):
                normalized = re.sub(r"[^A-Za-z0-9']+", "", word).lower()
                fill = preset["text"]
                if normalized in emphasis_words:
                    fill = preset["highlight"]
                if word_cursor == active_idx:
                    fill = preset["highlight"]

                text_draw.text(
                    (x, y),
                    word,
                    font=font,
                    fill=fill,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 230),
                )
                x += widths[i] + (space_w if i < len(line_words) - 1 else 0)
                word_cursor += 1

            y += line_h + line_spacing

        text_x = rect_x0 + pad_x
        text_y = max(0, (box_height - safe_text_h) // 2)
        image.alpha_composite(text_canvas, dest=(text_x, text_y))
        return np.array(image)

    def make_caption_frame(t: float) -> np.ndarray:
        rgba = make_caption_rgba_frame(t)
        if rgba.ndim == 3 and rgba.shape[2] >= 3:
            return rgba[:, :, :3]
        return rgba

    def make_caption_mask(t: float) -> np.ndarray:
        rgba = make_caption_rgba_frame(t)
        if rgba.ndim == 3 and rgba.shape[2] >= 4:
            return rgba[:, :, 3].astype(np.float32) / 255.0
        return np.ones((box_height, width), dtype=np.float32)

    return make_caption_frame, make_caption_mask


def _subtitle_overlays(
    subtitle_plan: list[PlannedSegment],
    width: int,
    height: int,
    final_duration: float,
    caption_style: str,
    caption_position_ratio: float | None = None,
    caption_max_lines: int | None = None,
    caption_font_scale: float = 1.0,
    caption_pop_scale: float = 1.0,
    base_video: VideoFileClip | CompositeVideoClip | None = None,
    enable_adaptive_caption_safe_zones: bool = True,
    enable_karaoke_highlight: bool = True,
) -> list[VideoClip]:
    if not subtitle_plan:
        return []

    preset = dict(CAPTION_STYLE_PRESETS.get(caption_style, CAPTION_STYLE_PRESETS["beast"]))
    if caption_position_ratio is not None:
        preset["position_ratio"] = max(0.35, min(float(caption_position_ratio), 0.85))
    if caption_max_lines is not None:
        preset["max_lines"] = max(1, min(int(caption_max_lines), 6))

    min_box_height = max(180, int(height * 0.16))
    max_box_height = max(min_box_height, int(height * 0.65))
    max_text_width = int(width * 0.84)
    overlays: list[VideoClip] = []
    pos_cache: dict[int, float] = {}

    for segment in subtitle_plan:
        start = max(0.0, float(segment.start))
        end = min(final_duration, float(segment.end))
        if end <= start:
            continue
        text = (segment.text or "").strip()
        if not text:
            continue

        font = (
            _load_caption_font_bold(height, scale=caption_font_scale)
            if segment.emphasis
            else _load_caption_font(height, scale=caption_font_scale)
        )
        emphasis_words = _derive_emphasis_words(segment)

        # Measure text on a probe canvas first, then build a per-segment caption canvas.
        probe = Image.new("RGBA", (width, max(256, min_box_height)), (0, 0, 0, 0))
        probe_draw = ImageDraw.Draw(probe)

        wrapped = _wrap_text(
            probe_draw,
            text,
            font,
            max_text_width,
            max_lines=int(preset.get("max_lines", 3)),
        )
        if not wrapped:
            continue

        bbox = probe_draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=6)
        # Pillow 10+ returns float bbox values for TrueType fonts — cast immediately
        # so every downstream Image.new() size and alpha_composite dest stays int.
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
        line_count = max(1, wrapped.count("\n") + 1)
        # Extra vertical allowance for stroke + line spacing avoids clipped descenders.
        safe_text_h = text_h + line_count * 4 + 10

        pad_x = 28
        pad_y = 18
        rect_w = min(width - 20, text_w + pad_x * 2)
        rect_h = safe_text_h + pad_y * 2
        box_height = max(min_box_height, min(max_box_height, rect_h + 24))

        rect_x0 = (width - rect_w) // 2
        rect_y0 = (box_height - rect_h) // 2
        rect_x1 = rect_x0 + rect_w
        rect_y1 = rect_y0 + rect_h

        words_by_line = [line.split() for line in wrapped.split("\n")]
        word_windows = _word_windows_for_segment(segment)

        make_caption_frame, make_caption_mask = _make_segment_caption_fns(
            width=width,
            box_height=box_height,
            rect_x0=rect_x0,
            rect_y0=rect_y0,
            rect_x1=rect_x1,
            rect_y1=rect_y1,
            rect_w=rect_w,
            rect_h=rect_h,
            pad_x=pad_x,
            safe_text_h=safe_text_h,
            words_by_line=words_by_line,
            word_windows=word_windows,
            font=font,
            emphasis_words=emphasis_words,
            preset=preset,
            segment_emphasis=segment.emphasis,
            enable_karaoke_highlight=enable_karaoke_highlight,
        )

        seg_ratio = float(preset.get("position_ratio", 0.60))
        if enable_adaptive_caption_safe_zones:
            seg_ratio = _adaptive_caption_position_ratio(
                base_video=base_video,
                segment_start=start,
                segment_end=end,
                default_ratio=seg_ratio,
                cache=pos_cache,
            )

        duration = max(0.05, end - start)
        clip = (
            VideoClip(make_frame=make_caption_frame, duration=duration)
            .set_start(start)
            .set_end(end)
            .set_position(("center", int(height * seg_ratio)))
        )
        mask_clip = VideoClip(make_frame=make_caption_mask, ismask=True, duration=duration).set_start(start).set_end(end)
        clip = clip.set_mask(mask_clip)
        clip = clip.crossfadein(0.12).crossfadeout(0.10)
        if segment.emphasis:
            pop = float(preset.get("pop_strength", 0.05)) * max(0.5, float(caption_pop_scale))
            _cw, _ch, _p = int(width), int(box_height), pop

            def _pop_frame(gf, t: float, cw: int = _cw, ch: int = _ch, p: float = _p) -> np.ndarray:
                scale = 1.0 + p * math.exp(-8.0 * t)
                if abs(scale - 1.0) < 1e-4:
                    return gf(t)
                new_w = max(cw, int(round(cw * scale)))
                new_h = max(ch, int(round(ch * scale)))
                resized = Image.fromarray(gf(t)).resize((new_w, new_h), Image.LANCZOS)
                arr = np.array(resized)
                y0 = (new_h - ch) // 2
                x0 = (new_w - cw) // 2
                return arr[y0 : y0 + ch, x0 : x0 + cw]

            clip = clip.fl(_pop_frame, keep_duration=True)
        overlays.append(clip)

    return overlays


def _insert_gap_freeze_frames(
    assembled: list,
    timeline_clips: list[TimelineClip],
    remapped_plan: list[PlannedSegment],
    log: callable,
) -> tuple[list, list[TimelineClip]]:
    """Insert freeze frames at plan-segment boundaries to match trimmed-audio timing.

    Without this, video clips advance faster than the audio because the trimmed
    voiceover contains small silence gaps (≤0.22 s) between spoken segments that
    the raw clip timeline does not account for.  Inserting a freeze frame at each
    boundary closes the accumulated drift so clip changes and subtitle changes
    stay aligned with what the audience hears.
    """
    if not assembled or not remapped_plan:
        return assembled, timeline_clips

    new_assembled: list = []
    new_timeline_clips: list[TimelineClip] = []
    last_plan_idx = -1
    total_inserted = 0

    for clip, tc in zip(assembled, timeline_clips):
        pidx = tc.plan_idx

        if pidx != last_plan_idx:
            # Compute the gap that precedes this plan segment in the trimmed audio.
            if pidx == 0:
                gap_dur = float(remapped_plan[0].start) if remapped_plan else 0.0
            else:
                prev_seg = remapped_plan[pidx - 1] if pidx - 1 < len(remapped_plan) else None
                cur_seg = remapped_plan[pidx] if pidx < len(remapped_plan) else None
                gap_dur = (
                    max(0.0, float(cur_seg.start) - float(prev_seg.end))
                    if prev_seg and cur_seg
                    else 0.0
                )

            if gap_dur > 0.02:
                # Freeze frame source: last frame of previous clip, or first frame if at start.
                src_clip = new_assembled[-1] if new_assembled else clip
                t_src = max(0.0, src_clip.duration - 0.05) if new_assembled else 0.0
                try:
                    frame = src_clip.get_frame(t_src)
                    hold = ImageClip(frame).set_duration(gap_dur)
                    dummy_tc = TimelineClip(
                        source_path=tc.source_path,
                        timeline_start=tc.timeline_start,
                        timeline_end=tc.timeline_start + gap_dur,
                        transition_after=False,
                        transition_seconds=0.0,
                        transition_type="jump_cut",
                        emotion=tc.emotion,
                        plan_idx=pidx,
                    )
                    new_assembled.append(hold)
                    new_timeline_clips.append(dummy_tc)
                    total_inserted += 1
                except Exception as exc:
                    log(f"Gap freeze frame skipped (plan_seg={pidx}): {exc}")

            last_plan_idx = pidx

        new_assembled.append(clip)
        new_timeline_clips.append(tc)

    if total_inserted:
        log(f"Sync gap frames inserted: {total_inserted} (total drift correction)")
    return new_assembled, new_timeline_clips


def render_video(
    timeline_clips: list[TimelineClip],
    subtitle_plan: list[PlannedSegment],
    voiceover_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    render_preset: str,
    music_folder: Path | None,
    log: callable,
    transition_style: str = "crossfade",
    transition_duration: float = 0.4,
    caption_style: str = "beast",
    caption_position_ratio: float | None = None,
    caption_max_lines: int | None = None,
    caption_font_scale: float = 1.0,
    caption_pop_scale: float = 1.0,
    enable_adaptive_caption_safe_zones: bool = True,
    enable_karaoke_highlight: bool = True,
) -> None:
    if not timeline_clips:
        raise ValueError("No timeline clips available. Add clips in the clips folder.")

    assembled = []
    opened_video_clips: list[VideoClip] = []
    subtitle_clips: list[VideoClip] = []
    voiceover_clip: AudioClip | None = None
    music_clip: AudioFileClip | None = None
    final_video = None

    try:
        for shot in timeline_clips:
            needed = max(0.2, float(shot.timeline_end - shot.timeline_start))
            if shot.source_path.suffix.lower() in IMAGE_EXTENSIONS:
                clip = ImageClip(str(shot.source_path)).set_duration(needed)
            else:
                clip = VideoFileClip(str(shot.source_path))
            opened_video_clips.append(clip)

            if clip.duration is not None and clip.duration <= 0:
                continue

            if clip.duration is not None and clip.duration < needed:
                repeats = int(needed // clip.duration) + 1
                extended = concatenate_videoclips([clip] * repeats).subclip(0, needed)
            elif clip.duration is not None:
                extended = clip.subclip(0, needed)
            else:
                extended = clip.set_duration(needed)

            fitted = _fit_clip_to_canvas(extended, width=width, height=height)
            assembled.append(fitted)

        if not assembled:
            raise ValueError("Failed to assemble output timeline.")

        voiceover_clip, subtitle_plan = _build_trimmed_voiceover(voiceover_path, subtitle_plan, log)
        subtitle_plan = _sync_plan_to_voice_energy(subtitle_plan, voiceover_clip, log)

        # Sync video clip changes to trimmed-audio segment boundaries by inserting
        # freeze frames for the inter-segment gaps kept in the trimmed voiceover.
        assembled, timeline_clips = _insert_gap_freeze_frames(
            assembled=assembled,
            timeline_clips=timeline_clips,
            remapped_plan=subtitle_plan,
            log=log,
        )

        log(f"Applying adaptive '{transition_style}' transitions")
        final_video, transition_count = _compose_with_adaptive_transitions(
            clips=assembled,
            transition_plan=timeline_clips,
            style=transition_style,
            default_dur=transition_duration,
            size=(width, height),
        )
        log(f"Transitions applied: {transition_count}")

        final_video = _apply_micro_zooms(final_video)
        log(
            f"Micro-zooms enabled: every {MICRO_ZOOM_EVERY_SECONDS:.2f}s "
            f"(pulse {MICRO_ZOOM_PULSE_SECONDS:.2f}s, strength {MICRO_ZOOM_STRENGTH:.3f})"
        )

        voice_duration = float(voiceover_clip.duration)
        if float(final_video.duration) < voice_duration:
            # Prevent voice truncation by freezing the last frame as needed.
            missing = voice_duration - float(final_video.duration)
            hold = final_video.to_ImageClip(t=max(0.0, float(final_video.duration) - 0.05)).set_duration(missing)
            final_video = concatenate_videoclips([final_video, hold], method="compose")

        final_duration = voice_duration
        log(f"Voiceover duration: {voice_duration:.2f}s | Final duration: {final_duration:.2f}s")
        final_video = final_video.subclip(0, final_duration)

        subtitle_clips = _subtitle_overlays(
            subtitle_plan=subtitle_plan,
            width=width,
            height=height,
            final_duration=final_duration,
            caption_style=caption_style,
            caption_position_ratio=caption_position_ratio,
            caption_max_lines=caption_max_lines,
            caption_font_scale=caption_font_scale,
            caption_pop_scale=caption_pop_scale,
            base_video=final_video,
            enable_adaptive_caption_safe_zones=enable_adaptive_caption_safe_zones,
            enable_karaoke_highlight=enable_karaoke_highlight,
        )
        log(f"Caption segments burned in: {len(subtitle_clips)}")
        if subtitle_clips:
            final_video = CompositeVideoClip([final_video, *subtitle_clips], size=(width, height))

        tracks = [voiceover_clip.volumex(1.0)]
        music_track = _pick_music_track(music_folder)
        if music_track:
            log(f"Selected music track: {music_track.name}")
            music_clip = AudioFileClip(str(music_track))
            music_clip = audio_loop(music_clip, duration=final_duration).volumex(0.15)
            tracks.append(music_clip)

        if len(tracks) == 1:
            final_audio = tracks[0].set_duration(final_duration)
        else:
            final_audio = CompositeAudioClip(tracks).set_duration(final_duration)
        final_video = final_video.without_audio().set_audio(final_audio)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        preset = render_preset if render_preset in {"veryfast", "medium", "slow"} else "veryfast"
        thread_count = max(1, (os.cpu_count() or 4) - 1)
        log(f"Rendering MP4 with preset={preset}, threads={thread_count}")
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            audio=True,
            temp_audiofile=str(output_path.with_suffix(".temp-audio.m4a")),
            remove_temp=True,
            fps=fps,
            preset=preset,
            threads=thread_count,
        )
        log(f"Export complete: {output_path}")
    finally:
        if final_video is not None:
            final_video.close()
        if voiceover_clip is not None:
            voiceover_clip.close()
        if music_clip is not None:
            music_clip.close()
        for c in subtitle_clips:
            c.close()
        for c in opened_video_clips:
            c.close()
