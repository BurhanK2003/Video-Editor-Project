from __future__ import annotations

import hashlib
from importlib import import_module
import math
import os
import random
import re
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.AudioClip import concatenate_audioclips

try:
    # MoviePy 2.x style (effect classes).
    AudioFadeIn = import_module("moviepy.audio.fx.AudioFadeIn").AudioFadeIn
    AudioFadeOut = import_module("moviepy.audio.fx.AudioFadeOut").AudioFadeOut
    AudioLoop = import_module("moviepy.audio.fx.AudioLoop").AudioLoop

    def audio_fadein(clip, duration):
        return clip.with_effects([AudioFadeIn(float(duration))])

    def audio_fadeout(clip, duration):
        return clip.with_effects([AudioFadeOut(float(duration))])

    def audio_loop(clip, duration):
        return clip.with_effects([AudioLoop(duration=float(duration))])
except Exception:
    # MoviePy 1.x style via dynamic import to avoid static-analysis false positives.
    _fx_all = import_module("moviepy.audio.fx.all")
    audio_fadein = _fx_all.audio_fadein
    audio_fadeout = _fx_all.audio_fadeout
    audio_loop = _fx_all.audio_loop

try:
    # MoviePy 2.x exports these from top-level package.
    from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, VideoClip, VideoFileClip, concatenate_videoclips
except Exception:
    # MoviePy 1.x fallback via dynamic import.
    _editor = import_module("moviepy.editor")
    AudioFileClip = _editor.AudioFileClip
    CompositeVideoClip = _editor.CompositeVideoClip
    ImageClip = _editor.ImageClip
    VideoClip = _editor.VideoClip
    VideoFileClip = _editor.VideoFileClip
    concatenate_videoclips = _editor.concatenate_videoclips

from .models import PlannedSegment, TimelineClip, WordToken
from .overlays import create_motion_graphics_overlays

TRANSITION_STYLES = ("none", "pro_weighted")
SEGMENT_TRANSITIONS = ("jump_cut", "zoom_in", "whip", "fade")
PRO_TRANSITION_WEIGHTS = (
    ("zoom_punch", 0.40),
    ("smash_cut", 0.30),
    ("whip_pan", 0.20),
    ("fade_black", 0.10),
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MICRO_ZOOM_EVERY_SECONDS = 2.35
MICRO_ZOOM_PULSE_SECONDS = 0.46
MICRO_ZOOM_STRENGTH = 0.055
MUSIC_BASE_GAIN = 0.12
MUSIC_DUCKED_GAIN = 0.045
MUSIC_MAX_VOICE_RATIO = 0.28
MUSIC_ENVELOPE_STEP_SECONDS = 0.08
MUSIC_FADE_IN_SECONDS = 0.85
MUSIC_FADE_OUT_SECONDS = 1.10
CAPTION_IN_POP_SECONDS = 0.08
CAPTION_OUT_FADE_SECONDS = 0.10


def _subclip(clip, start_time: float = 0.0, end_time: float | None = None):
    """Compatibility shim: MoviePy 1.x uses subclip, MoviePy 2.x uses subclipped."""
    if hasattr(clip, "subclip"):
        if end_time is None:
            return clip.subclip(start_time)
        return clip.subclip(start_time, end_time)
    if hasattr(clip, "subclipped"):
        if end_time is None:
            return clip.subclipped(start_time)
        return clip.subclipped(start_time, end_time)
    raise AttributeError("Clip object does not support subclip/subclipped")


def _resize(clip, *, width: int | None = None, height: int | None = None):
    if hasattr(clip, "resize"):
        return clip.resize(width=width, height=height)
    if hasattr(clip, "resized"):
        return clip.resized(width=width, height=height)
    raise AttributeError("Clip object does not support resize/resized")


def _crop(clip, *, x_center: float, y_center: float, width: int, height: int):
    if hasattr(clip, "crop"):
        return clip.crop(x_center=x_center, y_center=y_center, width=width, height=height)
    if hasattr(clip, "cropped"):
        x1 = max(0, int(round(x_center - width / 2)))
        y1 = max(0, int(round(y_center - height / 2)))
        x2 = x1 + int(width)
        y2 = y1 + int(height)
        return clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    raise AttributeError("Clip object does not support crop/cropped")


def _set_duration(clip, duration: float):
    if hasattr(clip, "set_duration"):
        return clip.set_duration(duration)
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    raise AttributeError("Clip object does not support set_duration/with_duration")


def _set_audio(clip, audio_clip):
    if hasattr(clip, "set_audio"):
        return clip.set_audio(audio_clip)
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    raise AttributeError("Clip object does not support set_audio/with_audio")


def _set_position(clip, position):
    if hasattr(clip, "set_position"):
        return clip.set_position(position)
    if hasattr(clip, "with_position"):
        return clip.with_position(position)
    raise AttributeError("Clip object does not support set_position/with_position")

CAPTION_STYLE_PRESETS = {
    "bold_stroke": {
        "base_fill": (20, 20, 20, 190),
        "emphasis_fill": (30, 24, 10, 208),
        "accent": (255, 196, 0, 220),
        "emphasis_accent": (255, 223, 92, 235),
        "text": (255, 255, 255, 255),
        "highlight": (255, 238, 170, 255),
        "active_highlight": (255, 220, 64, 255),
        "shadow": (0, 0, 0, 90),
        "position_ratio": 0.54,
        "pop_strength": 0.0,
        "max_lines": 6,
        "gradient_text": False,
    },
    "yellow_active": {
        "base_fill": (14, 17, 24, 172),
        "emphasis_fill": (25, 38, 58, 188),
        "accent": (100, 197, 255, 210),
        "emphasis_accent": (146, 219, 255, 228),
        "text": (248, 251, 255, 255),
        "highlight": (248, 251, 255, 255),
        "active_highlight": (255, 220, 64, 255),
        "shadow": (0, 0, 0, 70),
        "position_ratio": 0.58,
        "pop_strength": 0.0,
        "max_lines": 5,
        "gradient_text": False,
    },
    "gradient_fill": {
        "base_fill": (15, 13, 19, 184),
        "emphasis_fill": (38, 15, 16, 205),
        "accent": (255, 107, 107, 218),
        "emphasis_accent": (255, 158, 158, 232),
        "text": (255, 255, 255, 255),
        "highlight": (255, 214, 163, 255),
        "active_highlight": (255, 228, 130, 255),
        "shadow": (0, 0, 0, 100),
        "position_ratio": 0.52,
        "pop_strength": 0.0,
        "max_lines": 5,
        "gradient_text": True,
        "gradient_top": (255, 116, 255, 255),
        "gradient_bottom": (102, 225, 255, 255),
    },
    # Backward-compat aliases for existing config values.
    "beast": {},
    "clean": {},
    "kinetic": {},
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


def _fit_clip_to_canvas(clip: Any, width: int, height: int) -> Any:
    """Scale to fill target frame and center-crop to preserve aspect ratio."""
    target_ratio = width / height
    source_ratio = clip.w / clip.h

    if source_ratio > target_ratio:
        resized = _resize(clip, height=height)
    else:
        resized = _resize(clip, width=width)

    return _crop(resized, x_center=resized.w / 2, y_center=resized.h / 2, width=width, height=height)


def _apply_micro_zooms(
    clip: Any,
    pulse_every: float = MICRO_ZOOM_EVERY_SECONDS,
    pulse_seconds: float = MICRO_ZOOM_PULSE_SECONDS,
    strength: float = MICRO_ZOOM_STRENGTH,
) -> Any:
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


def _deterministic_unit_interval(seed_key: str) -> float:
    digest = hashlib.sha1(seed_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _micro_zoom_params_for_shot(shot: TimelineClip) -> tuple[bool, float, float, float]:
    # Pulse-style micro zooms are disabled by request to keep motion cleaner.
    return False, MICRO_ZOOM_EVERY_SECONDS, MICRO_ZOOM_PULSE_SECONDS, MICRO_ZOOM_STRENGTH

    shot_duration = max(0.0, float(shot.timeline_end - shot.timeline_start))
    if shot_duration < 1.15:
        return False, MICRO_ZOOM_EVERY_SECONDS, MICRO_ZOOM_PULSE_SECONDS, MICRO_ZOOM_STRENGTH

    key_base = f"{shot.source_path.as_posix()}|{shot.plan_idx}|{shot.timeline_start:.3f}|{shot.timeline_end:.3f}"
    include_roll = _deterministic_unit_interval(f"include|{key_base}")
    cycle_roll = _deterministic_unit_interval(f"cycle|{key_base}")
    strength_roll = _deterministic_unit_interval(f"strength|{key_base}")

    include_probability = 0.52
    if shot.emotion == "excitement":
        include_probability = 0.72
    elif shot.emotion == "suspense":
        include_probability = 0.43

    if shot.transition_type in {"whip", "zoom_in"}:
        include_probability += 0.10
    elif shot.transition_type == "fade":
        include_probability -= 0.18

    include_probability = max(0.25, min(0.82, include_probability))
    if include_roll > include_probability:
        return False, MICRO_ZOOM_EVERY_SECONDS, MICRO_ZOOM_PULSE_SECONDS, MICRO_ZOOM_STRENGTH

    cycle = MICRO_ZOOM_EVERY_SECONDS * (0.9 + 0.45 * cycle_roll)
    pulse_seconds = min(0.40, max(0.22, cycle * 0.18))

    strength_base = MICRO_ZOOM_STRENGTH
    if shot.emotion == "suspense":
        strength_base *= 0.75
    elif shot.emotion == "excitement":
        strength_base *= 1.10
    if shot.transition_type == "fade":
        strength_base *= 0.72

    strength = strength_base * (0.85 + 0.30 * strength_roll)
    strength = max(0.028, min(0.060, strength))
    return True, cycle, pulse_seconds, strength


def _pick_music_track(music_folder: Path | None) -> Path | None:
    if not music_folder or not music_folder.exists():
        return None
    if music_folder.is_file() and music_folder.suffix.lower() in AUDIO_EXTENSIONS:
        return music_folder
    tracks = [
        p
        for p in sorted(music_folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not tracks:
        return None
    return random.choice(tracks)


def _clip_rms_profile(clip: AudioClip, duration: float, step: float) -> list[float]:
    if clip is None or duration <= 0:
        return []
    samples = max(4, int(math.ceil(duration / max(0.03, step))))
    values: list[float] = []
    for i in range(samples + 1):
        t = min(duration, i * step)
        try:
            frame = np.array(clip.get_frame(t), dtype=np.float32)
        except Exception:
            values.append(0.0)
            continue
        if frame.ndim > 0:
            frame = np.abs(frame)
            amp = float(np.mean(frame))
        else:
            amp = abs(float(frame))
        values.append(max(0.0, amp))
    return values


def _smooth_series(values: list[float], radius: int = 2) -> list[float]:
    if not values:
        return values
    smoothed: list[float] = []
    n = len(values)
    for idx in range(n):
        lo = max(0, idx - radius)
        hi = min(n, idx + radius + 1)
        window = values[lo:hi]
        smoothed.append(float(sum(window)) / max(1, len(window)))
    return smoothed


def _series_at(values: list[float], t: float, step: float) -> float:
    if not values:
        return 0.0
    pos = max(0.0, float(t) / max(1e-6, step))
    lo = int(math.floor(pos))
    hi = min(len(values) - 1, lo + 1)
    if lo >= len(values) - 1:
        return float(values[-1])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _adaptive_music_mix(
    music_clip: AudioClip,
    voiceover_clip: AudioClip,
    final_duration: float,
) -> tuple[AudioClip, dict[str, float]]:
    voice_env = _clip_rms_profile(voiceover_clip, final_duration, MUSIC_ENVELOPE_STEP_SECONDS)
    music_env = _clip_rms_profile(music_clip, final_duration, MUSIC_ENVELOPE_STEP_SECONDS)
    if not voice_env or not music_env:
        mixed = music_clip.volumex(MUSIC_BASE_GAIN)
        return mixed, {
            "base_gain": MUSIC_BASE_GAIN,
            "duck_gain": MUSIC_DUCKED_GAIN,
            "voice_median": 0.0,
            "music_median": 0.0,
        }

    voice_s = _smooth_series(voice_env, radius=2)
    music_s = _smooth_series(music_env, radius=2)

    voice_median = float(np.median(voice_s))
    voice_p90 = float(np.percentile(voice_s, 90)) if voice_s else voice_median
    music_median = float(np.median(music_s))

    base_gain = float(MUSIC_BASE_GAIN)
    if voice_median > 1e-6 and music_median > 1e-6:
        safe_gain = MUSIC_MAX_VOICE_RATIO * (voice_median / music_median)
        base_gain = max(0.04, min(base_gain, safe_gain))

    duck_gain = max(0.022, min(base_gain * 0.52, MUSIC_DUCKED_GAIN))
    spread = max(1e-6, voice_p90 - voice_median)

    def gain_at(t: float) -> float:
        e = _series_at(voice_s, float(t), MUSIC_ENVELOPE_STEP_SECONDS)
        speech_presence = max(0.0, min(1.0, (e - voice_median) / spread))
        return base_gain - (base_gain - duck_gain) * speech_presence

    def gain_array(times: np.ndarray) -> np.ndarray:
        time_arr = np.asarray(times, dtype=np.float64)
        if time_arr.size == 0:
            return np.array([], dtype=np.float32)
        gains = np.empty(time_arr.shape, dtype=np.float32)
        flat = gains.reshape(-1)
        flat_t = time_arr.reshape(-1)
        for i, ts in enumerate(flat_t):
            flat[i] = gain_at(float(ts))
        return gains

    def duck_frame(gf, t: float):
        frame_arr = np.asarray(gf(t), dtype=np.float32)
        time_arr = np.asarray(t, dtype=np.float64)

        # Scalar time path (single frame)
        if time_arr.ndim == 0:
            return frame_arr * gain_at(float(time_arr))

        gains = gain_array(time_arr).reshape(-1)
        if gains.size == 0:
            return frame_arr

        # Typical vectorized audio shape: (num_times, channels)
        if frame_arr.ndim == 2 and frame_arr.shape[0] == gains.shape[0]:
            return frame_arr * gains[:, None]

        # Mono vectorized shape: (num_times,)
        if frame_arr.ndim == 1 and frame_arr.shape[0] == gains.shape[0]:
            return frame_arr * gains

        # Unknown shape: apply conservative scalar gain using first sample.
        return frame_arr * float(gains[0])

    mixed = music_clip.fl(duck_frame, keep_duration=True)
    return mixed, {
        "base_gain": float(base_gain),
        "duck_gain": float(duck_gain),
        "voice_median": float(voice_median),
        "music_median": float(music_median),
    }


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

    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_caption_font_bold(height: int, scale: float = 1.0) -> ImageFont.ImageFont:
    size = max(26, int(height * 0.032 * max(0.6, float(scale))))
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

    for name in ("arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return _load_caption_font(height, scale=scale)


def _apply_ken_burns(
    clip: Any,
    zoom_ratio: float = 0.08,
    pan_ratio: float = 0.08,
    seed_key: str = "",
) -> Any:
    """Apply a deterministic Ken Burns pan+zoom for still-image motion."""
    w, h = int(clip.w), int(clip.h)
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()

    def _axis_pair(a: int, b: int) -> tuple[float, float]:
        start = a / 255.0
        end = b / 255.0
        # Keep movement inside a safe central band to avoid edge-heavy crops.
        margin = max(0.0, min(0.35, (1.0 - float(pan_ratio)) * 0.5))
        start = margin + (1.0 - 2.0 * margin) * start
        end = margin + (1.0 - 2.0 * margin) * end
        return start, end

    x_start, x_end = _axis_pair(digest[0], digest[1])
    y_start, y_end = _axis_pair(digest[2], digest[3])

    def zoom_frame(gf, t: float):
        progress = min(t / max(float(clip.duration or 0.0), 1e-3), 1.0)
        scale = 1.0 + max(0.0, float(zoom_ratio)) * progress
        frame = gf(t)
        new_w = max(w, int(round(w * scale)))
        new_h = max(h, int(round(h * scale)))

        resized = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(resized)

        max_x = max(0, new_w - w)
        max_y = max(0, new_h - h)
        x_frac = x_start + (x_end - x_start) * progress
        y_frac = y_start + (y_end - y_start) * progress
        x0 = int(round(max(0.0, min(1.0, x_frac)) * max_x))
        y0 = int(round(max(0.0, min(1.0, y_frac)) * max_y))
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
            audio_parts.append(_subclip(source, prev_end, prev_end + keep_gap))
            cursor += keep_gap
        trimmed_gap_total += max(0.0, gap - keep_gap)

        part = _subclip(source, seg_start, seg_end)
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
        audio_parts.append(_subclip(source, prev_end, prev_end + tail))

    if not audio_parts:
        return source, subtitle_plan

    trimmed_audio = concatenate_audioclips(audio_parts)
    log(f"Trimmed silence between sentences: {trimmed_gap_total:.2f}s removed")
    # Do not close `source` here; concatenated subclips still read frames from it.
    return trimmed_audio, remapped


def _stable_transition_seed(transition_plan: list[TimelineClip]) -> int:
    payload = "|".join(
        f"{clip.source_path.as_posix()}:{float(clip.timeline_start):.3f}:{float(clip.timeline_end):.3f}:{clip.plan_idx}"
        for clip in transition_plan
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _build_weighted_transition_sequence(cut_count: int, rng: random.Random) -> list[str]:
    if cut_count <= 0:
        return []

    counts: dict[str, int] = {}
    remainders: list[tuple[float, float, str]] = []
    assigned = 0
    for name, weight in PRO_TRANSITION_WEIGHTS:
        raw = cut_count * weight
        whole = int(math.floor(raw))
        counts[name] = whole
        assigned += whole
        remainders.append((raw - whole, weight, name))

    remaining = max(0, cut_count - assigned)
    remainders.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for idx in range(remaining):
        counts[remainders[idx % len(remainders)][2]] += 1

    sequence: list[str] = []
    for name, _weight in PRO_TRANSITION_WEIGHTS:
        sequence.extend([name] * counts.get(name, 0))

    rng.shuffle(sequence)
    return sequence


def _center_crop_resize(frame: np.ndarray, scale: float) -> np.ndarray:
    height, width = frame.shape[:2]
    new_w = max(width, int(round(width * scale)))
    new_h = max(height, int(round(height * scale)))
    resized = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(resized)
    x0 = max(0, (new_w - width) // 2)
    y0 = max(0, (new_h - height) // 2)
    return arr[y0 : y0 + height, x0 : x0 + width]


def _blend_with_color(frame: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    clamped = max(0.0, min(1.0, float(alpha)))
    target = np.empty_like(frame, dtype=np.float32)
    target[..., 0] = color[0]
    target[..., 1] = color[1]
    target[..., 2] = color[2]
    mixed = frame.astype(np.float32) * (1.0 - clamped) + target * clamped
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _apply_rgb_split(frame: np.ndarray, offset: int) -> np.ndarray:
    shifted = np.empty_like(frame)
    shifted[..., 0] = np.roll(frame[..., 0], offset, axis=1)
    shifted[..., 1] = frame[..., 1]
    shifted[..., 2] = np.roll(frame[..., 2], -offset, axis=1)
    shifted[::2, :, :] = np.clip(shifted[::2, :, :] * 0.92, 0, 255).astype(np.uint8)
    return shifted


def _apply_horizontal_motion_blur(frame: np.ndarray, offset: int) -> np.ndarray:
    samples = [frame.astype(np.float32)]
    for factor in (0.25, 0.5, 0.75, 1.0):
        shift = int(round(offset * factor))
        samples.append(np.roll(frame, shift, axis=1).astype(np.float32))
        samples.append(np.roll(frame, -shift, axis=1).astype(np.float32))
    return np.clip(np.mean(samples, axis=0), 0, 255).astype(np.uint8)


def _apply_tail_frames(
    clip: Any,
    fps: int,
    frame_count: int,
    transform,
) -> Any:
    duration = float(clip.duration or 0.0)
    if duration <= 0.0 or frame_count <= 0:
        return clip

    frame_count = max(1, int(frame_count))
    window = min(duration, frame_count / max(1, int(fps)))
    start_t = max(0.0, duration - window)

    def _frame(gf, t: float):
        frame = np.array(gf(t), copy=False)
        if t < start_t:
            return frame
        local = max(0.0, t - start_t)
        index = min(frame_count - 1, int(local * fps + 1e-6))
        progress = (index + 1) / frame_count
        return transform(frame, index, frame_count, progress)

    return clip.fl(_frame, keep_duration=True)


def _apply_head_frames(
    clip: Any,
    fps: int,
    frame_count: int,
    transform,
) -> Any:
    duration = float(clip.duration or 0.0)
    if duration <= 0.0 or frame_count <= 0:
        return clip

    frame_count = max(1, int(frame_count))
    window = min(duration, frame_count / max(1, int(fps)))

    def _frame(gf, t: float):
        frame = np.array(gf(t), copy=False)
        if t >= window:
            return frame
        index = min(frame_count - 1, int(t * fps + 1e-6))
        progress = (index + 1) / frame_count
        return transform(frame, index, frame_count, progress)

    return clip.fl(_frame, keep_duration=True)


def _apply_zoom_punch_outgoing(clip: Any, fps: int) -> Any:
    return _apply_tail_frames(
        clip,
        fps=fps,
        frame_count=6,
        transform=lambda frame, _idx, _total, progress: _center_crop_resize(frame, 1.0 + 0.20 * progress),
    )


def _apply_whip_pan_outgoing(clip: Any, fps: int, direction: int) -> Any:
    return _apply_tail_frames(
        clip,
        fps=fps,
        frame_count=4,
        transform=lambda frame, _idx, _total, progress: _apply_horizontal_motion_blur(
            frame,
            max(8, int(round(frame.shape[1] * (0.015 + 0.03 * progress)))) * (1 if direction >= 0 else -1),
        ),
    )


def _apply_smash_cut_incoming(clip: Any, fps: int) -> Any:
    return _apply_head_frames(
        clip,
        fps=fps,
        frame_count=1,
        transform=lambda frame, _idx, _total, _progress: np.full_like(frame, 255, dtype=np.uint8),
    )


def _apply_fade_black_pair(
    outgoing: Any,
    incoming: Any,
    fps: int,
    frame_count: int,
) -> tuple[Any, Any]:
    outgoing_fx = _apply_tail_frames(
        outgoing,
        fps=fps,
        frame_count=frame_count,
        transform=lambda frame, _idx, _total, progress: _blend_with_color(frame, (0, 0, 0), progress),
    )
    incoming_fx = _apply_head_frames(
        incoming,
        fps=fps,
        frame_count=frame_count,
        transform=lambda frame, _idx, _total, progress: _blend_with_color(frame, (0, 0, 0), 1.0 - progress),
    )
    return outgoing_fx, incoming_fx


def _compose_with_adaptive_transitions(
    clips: list[Any],
    transition_plan: list[TimelineClip],
    style: str,
    fps: int,
    size: tuple[int, int],
) -> tuple[Any, int, dict[str, int]]:
    """Build a frame-accurate professional cut timeline from the plan."""
    if not clips:
        raise ValueError("No clips to compose")

    if style not in TRANSITION_STYLES:
        style = "pro_weighted"

    prepared = list(clips)
    if style == "none" or len(prepared) == 1:
        return concatenate_videoclips(prepared, method="compose"), 0, {}

    active_cut_indices = [
        idx
        for idx in range(len(prepared) - 1)
        if idx < len(transition_plan) and bool(transition_plan[idx].transition_after)
    ]
    if not active_cut_indices:
        return concatenate_videoclips(prepared, method="compose"), 0, {}

    rng = random.Random(_stable_transition_seed(transition_plan))
    sequence = _build_weighted_transition_sequence(len(active_cut_indices), rng)
    counts: dict[str, int] = {name: 0 for name, _weight in PRO_TRANSITION_WEIGHTS}

    for cut_order, cut_idx in enumerate(active_cut_indices):
        effect = sequence[cut_order]
        counts[effect] = counts.get(effect, 0) + 1
        direction = -1 if (cut_idx % 2) else 1

        if effect == "zoom_punch":
            prepared[cut_idx] = _apply_zoom_punch_outgoing(prepared[cut_idx], fps=fps)
        elif effect == "whip_pan":
            prepared[cut_idx] = _apply_whip_pan_outgoing(prepared[cut_idx], fps=fps, direction=direction)
        elif effect == "smash_cut":
            prepared[cut_idx + 1] = _apply_smash_cut_incoming(prepared[cut_idx + 1], fps=fps)
        elif effect == "fade_black":
            fade_frames = rng.choice((2, 3, 4))
            prepared[cut_idx], prepared[cut_idx + 1] = _apply_fade_black_pair(
                prepared[cut_idx],
                prepared[cut_idx + 1],
                fps=fps,
                frame_count=fade_frames,
            )

    final_clip = concatenate_videoclips(prepared, method="compose")
    return final_clip, len(active_cut_indices), {k: v for k, v in counts.items() if v > 0}


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
    base_video: Any | None,
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

    def _draw_gradient_word(
        canvas: Image.Image,
        text_draw: ImageDraw.ImageDraw,
        word: str,
        x: int,
        y: int,
    ) -> None:
        # Draw stroke first, then fill glyph with a vertical gradient masked to the word shape.
        text_draw.text(
            (x, y),
            word,
            font=font,
            fill=(255, 255, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 230),
        )
        bbox = text_draw.textbbox((x, y), word, font=font)
        bw = max(1, int(bbox[2] - bbox[0]))
        bh = max(1, int(bbox[3] - bbox[1]))

        gradient = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(gradient)
        top = preset.get("gradient_top", (255, 116, 255, 255))
        bottom = preset.get("gradient_bottom", (102, 225, 255, 255))
        denom = max(1, bh - 1)
        for row in range(bh):
            mix = row / denom
            color = (
                int(top[0] + (bottom[0] - top[0]) * mix),
                int(top[1] + (bottom[1] - top[1]) * mix),
                int(top[2] + (bottom[2] - top[2]) * mix),
                255,
            )
            gdraw.line((0, row, bw, row), fill=color, width=1)

        mask = Image.new("L", (bw, bh), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.text(
            (-int(bbox[0] - x), -int(bbox[1] - y)),
            word,
            font=font,
            fill=255,
        )
        canvas.paste(gradient, (int(bbox[0]), int(bbox[1])), mask)

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
        use_gradient_text = bool(preset.get("gradient_text", False))
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
                    fill = preset.get("active_highlight", preset["highlight"])

                if use_gradient_text and word_cursor != active_idx:
                    _draw_gradient_word(text_canvas, text_draw, word, x, y)
                else:
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
    base_video: Any | None = None,
    enable_adaptive_caption_safe_zones: bool = True,
    enable_karaoke_highlight: bool = True,
) -> list[Any]:
    if not subtitle_plan:
        return []

    style_alias = {
        "beast": "bold_stroke",
        "clean": "yellow_active",
        "kinetic": "gradient_fill",
    }
    resolved_style = style_alias.get((caption_style or "").strip().lower(), caption_style)
    preset = dict(CAPTION_STYLE_PRESETS.get(resolved_style, CAPTION_STYLE_PRESETS["bold_stroke"]))
    if caption_position_ratio is not None:
        preset["position_ratio"] = max(0.35, min(float(caption_position_ratio), 0.85))
    if caption_max_lines is not None:
        preset["max_lines"] = max(1, min(int(caption_max_lines), 6))

    min_box_height = max(180, int(height * 0.16))
    max_box_height = max(min_box_height, int(height * 0.65))
    max_text_width = int(width * 0.84)
    overlays: list[Any] = []
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
        clip = clip.crossfadeout(CAPTION_OUT_FADE_SECONDS)

        # CapCut-like caption entrance: 0.8 -> 1.0 scale over 80ms.
        _cw, _ch = int(width), int(box_height)

        def _pop_in_frame(gf, t: float, cw: int = _cw, ch: int = _ch) -> np.ndarray:
            if t >= CAPTION_IN_POP_SECONDS:
                return gf(t)
            ratio = max(0.0, min(1.0, t / max(1e-4, CAPTION_IN_POP_SECONDS)))
            scale = 0.8 + (0.2 * ratio)
            new_w = max(1, int(round(cw * scale)))
            new_h = max(1, int(round(ch * scale)))
            resized = Image.fromarray(gf(t)).resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
            x0 = (cw - new_w) // 2
            y0 = (ch - new_h) // 2
            canvas.paste(resized, (x0, y0))
            return np.array(canvas)

        def _pop_in_mask(gf, t: float, cw: int = _cw, ch: int = _ch) -> np.ndarray:
            if t >= CAPTION_IN_POP_SECONDS:
                return gf(t)
            ratio = max(0.0, min(1.0, t / max(1e-4, CAPTION_IN_POP_SECONDS)))
            scale = 0.8 + (0.2 * ratio)
            new_w = max(1, int(round(cw * scale)))
            new_h = max(1, int(round(ch * scale)))
            resized = Image.fromarray((gf(t) * 255.0).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("L", (cw, ch), 0)
            x0 = (cw - new_w) // 2
            y0 = (ch - new_h) // 2
            canvas.paste(resized, (x0, y0))
            return np.array(canvas).astype(np.float32) / 255.0

        clip = clip.fl(_pop_in_frame, keep_duration=True)
        clip = clip.set_mask(clip.mask.fl(_pop_in_mask, keep_duration=True))
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
    enable_motion_overlays: bool = True,
    hook_text_override: str = "",
    stat_badge_text: str = "",
    cta_text: str = "Learn More",
    logo_path: Path | None = None,
    enable_progress_bar: bool = True,
) -> None:
    if not timeline_clips:
        raise ValueError("No timeline clips available. Add clips in the clips folder.")

    assembled = []
    opened_video_clips: list[Any] = []
    subtitle_clips: list[Any] = []
    overlay_clips: list[Any] = []
    voiceover_clip: AudioClip | None = None
    music_clip: Any | None = None
    final_video = None

    try:
        image_motion_count = 0
        micro_zoom_count = 0
        micro_zoom_strength_total = 0.0
        for shot in timeline_clips:
            needed = max(0.2, float(shot.timeline_end - shot.timeline_start))
            if shot.source_path.suffix.lower() in IMAGE_EXTENSIONS:
                clip = ImageClip(str(shot.source_path)).set_duration(needed)
                clip = _apply_ken_burns(
                    clip,
                    zoom_ratio=0.10,
                    pan_ratio=0.10,
                    seed_key=f"{shot.source_path.as_posix()}|{shot.timeline_start:.3f}|{shot.timeline_end:.3f}",
                )
                image_motion_count += 1
            else:
                clip = VideoFileClip(str(shot.source_path))
            opened_video_clips.append(clip)

            if clip.duration is not None and clip.duration <= 0:
                continue

            if shot.source_path.suffix.lower() in IMAGE_EXTENSIONS:
                base_clip = clip
            else:
                source_start = max(0.0, float(shot.source_start or 0.0))
                source_end = float(shot.source_end) if shot.source_end is not None else None
                clip_duration = float(clip.duration or 0.0)
                if clip_duration > 0 and source_start >= clip_duration:
                    source_start = max(0.0, clip_duration - 0.25)
                if source_end is not None and clip_duration > 0:
                    source_end = min(max(source_start + 0.1, source_end), clip_duration)

                if source_end is not None and source_end > source_start:
                    base_clip = _subclip(clip, source_start, source_end)
                elif source_start > 0 and clip_duration > source_start:
                    base_clip = _subclip(clip, source_start, clip_duration)
                else:
                    base_clip = clip

            if base_clip.duration is not None and base_clip.duration < needed:
                repeats = int(needed // base_clip.duration) + 1
                extended = _subclip(concatenate_videoclips([base_clip] * repeats), 0, needed)
            elif base_clip.duration is not None:
                extended = _subclip(base_clip, 0, needed)
            else:
                extended = base_clip.set_duration(needed)

            fitted = _fit_clip_to_canvas(extended, width=width, height=height)
            should_zoom, cycle, pulse_seconds, strength = _micro_zoom_params_for_shot(shot)
            if should_zoom:
                fitted = _apply_micro_zooms(
                    fitted,
                    pulse_every=cycle,
                    pulse_seconds=pulse_seconds,
                    strength=strength,
                )
                micro_zoom_count += 1
                micro_zoom_strength_total += strength
            assembled.append(fitted)

        if not assembled:
            raise ValueError("Failed to assemble output timeline.")
        if image_motion_count > 0:
            log(f"Ken Burns motion applied to {image_motion_count} image shots")

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

        log("Applying weighted professional transitions")
        final_video, transition_count, transition_counts = _compose_with_adaptive_transitions(
            clips=assembled,
            transition_plan=timeline_clips,
            style=transition_style,
            fps=fps,
            size=(width, height),
        )
        log(f"Transitions applied: {transition_count}")
        if transition_counts:
            detail = ", ".join(f"{name}={count}" for name, count in sorted(transition_counts.items()))
            log(f"Transition mix: {detail}")

        if micro_zoom_count > 0:
            avg_strength = micro_zoom_strength_total / micro_zoom_count
            log(
                f"Micro-zooms enabled on {micro_zoom_count}/{len(timeline_clips)} shots "
                f"(adaptive cycle around {MICRO_ZOOM_EVERY_SECONDS:.2f}s, avg strength {avg_strength:.3f})"
            )
        else:
            log("Micro-zooms skipped: scene pacing favored stable framing")

        voice_duration = float(voiceover_clip.duration)
        if float(final_video.duration) < voice_duration:
            # Prevent voice truncation by freezing the last frame as needed.
            missing = voice_duration - float(final_video.duration)
            hold = final_video.to_ImageClip(t=max(0.0, float(final_video.duration) - 0.05)).set_duration(missing)
            final_video = concatenate_videoclips([final_video, hold], method="compose")

        final_duration = voice_duration
        log(f"Voiceover duration: {voice_duration:.2f}s | Final duration: {final_duration:.2f}s")
        final_video = _subclip(final_video, 0, final_duration)

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

        overlay_clips = []
        if enable_motion_overlays:
            overlay_clips = create_motion_graphics_overlays(
                subtitle_plan=subtitle_plan,
                width=width,
                height=height,
                final_duration=final_duration,
                hook_text_override=hook_text_override,
                stat_badge_text=stat_badge_text,
                cta_text=cta_text,
                logo_path=logo_path,
                enable_progress_bar=enable_progress_bar,
            )
            log(f"Motion graphic overlays: {len(overlay_clips)}")

        if subtitle_clips or overlay_clips:
            final_video = CompositeVideoClip([final_video, *subtitle_clips, *overlay_clips], size=(width, height))

        tracks = [voiceover_clip.volumex(1.0)]
        music_track = _pick_music_track(music_folder)
        if music_track:
            log(f"Selected music track: {music_track.name}")
            music_clip = AudioFileClip(str(music_track))
            music_looped = audio_loop(music_clip, duration=final_duration)
            try:
                mixed_music, mix_stats = _adaptive_music_mix(
                    music_clip=music_looped,
                    voiceover_clip=voiceover_clip,
                    final_duration=final_duration,
                )
                log(
                    "Music blend: adaptive ducking "
                    f"(base={mix_stats['base_gain']:.3f}, ducked={mix_stats['duck_gain']:.3f})"
                )
            except Exception as exc:
                log(f"Music blend fallback: adaptive ducking failed ({exc}); using static background gain")
                mixed_music = music_looped.volumex(0.10)

            fade_in = min(MUSIC_FADE_IN_SECONDS, max(0.06, final_duration * 0.20))
            fade_out = min(MUSIC_FADE_OUT_SECONDS, max(0.06, final_duration * 0.24))
            if (fade_in + fade_out) > (final_duration * 0.90):
                scale = (final_duration * 0.90) / max(1e-6, (fade_in + fade_out))
                fade_in *= scale
                fade_out *= scale

            mixed_music = audio_fadein(mixed_music, fade_in)
            mixed_music = audio_fadeout(mixed_music, fade_out)
            log(f"Music fades: in={fade_in:.2f}s, out={fade_out:.2f}s")
            tracks.append(mixed_music)

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
        for c in overlay_clips:
            c.close()
        for c in opened_video_clips:
            c.close()
