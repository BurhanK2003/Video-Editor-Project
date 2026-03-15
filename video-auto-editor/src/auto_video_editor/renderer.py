from __future__ import annotations

import math
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.audio.fx.all import audio_loop
from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips

from .models import PlannedSegment, TimelineClip

TRANSITION_STYLES = ("none", "crossfade", "zoom", "fade_black")
SEGMENT_TRANSITIONS = ("jump_cut", "zoom_in", "whip", "fade")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
MICRO_ZOOM_EVERY_SECONDS = 2.35
MICRO_ZOOM_PULSE_SECONDS = 0.46
MICRO_ZOOM_STRENGTH = 0.055


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
    width, height = clip.w, clip.h

    def scale_at_time(t: float) -> float:
        phase = float(t) % cycle
        if phase > pulse_seconds:
            return 1.0
        pulse_pos = phase / max(0.05, pulse_seconds)
        envelope = math.sin(math.pi * pulse_pos)
        return 1.0 + max(0.0, float(strength)) * envelope

    animated = clip.resize(lambda t: scale_at_time(t))
    return animated.crop(x_center=animated.w / 2, y_center=animated.h / 2, width=width, height=height)


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


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
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
    return "\n".join(lines[:3])


def _load_caption_font(height: int) -> ImageFont.ImageFont:
    size = max(28, int(height * 0.03))
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _load_caption_font_bold(height: int) -> ImageFont.ImageFont:
    size = max(30, int(height * 0.032))
    for name in ("arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return _load_caption_font(height)


def _apply_ken_burns(clip: VideoFileClip, zoom_ratio: float = 0.06) -> VideoFileClip:
    """Slowly zoom into the clip over its duration (Ken Burns effect).

    The clip is centre-cropped so the output dimensions never change.
    *zoom_ratio* is the fractional size increase at the end (0.06 = 6%).
    """
    w, h = clip.w, clip.h

    def zoom_frame(gf, t: float):
        progress = min(t / max(clip.duration, 1e-3), 1.0)
        scale = 1.0 + zoom_ratio * progress
        frame = gf(t)
        new_w = max(w, int(w * scale))
        new_h = max(h, int(h * scale))
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
                clip = clip.resize(lambda t: 1.03 + 0.05 * math.exp(-16.0 * t)).crossfadein(whip_dur)
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


def _subtitle_overlays(
    subtitle_plan: list[PlannedSegment],
    width: int,
    height: int,
    final_duration: float,
) -> list[ImageClip]:
    if not subtitle_plan:
        return []

    box_height = max(200, int(height * 0.22))
    max_text_width = int(width * 0.84)
    overlays: list[ImageClip] = []

    for segment in subtitle_plan:
        start = max(0.0, float(segment.start))
        end = min(final_duration, float(segment.end))
        if end <= start:
            continue
        text = (segment.text or "").strip()
        if not text:
            continue

        image = Image.new("RGBA", (width, box_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = _load_caption_font_bold(height) if segment.emphasis else _load_caption_font(height)

        wrapped = _wrap_text(draw, text, font, max_text_width)
        if not wrapped:
            continue

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=6)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x = 28
        pad_y = 18
        rect_w = min(width - 20, text_w + pad_x * 2)
        rect_h = text_h + pad_y * 2
        rect_x0 = (width - rect_w) // 2
        rect_y0 = (box_height - rect_h) // 2
        rect_x1 = rect_x0 + rect_w
        rect_y1 = rect_y0 + rect_h

        # Layered rounded cards create a more polished subtitle style.
        draw.rounded_rectangle(
            (rect_x0 + 5, rect_y0 + 5, rect_x1 + 5, rect_y1 + 5),
            radius=14,
            fill=(0, 0, 0, 90),
        )
        base_fill = (20, 20, 20, 190)
        if segment.emphasis:
            base_fill = (30, 24, 10, 208)
        draw.rounded_rectangle((rect_x0, rect_y0, rect_x1, rect_y1), radius=14, fill=base_fill)

        accent_h = max(6, int(rect_h * 0.12))
        accent_color = (255, 196, 0, 220) if not segment.emphasis else (255, 223, 92, 235)
        draw.rounded_rectangle((rect_x0, rect_y0, rect_x1, rect_y0 + accent_h), radius=14, fill=accent_color)

        text_x = (width - text_w) // 2
        text_y = (box_height - text_h) // 2
        text_fill = (255, 255, 255, 255)
        if segment.highlight_phrase:
            wrapped = wrapped.replace(segment.highlight_phrase, segment.highlight_phrase.upper())
            text_fill = (255, 244, 179, 255)

        draw.multiline_text(
            (text_x, text_y),
            wrapped,
            font=font,
            fill=text_fill,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 230),
            align="center",
            spacing=6,
        )

        arr = np.array(image)
        clip = (
            ImageClip(arr)
            .set_start(start)
            .set_end(end)
            .set_position(("center", int(height * 0.60)))
        )
        clip = clip.crossfadein(0.12).crossfadeout(0.10)
        if segment.emphasis:
            clip = clip.resize(lambda t: 1.0 + 0.05 * math.exp(-8.0 * t))
        overlays.append(clip)

    return overlays


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
) -> None:
    if not timeline_clips:
        raise ValueError("No timeline clips available. Add clips in the clips folder.")

    assembled = []
    opened_video_clips: list[VideoFileClip] = []
    subtitle_clips: list[ImageClip] = []
    voiceover_clip: AudioClip | None = None
    music_clip: AudioFileClip | None = None
    final_video = None

    try:
        for shot in timeline_clips:
            needed = max(0.2, float(shot.timeline_end - shot.timeline_start))
            clip = VideoFileClip(str(shot.source_path))
            opened_video_clips.append(clip)

            if clip.duration <= 0:
                continue

            if clip.duration < needed:
                repeats = int(needed // clip.duration) + 1
                extended = concatenate_videoclips([clip] * repeats).subclip(0, needed)
            else:
                extended = clip.subclip(0, needed)

            fitted = _fit_clip_to_canvas(extended, width=width, height=height)
            assembled.append(fitted)

        if not assembled:
            raise ValueError("Failed to assemble output timeline.")

        voiceover_clip, subtitle_plan = _build_trimmed_voiceover(voiceover_path, subtitle_plan, log)
        subtitle_plan = _sync_plan_to_voice_energy(subtitle_plan, voiceover_clip, log)

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
