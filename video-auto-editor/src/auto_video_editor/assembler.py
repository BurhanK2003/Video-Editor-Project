from __future__ import annotations

from .clip_searcher import SelectedClip
from .models import PlannedSegment, TimelineClip
from .renderer import render_video


def build_timeline_from_selected(plan: list[PlannedSegment], picks: list[SelectedClip]) -> list[TimelineClip]:
    timeline: list[TimelineClip] = []
    cursor = 0.0

    count = min(len(plan), len(picks))
    for idx in range(count):
        seg = plan[idx]
        pick = picks[idx]

        dur = max(0.35, float(seg.end - seg.start))
        clip = TimelineClip(
            source_path=pick.source_path,
            timeline_start=cursor,
            timeline_end=cursor + dur,
            source_start=float(pick.window_start),
            source_end=float(pick.window_end),
            transition_after=seg.transition_after,
            transition_seconds=seg.transition_seconds,
            transition_type=seg.transition_type,
            emotion=seg.emotion,
            plan_idx=idx,
        )
        timeline.append(clip)
        cursor += dur

    return timeline


def render_timeline(
    timeline: list[TimelineClip],
    subtitle_plan: list[PlannedSegment],
    voiceover_path,
    output_path,
    width: int,
    height: int,
    fps: int,
    render_preset: str,
    music_track,
    transition_style: str,
    transition_duration: float,
    caption_style: str,
    caption_position_ratio: float | None,
    caption_max_lines: int | None,
    caption_font_scale: float,
    caption_pop_scale: float,
    enable_adaptive_caption_safe_zones: bool,
    enable_karaoke_highlight: bool,
    enable_motion_overlays: bool,
    hook_text_override: str,
    stat_badge_text: str,
    cta_text: str,
    logo_path,
    enable_progress_bar: bool,
    log: callable,
) -> None:
    render_video(
        timeline_clips=timeline,
        subtitle_plan=subtitle_plan,
        voiceover_path=voiceover_path,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        render_preset=render_preset,
        music_folder=music_track,
        log=log,
        transition_style=transition_style,
        transition_duration=transition_duration,
        caption_style=caption_style,
        caption_position_ratio=caption_position_ratio,
        caption_max_lines=caption_max_lines,
        caption_font_scale=caption_font_scale,
        caption_pop_scale=caption_pop_scale,
        enable_adaptive_caption_safe_zones=enable_adaptive_caption_safe_zones,
        enable_karaoke_highlight=enable_karaoke_highlight,
        enable_motion_overlays=enable_motion_overlays,
        hook_text_override=hook_text_override,
        stat_badge_text=stat_badge_text,
        cta_text=cta_text,
        logo_path=logo_path,
        enable_progress_bar=enable_progress_bar,
    )
