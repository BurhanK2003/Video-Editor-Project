from __future__ import annotations

from .matcher import assign_clips, list_video_clips
from .models import AutoEditRequest
from .planner import build_plan
from .renderer import render_video
from .stock_fetcher import fetch_stock_clips
from .transcribe import transcribe_voiceover


def run_auto_edit(request: AutoEditRequest, log: callable) -> None:
    log("Starting auto-edit pipeline...")
    segments = transcribe_voiceover(
        voiceover_path=request.voiceover_path,
        model_size=request.whisper_model,
        log=log,
    )
    log(f"Transcript segments: {len(segments)}")

    plan = build_plan(segments)
    clips = list_video_clips(request.clips_folder)
    if not clips and request.allow_stock_fetch:
        log("No local clips found. Attempting stock footage download.")
        clips = fetch_stock_clips(
            plan=plan,
            output_path=request.output_path,
            width=request.output_width,
            height=request.output_height,
            keywords_override=request.stock_keywords,
            log=log,
        )
    if not clips:
        raise ValueError(
            "No video clips found locally, and no stock clips could be fetched. "
            "Add local clips or set PEXELS_API_KEY / PIXABAY_API_KEY in the environment or a .env file."
        )
    log(f"Available source clips: {len(clips)}")

    timeline = assign_clips(plan, clips, log=log)
    log(f"Timeline shots: {len(timeline)}")

    render_video(
        timeline_clips=timeline,
        subtitle_plan=plan,
        voiceover_path=request.voiceover_path,
        output_path=request.output_path,
        width=request.output_width,
        height=request.output_height,
        fps=request.fps,
        render_preset=request.render_preset,
        music_folder=request.music_folder,
        log=log,
        transition_style=request.transition_style,
        transition_duration=request.transition_duration,
    )
