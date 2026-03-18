from __future__ import annotations

from .matcher import assign_clips, find_low_confidence_segments, list_video_clips
from .models import AutoEditRequest
from .planner import build_plan
from .quality import build_export_quality_report, write_export_quality_report
from .renderer import render_video
from .rhythm import apply_rhythm_sync, resolve_music_track
from .stock_fetcher import fetch_stock_clips
from .transcribe import transcribe_voiceover


def run_auto_edit(request: AutoEditRequest, log: callable) -> None:
    log("Starting auto-edit pipeline...")

    # Script-to-video: generate voiceover from text if no audio file was provided.
    voiceover_path = request.voiceover_path
    if request.script_text.strip() and (not voiceover_path or not voiceover_path.exists()):
        from .tts import generate_voiceover
        tts_out = request.output_path.with_suffix(".tts.wav")
        log("Script-to-video: generating voiceover with free TTS...")
        voiceover_path = generate_voiceover(
            text=request.script_text,
            output_path=tts_out,
            voice=request.script_voice or "",
            log=log,
        )
        log(f"Voiceover generated: {voiceover_path}")
    segments = transcribe_voiceover(
        voiceover_path=voiceover_path,
        model_size=request.whisper_model,
        log=log,
    )
    log(f"Transcript segments: {len(segments)}")

    plan = build_plan(segments, log=log)
    log(f"Subtitle plan segments: {len(plan)}")
    local_clips = list_video_clips(request.clips_folder)
    clips = list(local_clips)
    if request.allow_stock_fetch:
        stock_plan = plan
        if local_clips:
            log(f"Local source clips found: {len(local_clips)}. Checking local scene coverage before stock download.")
            stock_plan = find_low_confidence_segments(
                plan=plan,
                clip_paths=local_clips,
                log=log,
                min_best_score=0.60,
                max_segments=10,
            )
            if stock_plan:
                log(f"Local pool looks weak for {len(stock_plan)} scenes. Fetching targeted stock for those scenes.")
            else:
                log("Local pool already matches planned scenes well. Skipping stock download.")
        else:
            log("No local clips found. Attempting stock footage download.")

        if stock_plan:
            stock_clips = fetch_stock_clips(
                plan=stock_plan,
                output_path=request.output_path,
                width=request.output_width,
                height=request.output_height,
                keywords_override=request.stock_keywords,
                log=log,
            )
            if stock_clips:
                seen = {p.resolve() for p in clips}
                appended = 0
                for path in stock_clips:
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    clips.append(path)
                    appended += 1
                log(f"Stock clips downloaded and merged: {appended}")
    if not clips:
        raise ValueError(
            "No source clips available: no local clips were found and stock clips could not be fetched. "
            "Add local clips, enable stock fetch, and optionally set PEXELS_API_KEY / PIXABAY_API_KEY in .env."
        )
    log(f"Available source clips for matching: {len(clips)}")

    timeline = assign_clips(plan, clips, log=log)
    selected_music_track = resolve_music_track(
        request.music_folder,
        voiceover_path=voiceover_path,
        log=log,
    )
    timeline = apply_rhythm_sync(
        timeline=timeline,
        voiceover_path=request.voiceover_path,
        music_path=selected_music_track,
        log=log,
    )
    log(f"Timeline shots: {len(timeline)}")

    render_video(
        timeline_clips=timeline,
        subtitle_plan=plan,
        voiceover_path=voiceover_path,
        output_path=request.output_path,
        width=request.output_width,
        height=request.output_height,
        fps=request.fps,
        render_preset=request.render_preset,
        music_folder=selected_music_track,
        log=log,
        transition_style=request.transition_style,
        transition_duration=request.transition_duration,
        caption_style=request.caption_style,
        caption_position_ratio=request.caption_position_ratio,
        caption_max_lines=request.caption_max_lines,
        caption_font_scale=request.caption_font_scale,
        caption_pop_scale=request.caption_pop_scale,
        enable_adaptive_caption_safe_zones=request.enable_adaptive_caption_safe_zones,
        enable_karaoke_highlight=request.enable_karaoke_highlight,
        enable_motion_overlays=request.enable_motion_overlays,
        hook_text_override=request.hook_text_override,
        stat_badge_text=request.stat_badge_text,
        cta_text=request.cta_text,
        logo_path=request.logo_path,
        enable_progress_bar=request.enable_progress_bar,
    )

    report = build_export_quality_report(
        timeline_clips=timeline,
        subtitle_plan=plan,
        output_path=request.output_path,
    )
    report_json, report_md = write_export_quality_report(request.output_path, report)

    # Echo gate results to the UI log so problems are visible without opening files.
    summary = report.get("summary", {})
    log(
        f"Quality gates: {summary.get('pass_count', 0)} passed, "
        f"{summary.get('fail_count', 0)} failed"
    )
    for gate_name, gate_data in report.get("checks", {}).items():
        status = (
            "PASS" if gate_data.get("pass") is True
            else ("FAIL" if gate_data.get("pass") is False else "N/A ")
        )
        log(
            f"  [{status}] {gate_name} "
            f"— target: {gate_data.get('target')}, actual: {gate_data.get('actual')}"
        )
    if not summary.get("all_passed"):
        log("WARNING: one or more quality gates failed — see quality report for details.")

    log(f"Quality report (JSON): {report_json}")
    log(f"Quality report (Markdown): {report_md}")
