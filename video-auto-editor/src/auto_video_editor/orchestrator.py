from __future__ import annotations

import os
from pathlib import Path

from .assembler import build_timeline_from_selected, render_timeline
from .audio_analyzer import analyze_audio
from .clip_searcher import select_best_clips
from .matcher import assign_clips, find_low_confidence_segments, list_video_clips
from .models import AutoEditRequest
from .planner import build_plan
from .quality import build_export_quality_report, write_export_quality_report
from .renderer import render_video
from .rhythm import apply_rhythm_sync, resolve_music_track
from .stock_fetcher import fetch_stock_clips
from .transition_mapper import apply_energy_and_beat_mapping
from .transcribe import transcribe_voiceover


def _read_setting(name: str) -> str:
    direct = os.getenv(name, "").strip()
    if direct:
        return direct

    current = os.getcwd()
    for folder in [Path(current), *Path(current).parents]:
        env_file = folder / ".env"
        if not env_file.exists():
            continue
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ""


def _runtime_tuning(
    request: AutoEditRequest,
    *,
    transcript_segments: int,
    local_clip_count: int,
) -> dict[str, int | bool]:
    if not request.auto_optimize_runtime:
        return {
            "ollama_scene_budget": max(0, int(request.ollama_scene_budget)),
            "enable_ollama_critic": bool(request.enable_ollama_critic),
            "match_scene_shortlist": max(8, int(request.match_scene_shortlist)),
        }

    cpu_count = max(1, int(os.cpu_count() or 4))
    clip_load = max(0, int(local_clip_count))
    seg_load = max(1, int(transcript_segments))

    # Start with user intent and constrain by workload.
    budget = max(0, int(request.ollama_scene_budget))
    shortlist = max(8, int(request.match_scene_shortlist))
    critic = bool(request.enable_ollama_critic)

    if clip_load >= 450:
        shortlist = min(shortlist, 18)
        budget = min(budget, 6)
    elif clip_load >= 250:
        shortlist = min(shortlist, 22)
        budget = min(budget, 8)
    elif clip_load >= 120:
        shortlist = min(shortlist, 26)
        budget = min(budget, 10)

    if seg_load >= 42:
        budget = min(budget, 6)
        critic = False
    elif seg_load >= 30:
        budget = min(budget, 8)
        critic = critic and True

    if cpu_count <= 4:
        budget = max(2, budget - 2) if budget > 0 else 0
        shortlist = max(10, shortlist - 4)
        critic = False if seg_load >= 24 else critic
    elif cpu_count >= 12 and seg_load <= 24 and clip_load <= 220:
        budget = min(12, max(budget, 9))
        shortlist = min(36, max(shortlist, 28))

    return {
        "ollama_scene_budget": max(0, int(budget)),
        "enable_ollama_critic": bool(critic),
        "match_scene_shortlist": max(8, int(shortlist)),
    }


def _use_advanced_clip_search(
    request: AutoEditRequest,
    *,
    transcript_segments: int,
    local_clip_count: int,
) -> bool:
    # Keep user override: when auto runtime optimization is disabled, try advanced path.
    if not request.auto_optimize_runtime:
        return True

    cpu_count = max(1, int(os.cpu_count() or 4))
    if cpu_count <= 6 and local_clip_count >= 90 and transcript_segments >= 14:
        return False
    if cpu_count <= 8 and local_clip_count >= 150:
        return False
    return True


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
    selected_music_track = resolve_music_track(
        request.music_folder,
        voiceover_path=voiceover_path,
        log=log,
    )

    audio = analyze_audio(
        voiceover_path=voiceover_path,
        music_path=selected_music_track,
        whisper_model=request.whisper_model,
        log=log,
    )
    segments = audio.transcript_segments
    log(f"Transcript segments: {len(segments)}")

    local_clips = list_video_clips(request.clips_folder)

    tuned = _runtime_tuning(
        request,
        transcript_segments=len(segments),
        local_clip_count=len(local_clips),
    )
    log(
        "Runtime tuning: "
        f"budget={tuned['ollama_scene_budget']}, "
        f"critic={'on' if tuned['enable_ollama_critic'] else 'off'}, "
        f"shortlist={tuned['match_scene_shortlist']}, "
        f"auto={'on' if request.auto_optimize_runtime else 'off'}"
    )

    plan = build_plan(
        segments,
        log=log,
        enable_ollama_scene_planning=request.enable_ollama_scene_planning,
        ollama_scene_budget=int(tuned["ollama_scene_budget"]),
        enable_ollama_critic=bool(tuned["enable_ollama_critic"]),
    )
    log(f"Subtitle plan segments: {len(plan)}")
    clip_model = os.getenv("CLIP_MODEL", "ViT-B/32").strip() or "ViT-B/32"
    pexels_key = _read_setting("PEXELS_API_KEY")
    pixabay_key = _read_setting("PIXABAY_API_KEY")

    picks = []
    if _use_advanced_clip_search(
        request,
        transcript_segments=len(plan),
        local_clip_count=len(local_clips),
    ):
        os.environ.setdefault("CLIP_INDEXER_FAST", "1")
        try:
            picks = select_best_clips(
                plan=plan,
                clips_folder=request.clips_folder,
                output_path=request.output_path,
                allow_online=bool(request.allow_stock_fetch),
                pexels_key=pexels_key,
                pixabay_key=pixabay_key,
                clip_model_name=clip_model,
                log=log,
            )
        except Exception as exc:
            log(f"Advanced CLIP search path failed ({exc}). Falling back to matcher pipeline.")
    else:
        log("Runtime optimization: skipping advanced CLIP scene search for this workload.")

    if len(picks) >= len(plan):
        timeline = build_timeline_from_selected(plan, picks)
        timeline = apply_energy_and_beat_mapping(
            timeline=timeline,
            rms_times=audio.rms_times,
            rms_values=audio.rms_values,
            beat_times=audio.beat_times,
            log=log,
        )
    else:
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
                    scene_shortlist=int(tuned["match_scene_shortlist"]),
                )
                if not stock_plan and (pexels_key or pixabay_key or request.stock_keywords.strip()):
                    # Keep API usage alive for diversity even when low-confidence scan is overly optimistic.
                    stock_plan = plan[: min(6, len(plan))]
                    log(
                        "Low-confidence scan found no weak scenes. "
                        "Fetching a small stock API seed for diversity and better alternatives."
                    )
            if stock_plan:
                stock_clips = fetch_stock_clips(
                    plan=stock_plan,
                    output_path=request.output_path,
                    width=request.output_width,
                    height=request.output_height,
                    keywords_override=request.stock_keywords,
                    log=log,
                )
                seen = {p.resolve() for p in clips}
                for path in stock_clips:
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    clips.append(path)
                if not stock_clips:
                    log("Stock fetch returned no clips (check API keys/network/provider limits).")

        if not clips:
            raise ValueError(
                "No source clips available: no local clips were found and stock clips could not be fetched. "
                "Add local clips, enable stock fetch, and optionally set PEXELS_API_KEY / PIXABAY_API_KEY in .env."
            )

        timeline = assign_clips(
            plan,
            clips,
            log=log,
            scene_shortlist=int(tuned["match_scene_shortlist"]),
            avoid_clip_repetition=request.avoid_clip_repetition,
            clip_repeat_cooldown=request.clip_repeat_cooldown,
        )
        timeline = apply_rhythm_sync(
            timeline=timeline,
            voiceover_path=request.voiceover_path,
            music_path=selected_music_track,
            log=log,
        )

    log(f"Timeline shots: {len(timeline)}")

    render_timeline(
        timeline=timeline,
        subtitle_plan=plan,
        voiceover_path=voiceover_path,
        output_path=request.output_path,
        width=request.output_width,
        height=request.output_height,
        fps=request.fps,
        render_preset=request.render_preset,
        music_track=selected_music_track,
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
        log=log,
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
