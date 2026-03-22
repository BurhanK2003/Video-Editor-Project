from __future__ import annotations

from pathlib import Path
from typing import Any

from .matcher import assign_clips
from .planner import build_plan
from .renderer import render_video
from .rhythm import apply_rhythm_sync, resolve_music_track
from .transcribe import transcribe_voiceover


class VideoEngine:
    """Refactored rendering entrypoint.

    Phase 1 currently upgrades the caption system while preserving the existing
    end-to-end pipeline behavior.
    """

    def __init__(self, log: callable | None = None):
        self._log = log or (lambda _msg: None)

    def render(
        self,
        voiceover_path: str | Path,
        clips: list[str | Path],
        music_path: str | Path | None,
        config: dict[str, Any],
    ) -> Path:
        """Render a short-form edit from voiceover + source clips.

        Args:
            voiceover_path: Path to a local voiceover audio file.
            clips: Candidate source clips (video/image files) already available locally.
            music_path: Optional background music directory.
            config: Rendering config values (width, height, fps, styles, etc.).

        Returns:
            Output path of the rendered mp4 file.
        """
        voiceover = Path(voiceover_path)
        output_path = Path(config.get("output_path") or "output/edited_video.mp4")

        if not voiceover.exists() or not voiceover.is_file():
            raise ValueError(f"Voiceover file not found: {voiceover}")
        if not clips:
            raise ValueError("At least one source clip is required.")

        clip_paths = [Path(c) for c in clips]
        available_clips = [p for p in clip_paths if p.exists() and p.is_file()]
        if not available_clips:
            raise ValueError("No valid source clips were found in the provided list.")

        model_size = str(config.get("whisper_model", "base"))
        segments = transcribe_voiceover(voiceover_path=voiceover, model_size=model_size, log=self._log)
        plan = build_plan(
            segments,
            log=self._log,
            enable_ollama_scene_planning=bool(config.get("enable_ollama_scene_planning", True)),
            ollama_scene_budget=int(config.get("ollama_scene_budget", 8)),
            enable_ollama_critic=bool(config.get("enable_ollama_critic", True)),
        )
        timeline = assign_clips(
            plan,
            available_clips,
            log=self._log,
            scene_shortlist=int(config.get("match_scene_shortlist", 28)),
            avoid_clip_repetition=bool(config.get("avoid_clip_repetition", True)),
            clip_repeat_cooldown=int(config.get("clip_repeat_cooldown", 8)),
        )
        selected_music_track = resolve_music_track(
            Path(music_path) if music_path else None,
            voiceover_path=voiceover,
            log=self._log,
        )
        timeline = apply_rhythm_sync(
            timeline=timeline,
            voiceover_path=voiceover,
            music_path=selected_music_track,
            log=self._log,
        )

        render_video(
            timeline_clips=timeline,
            subtitle_plan=plan,
            voiceover_path=voiceover,
            output_path=output_path,
            width=int(config.get("width", 1080)),
            height=int(config.get("height", 1920)),
            fps=int(config.get("fps", 24)),
            render_preset=str(config.get("render_preset", "veryfast")),
            music_folder=selected_music_track,
            log=self._log,
            transition_style=str(config.get("transition_style", "pro_weighted")),
            transition_duration=float(config.get("transition_duration", 0.22)),
            caption_style=str(config.get("caption_style", "bold_stroke")),
            caption_position_ratio=config.get("caption_position_ratio"),
            caption_max_lines=config.get("caption_max_lines"),
            caption_font_scale=float(config.get("caption_font_scale", 1.0)),
            caption_pop_scale=float(config.get("caption_pop_scale", 1.0)),
            enable_adaptive_caption_safe_zones=bool(config.get("enable_adaptive_caption_safe_zones", True)),
            enable_karaoke_highlight=bool(config.get("enable_karaoke_highlight", True)),
            enable_motion_overlays=bool(config.get("enable_motion_overlays", True)),
            hook_text_override=str(config.get("hook_text_override", "")),
            stat_badge_text=str(config.get("stat_badge_text", "")),
            cta_text=str(config.get("cta_text", "Learn More")),
            logo_path=(Path(config["logo_path"]) if config.get("logo_path") else None),
            enable_progress_bar=bool(config.get("enable_progress_bar", True)),
        )

        return output_path
