from __future__ import annotations

from pathlib import Path

from .models import PlannedSegment, TimelineClip

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def list_video_clips(clips_folder: Path | None) -> list[Path]:
    if clips_folder is None or not clips_folder.exists():
        return []
    clips = [
        p
        for p in sorted(clips_folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return clips


def assign_clips(plan: list[PlannedSegment], clip_paths: list[Path]) -> list[TimelineClip]:
    if not plan or not clip_paths:
        return []

    timeline: list[TimelineClip] = []
    cursor = 0.0
    for idx, segment in enumerate(plan):
        clip_path = clip_paths[idx % len(clip_paths)]
        end = cursor + segment.duration
        timeline.append(
            TimelineClip(
                source_path=clip_path,
                timeline_start=cursor,
                timeline_end=end,
            )
        )
        cursor = end
    return timeline
