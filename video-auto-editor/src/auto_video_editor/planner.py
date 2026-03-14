from __future__ import annotations

from .models import PlannedSegment, TranscriptSegment


def build_plan(segments: list[TranscriptSegment]) -> list[PlannedSegment]:
    plan: list[PlannedSegment] = []
    for seg in segments:
        duration = max(0.5, float(seg.end - seg.start))
        plan.append(
            PlannedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                duration=duration,
            )
        )
    return plan
