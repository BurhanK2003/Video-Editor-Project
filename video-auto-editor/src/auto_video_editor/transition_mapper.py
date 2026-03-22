from __future__ import annotations

import bisect
from dataclasses import replace

from .audio_analyzer import energy_at_time
from .models import TimelineClip


def map_transition_from_energy(energy: float) -> tuple[str, float]:
    if energy < 0.3:
        return "fade", 0.6
    if energy <= 0.6:
        return "fade", 0.4
    return "zoom_in", 0.18


def _snap_time_to_beat(timestamp: float, beats: list[float], tolerance: float = 0.2) -> float:
    if not beats:
        return timestamp
    idx = bisect.bisect_left(beats, timestamp)
    candidates: list[float] = []
    if idx < len(beats):
        candidates.append(beats[idx])
    if idx > 0:
        candidates.append(beats[idx - 1])
    if not candidates:
        return timestamp
    nearest = min(candidates, key=lambda b: abs(b - timestamp))
    if abs(nearest - timestamp) <= max(0.01, tolerance):
        return float(nearest)
    return timestamp


def apply_energy_and_beat_mapping(
    timeline: list[TimelineClip],
    rms_times: list[float],
    rms_values: list[float],
    beat_times: list[float],
    log: callable | None = None,
) -> list[TimelineClip]:
    if not timeline:
        return []

    cuts = [float(clip.timeline_end) for clip in timeline[:-1]]
    snapped = [_snap_time_to_beat(ts, beat_times, tolerance=0.2) for ts in cuts]

    rebuilt: list[TimelineClip] = []
    cursor = 0.0
    for idx, clip in enumerate(timeline):
        end = float(timeline[-1].timeline_end) if idx == len(timeline) - 1 else float(snapped[idx])
        if end <= cursor + 0.10:
            end = cursor + max(0.10, float(clip.timeline_end - clip.timeline_start))

        cut_energy = energy_at_time(rms_times, rms_values, cursor)
        transition_type, transition_seconds = map_transition_from_energy(cut_energy)

        rebuilt.append(
            replace(
                clip,
                timeline_start=cursor,
                timeline_end=end,
                transition_type=transition_type,
                transition_seconds=transition_seconds,
            )
        )
        cursor = end

    if log:
        log(
            "Transition mapping: "
            f"cuts={len(cuts)}, beats={len(beat_times)}, rms_points={len(rms_values)}"
        )
    return rebuilt
