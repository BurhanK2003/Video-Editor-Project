from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

from .models import PlannedSegment, TimelineClip


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _timeline_duration(timeline_clips: list[TimelineClip], subtitle_plan: list[PlannedSegment]) -> float:
    timeline_end = max((float(c.timeline_end) for c in timeline_clips), default=0.0)
    subtitle_end = max((float(s.end) for s in subtitle_plan), default=0.0)
    return max(0.0, timeline_end, subtitle_end)


def _event_count_first_seconds(
    timeline_clips: list[TimelineClip],
    subtitle_plan: list[PlannedSegment],
    seconds: float,
) -> int:
    events: set[float] = set()
    window = max(0.0, float(seconds))

    # First visual frame is a meaningful event in short-form hooks.
    events.add(0.0)

    for clip in timeline_clips[1:]:
        t = float(clip.timeline_start)
        if 0.0 <= t <= window:
            events.add(round(t, 3))

    for seg in subtitle_plan:
        t = float(seg.start)
        if 0.0 <= t <= window:
            events.add(round(t, 3))

    return len(events)


def _first_caption_start(subtitle_plan: list[PlannedSegment]) -> float | None:
    if not subtitle_plan:
        return None
    return min(float(seg.start) for seg in subtitle_plan)


def _shot_durations(timeline_clips: list[TimelineClip]) -> list[float]:
    durations: list[float] = []
    for clip in timeline_clips:
        dur = max(0.0, float(clip.timeline_end) - float(clip.timeline_start))
        if dur > 0:
            durations.append(dur)
    return durations


def _render_scene_gaps_seconds(video_path: Path, threshold: float = 0.25) -> list[float]:
    movie_expr = str(video_path).replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"movie='{movie_expr}',select=gt(scene\\,{threshold})",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, errors="ignore")
    except Exception:
        return []

    times: list[float] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split(",", 1)[0]
        val = _safe_float(token, default=-1.0)
        if val >= 0.0:
            times.append(val)

    if len(times) < 2:
        return []

    return [max(0.0, times[i] - times[i - 1]) for i in range(1, len(times))]


def build_export_quality_report(
    timeline_clips: list[TimelineClip],
    subtitle_plan: list[PlannedSegment],
    output_path: Path,
) -> dict:
    total_duration = _timeline_duration(timeline_clips, subtitle_plan)
    shot_durations = _shot_durations(timeline_clips)
    cut_count = max(0, len(timeline_clips) - 1)
    cut_density = (cut_count / total_duration) if total_duration > 0 else 0.0

    first_caption = _first_caption_start(subtitle_plan)
    hook_events_3s = _event_count_first_seconds(timeline_clips, subtitle_plan, seconds=3.0)
    long_shots = [d for d in shot_durations if d > 1.8]

    render_gaps = _render_scene_gaps_seconds(output_path, threshold=0.25) if output_path.exists() else []
    max_render_gap = max(render_gaps, default=0.0)

    checks = {
        "hook_events_first_3s": {
            "target": ">= 3",
            "actual": hook_events_3s,
            "pass": hook_events_3s >= 3,
        },
        "first_caption_start": {
            "target": "<= 0.70s",
            "actual": None if first_caption is None else round(first_caption, 3),
            "pass": first_caption is not None and first_caption <= 0.70,
        },
        "long_shot_count": {
            "target": "0 shots > 1.8s",
            "actual": len(long_shots),
            "pass": len(long_shots) == 0,
        },
        "max_render_static_gap": {
            "target": "<= 1.80s",
            "actual": round(max_render_gap, 3) if max_render_gap > 0 else None,
            "pass": (max_render_gap <= 1.80) if max_render_gap > 0 else None,
        },
        "cut_density": {
            "target": "0.40-1.60 cuts/s",
            "actual": round(cut_density, 3),
            "pass": 0.40 <= cut_density <= 1.60,
        },
    }

    report = {
        "video": {
            "output_path": str(output_path),
            "duration_seconds": round(total_duration, 3),
            "timeline_shots": len(timeline_clips),
            "subtitle_segments": len(subtitle_plan),
        },
        "metrics": {
            "cut_count": cut_count,
            "cut_density": round(cut_density, 3),
            "avg_shot_seconds": round(statistics.fmean(shot_durations), 3) if shot_durations else 0.0,
            "max_shot_seconds": round(max(shot_durations), 3) if shot_durations else 0.0,
            "hook_events_first_3s": hook_events_3s,
            "first_caption_start_seconds": None if first_caption is None else round(first_caption, 3),
            "render_scene_gap_max_seconds": round(max_render_gap, 3) if max_render_gap > 0 else None,
        },
        "checks": checks,
    }

    check_values = [item["pass"] for item in checks.values() if isinstance(item.get("pass"), bool)]
    report["summary"] = {
        "pass_count": sum(1 for val in check_values if val),
        "fail_count": sum(1 for val in check_values if not val),
        "all_passed": all(check_values) if check_values else False,
    }
    return report


def _markdown_check_line(name: str, data: dict) -> str:
    passed = data.get("pass")
    status = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
    return f"- {name}: {status} (target: {data.get('target')}, actual: {data.get('actual')})"


def write_export_quality_report(output_path: Path, report: dict) -> tuple[Path, Path]:
    report_json = output_path.with_name(f"{output_path.stem}.quality-report.json")
    report_md = output_path.with_name(f"{output_path.stem}.quality-report.md")

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Export Quality Report",
        "",
        f"Output: {report['video']['output_path']}",
        f"Duration: {report['video']['duration_seconds']}s",
        f"Timeline shots: {report['video']['timeline_shots']}",
        f"Subtitle segments: {report['video']['subtitle_segments']}",
        "",
        "## Metrics",
        f"- Cut count: {report['metrics']['cut_count']}",
        f"- Cut density: {report['metrics']['cut_density']} cuts/s",
        f"- Avg shot length: {report['metrics']['avg_shot_seconds']}s",
        f"- Max shot length: {report['metrics']['max_shot_seconds']}s",
        f"- Hook events first 3s: {report['metrics']['hook_events_first_3s']}",
        f"- First caption start: {report['metrics']['first_caption_start_seconds']}",
        f"- Render max scene gap: {report['metrics']['render_scene_gap_max_seconds']}",
        "",
        "## Gate Checks",
    ]

    for name, data in report.get("checks", {}).items():
        lines.append(_markdown_check_line(name, data))

    summary = report.get("summary", {})
    lines.extend(
        [
            "",
            "## Summary",
            f"- Passed: {summary.get('pass_count', 0)}",
            f"- Failed: {summary.get('fail_count', 0)}",
            f"- All passed: {summary.get('all_passed', False)}",
            "",
        ]
    )

    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_json, report_md
