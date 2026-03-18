from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path

from .models import AutoEditRequest
from .orchestrator import run_auto_edit

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip().lower())
    slug = slug.strip("-")
    return slug or "video"


def _list_voiceovers(folder: Path) -> list[Path]:
    return [
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def _load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if not path or not path.exists() or not path.is_file():
        return {}

    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            normalized = {str(k or "").strip().lower(): str(v or "").strip() for k, v in row.items()}
            key = (
                normalized.get("voiceover")
                or normalized.get("filename")
                or normalized.get("file")
                or normalized.get("stem")
                or ""
            ).strip()
            if not key:
                continue
            stem = Path(key).stem.lower()
            mapping[stem] = normalized
    return mapping


def run_batch_auto_edit(
    base_request: AutoEditRequest,
    voiceovers_folder: Path,
    output_folder: Path,
    manifest_path: Path | None,
    log: callable,
) -> dict[str, int]:
    """Render one output per voiceover file using shared settings and optional CSV overrides."""
    voiceovers = _list_voiceovers(voiceovers_folder)
    if not voiceovers:
        raise ValueError("No voiceover files found in the selected folder.")

    manifest = _load_manifest(manifest_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    batch_logs_dir = output_folder / "batch_logs"
    batch_logs_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"Batch mode: {len(voiceovers)} voiceovers | "
        f"manifest={'yes' if manifest_path else 'no'} | output={output_folder}"
    )

    success = 0
    failed = 0

    for idx, voiceover in enumerate(voiceovers, start=1):
        row = manifest.get(voiceover.stem.lower(), {})
        title = row.get("title") or voiceover.stem
        stock_keywords = row.get("keywords") or base_request.stock_keywords

        output_stem = _slugify(title)
        output_path = output_folder / f"{output_stem}.mp4"

        transition_style = row.get("transition_style", "").strip().lower() or base_request.transition_style
        if transition_style not in {"none", "pro_weighted"}:
            transition_style = base_request.transition_style

        caption_style = row.get("caption_style", "").strip().lower() or base_request.caption_style
        if caption_style not in {"bold_stroke", "yellow_active", "gradient_fill", "beast", "clean", "kinetic"}:
            caption_style = base_request.caption_style

        job_request = replace(
            base_request,
            voiceover_path=voiceover,
            output_path=output_path,
            stock_keywords=stock_keywords,
            transition_style=transition_style,
            caption_style=caption_style,
            hook_text_override=row.get("title") or base_request.hook_text_override,
            script_text="",
            script_voice="",
        )

        job_lines: list[str] = []

        def _job_log(message: str) -> None:
            line = f"[{idx}/{len(voiceovers)}][{voiceover.name}] {message}"
            job_lines.append(line)
            log(line)

        _job_log("Starting job")
        try:
            run_auto_edit(job_request, log=_job_log)
            success += 1
            _job_log(f"Done: {output_path.name}")
        except Exception as exc:
            failed += 1
            _job_log(f"Failed: {exc}")

        (batch_logs_dir / f"{output_stem}.log").write_text(
            "\n".join(job_lines) + "\n",
            encoding="utf-8",
        )

    log(f"Batch complete: {success} succeeded, {failed} failed")
    return {"total": len(voiceovers), "success": success, "failed": failed}
