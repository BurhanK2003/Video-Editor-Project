from __future__ import annotations

import argparse
from pathlib import Path

from .ffmpeg_setup import ensure_ffmpeg_for_moviepy, ensure_pillow_compatibility


def _run_cli_pipeline() -> None:
    from .models import AutoEditRequest
    from .orchestrator import run_auto_edit

    parser = argparse.ArgumentParser(description="Auto Video Editor")
    parser.add_argument("--pipeline", action="store_true", help="Run orchestration pipeline in CLI mode")
    parser.add_argument("--voiceover", type=str, default="", help="Path to voiceover audio")
    parser.add_argument("--clips", type=str, default="", help="Path to local clips folder")
    parser.add_argument("--music", type=str, default="", help="Path to music file/folder")
    parser.add_argument("--output", type=str, default="output/edited_video.mp4", help="Output mp4 path")
    parser.add_argument("--whisper", type=str, default="base", help="Whisper model size")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model: ViT-B/32 or ViT-L/14")
    args = parser.parse_args()

    if not args.pipeline:
        from .app import run_app

        run_app()
        return

    voiceover_path = Path(args.voiceover).expanduser().resolve()
    clips_folder = Path(args.clips).expanduser().resolve() if args.clips else None
    music_folder = Path(args.music).expanduser().resolve() if args.music else None
    output_path = Path(args.output).expanduser().resolve()

    if not voiceover_path.exists() or not voiceover_path.is_file():
        raise ValueError("--voiceover must point to an existing audio file")

    if clips_folder is not None and (not clips_folder.exists() or not clips_folder.is_dir()):
        raise ValueError("--clips must point to an existing folder")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = AutoEditRequest(
        voiceover_path=voiceover_path,
        clips_folder=clips_folder,
        output_path=output_path,
        music_folder=music_folder,
        whisper_model=args.whisper,
        allow_stock_fetch=True,
    )

    # Keep CLIP model configurable through CLI and env var fallback in orchestrator.
    import os

    os.environ["CLIP_MODEL"] = str(args.clip_model)
    run_auto_edit(request, log=print)


if __name__ == "__main__":
    ensure_pillow_compatibility()
    ensure_ffmpeg_for_moviepy()
    _run_cli_pipeline()
