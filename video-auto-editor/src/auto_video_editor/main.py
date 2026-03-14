from .ffmpeg_setup import ensure_ffmpeg_for_moviepy, ensure_pillow_compatibility


if __name__ == "__main__":
    ensure_pillow_compatibility()
    ensure_ffmpeg_for_moviepy()

    from .app import run_app

    run_app()
