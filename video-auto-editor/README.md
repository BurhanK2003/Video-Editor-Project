# Local Auto Video Editor (Free Tools)

This is a local-first desktop prototype for auto-editing videos from a voiceover.

## What it does now

- Import a voiceover audio file
- Import a local folder of video clips, or fetch stock clips from Pexels/Pixabay when local footage is missing
- Optionally import a music folder
- Auto-transcribe the voiceover (if `faster-whisper` is installed and model download succeeds)
- Build a rough cut by matching transcript segments to clips
- Auto-mix background music under voiceover
- Export final MP4

## Tech

- UI: Tkinter (built into Python)
- Video pipeline: MoviePy + FFmpeg
- Transcription: optional faster-whisper (fallback included)

## Prerequisites

1. Python 3.10+
2. Optional: FFmpeg in PATH for terminal use (`ffmpeg -version`)

The app can run without FFmpeg in PATH because it auto-resolves a bundled FFmpeg binary via `imageio-ffmpeg`.

To verify FFmpeg:

```powershell
ffmpeg -version
```

## Install

```powershell
cd video-auto-editor
& "E:/Personal Projects/Video-Editor-Project/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

Optional transcription upgrade:

```powershell
& "E:/Personal Projects/Video-Editor-Project/.venv/Scripts/python.exe" -m pip install faster-whisper
```

Optional Phase 1 AI semantic matching upgrade:

```powershell
& "E:/Personal Projects/Video-Editor-Project/.venv/Scripts/python.exe" -m pip install sentence-transformers
```

Optional stock footage setup:

```powershell
@"
PEXELS_API_KEY=your_pexels_key
PIXABAY_API_KEY=your_pixabay_key
"@ | Set-Content .env
```

The app will read `PEXELS_API_KEY` and `PIXABAY_API_KEY` from the process environment or a local `.env` file in the project root.

## Run

```powershell
& "E:/Personal Projects/Video-Editor-Project/.venv/Scripts/python.exe" -m src.auto_video_editor.main
```

If you leave `Clips Folder` empty, keep stock fetching enabled and optionally enter `Stock Search` keywords such as `city skyline, office, teamwork`. When local footage is missing, the app will try transcript-based searches first and download stock clips into `output/_stock_cache`.

Phase 1 AI matching now does three things:

- Extracts scene keywords from transcript segments.
- Uses those keywords to search stock clips.
- Picks the best clip per scene using semantic matching (when `sentence-transformers` is installed) or keyword overlap fallback.

Check the log panel for lines like:

- `Scene 3: keywords=brain, cells, neuron | query='brain cells neuron' | chosen=pexels_12345.mp4`

## Notes

- First transcription run may download a model and take longer.
- If transcription is unavailable, the app still renders using one full-length segment.
- Pexels and Pixabay both require API keys for automated downloads.
- Stock footage downloads need an internet connection and provider availability.

## Next upgrades

- Smarter semantic clip matching
- Subtitle styling editor
- AI music generation option
