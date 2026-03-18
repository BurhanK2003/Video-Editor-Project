# Local Auto Video Editor (Free Tools)

This is a local-first desktop prototype for auto-editing videos from a voiceover.

## What it does now

- Import a voiceover audio file
- Import a local folder of video clips, or fetch stock clips from Pexels/Pixabay/Openverse when local footage is missing
- Optionally import a music folder
- Auto-transcribe the voiceover (if `faster-whisper` is installed and model download succeeds)
- Build a rough cut by matching transcript segments to clips
- Auto-mix background music under voiceover
- Adaptive caption-safe zones that move subtitles away from likely subject regions
- Per-word karaoke highlight timing from word-level transcript timestamps
- Export final MP4
- Creator Mode: one voiceover, full controls, final export
- Batch Mode: folder of voiceovers, optional CSV overrides, one video generated per voiceover

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
Openverse image search is keyless and works without API credentials.

## Run

```powershell
& "E:/Personal Projects/Video-Editor-Project/.venv/Scripts/python.exe" -m src.auto_video_editor.main
```

## Workflow Modes

### Creator Mode

Use this when you want to craft one video with full control.

1. Open the app.
2. Set `Voiceover` (or use Script-to-Video).
3. Optionally set `Clips Folder`, `Stock Search`, `Music Folder`.
4. Set `Output File`.
5. Tune render/caption/overlay settings.
6. Click `Auto Edit`.

Outputs:

- Final video MP4
- Quality reports next to the output:
	- `your_video.quality-report.json`
	- `your_video.quality-report.md`

### Batch Mode

Use this when you want to generate one video per voiceover automatically.

1. Open the app.
2. In `Batch Mode` section, set:
	 - `Voiceovers Folder`
	 - `Batch Output`
	 - Optional `Manifest CSV (opt)`
3. Keep your global render settings (resolution, captions, transitions, stock fetch).
4. Click `Batch Auto Edit`.

Batch behavior:

- Generates one MP4 per voiceover file found in the selected folder.
- Reuses on-disk caches between jobs (`output/_stock_cache`, `.llm_query_cache.json`, `.llm_scene_plan_cache.json`, clip embedding cache).
- Writes per-job batch logs in `Batch Output/batch_logs/*.log`.
- Writes per-job quality reports beside each output MP4.

Supported voiceover formats for batch input:

- `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`

Optional CSV manifest columns:

- `voiceover` or `filename` or `file` or `stem` (required key column)
- `title` (used in output filename + hook text override)
- `keywords` (overrides stock search keywords for that voiceover)
- `caption_style` (`bold_stroke`, `yellow_active`, `gradient_fill`)
- `transition_style` (`none`, `pro_weighted`)

Example `batch_manifest.csv`:

```csv
voiceover,title,keywords,caption_style,transition_style
episode1.wav,How Trees Communicate,forest roots mycelium macro,yellow_active,pro_weighted
episode2.wav,Why Birds Fly in Formation,birds sky migration aerial,gradient_fill,pro_weighted
```

With stock fetching enabled, the app can download stock clips/images and then match from the combined pool of your local clips + downloaded stock assets. If you leave `Clips Folder` empty, it will rely entirely on stock assets. Downloaded stock files are cached in `output/_stock_cache`.

Phase 1 AI matching now does three things:

- Extracts scene keywords from transcript segments.
- Uses those keywords to search stock clips.
- Picks the best clip per scene using semantic matching (when `sentence-transformers` is installed) or keyword overlap fallback.

The matcher now also adds:

- Nature-theme alignment scoring to keep selected clips on-topic for wildlife/nature stories.
- Relevance and diversity scoring to reduce repetitive clip reuse when top candidates are close.

Check the log panel for lines like:

- `Scene 3: keywords=brain, cells, neuron | query='brain cells neuron' | chosen=pexels_12345.mp4`

## Notes

- First transcription run may download a model and take longer.
- If transcription is unavailable, the app still renders using one full-length segment.
- Pexels and Pixabay require API keys for automated downloads.
- Openverse image fallback does not require an API key.
- Stock footage downloads need an internet connection and provider availability.

## Next upgrades

- AI music generation option
- Automatic pipeline where folder of voiceover are added and a video is generated for each automatically. It runs every week for new ones.
