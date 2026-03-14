from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class PlannedSegment:
    start: float
    end: float
    text: str
    duration: float


@dataclass
class TimelineClip:
    source_path: Path
    timeline_start: float
    timeline_end: float


@dataclass
class AutoEditRequest:
    voiceover_path: Path
    clips_folder: Path | None
    output_path: Path
    music_folder: Path | None = None
    whisper_model: str = "base"
    output_width: int = 1080
    output_height: int = 1920
    fps: int = 24
    render_preset: str = "veryfast"
    allow_stock_fetch: bool = True
    stock_keywords: str = ""
