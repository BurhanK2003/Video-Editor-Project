from dataclasses import dataclass
from pathlib import Path


@dataclass
class WordToken:
    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[WordToken] | None = None


@dataclass
class PlannedSegment:
    start: float
    end: float
    text: str
    duration: float
    transition_after: bool = True
    transition_seconds: float = 0.22
    emphasis: bool = False
    highlight_phrase: str = ""
    emphasis_words: list[str] | None = None
    visual_query: str = ""
    emotion: str = "curiosity"
    pacing: str = "fast"
    transition_type: str = "jump_cut"
    clip_length_seconds: float = 2.0


@dataclass
class TimelineClip:
    source_path: Path
    timeline_start: float
    timeline_end: float
    transition_after: bool = True
    transition_seconds: float = 0.2
    transition_type: str = "jump_cut"
    emotion: str = "curiosity"


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
    transition_style: str = "crossfade"  # none | crossfade | zoom | fade_black
    transition_duration: float = 0.22
