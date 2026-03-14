from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .models import PlannedSegment, TimelineClip

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
STOP_WORDS = {
    "and",
    "about",
    "after",
    "again",
    "also",
    "are",
    "because",
    "been",
    "can",
    "for",
    "from",
    "have",
    "how",
    "in",
    "into",
    "its",
    "just",
    "like",
    "make",
    "more",
    "not",
    "our",
    "over",
    "should",
    "some",
    "such",
    "than",
    "the",
    "them",
    "than",
    "that",
    "their",
    "there",
    "they",
    "this",
    "those",
    "too",
    "use",
    "very",
    "what",
    "with",
    "would",
    "you",
}


def suggest_scene_keywords(text: str, max_keywords: int = 4) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    if not filtered:
        return ["storytelling"]

    frequency: dict[str, int] = {}
    for word in filtered:
        frequency[word] = frequency.get(word, 0) + 1

    ordered = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ordered[:max_keywords]]


def _scene_query(text: str) -> str:
    keywords = suggest_scene_keywords(text, max_keywords=3)
    return " ".join(keywords)


def _clip_metadata_text(clip_path: Path) -> str:
    sidecar = clip_path.with_suffix(clip_path.suffix + ".json")
    if not sidecar.exists():
        return ""
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return ""

    parts = []
    for key in ("provider", "query", "title", "tags"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def _clip_description(clip_path: Path) -> str:
    stem_text = re.sub(r"[_\-]+", " ", clip_path.stem.lower())
    metadata_text = _clip_metadata_text(clip_path)
    if metadata_text:
        return f"{metadata_text} {stem_text}".strip()
    return stem_text.strip() or "generic b roll footage"


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOP_WORDS}


def _lexical_similarity(a: str, b: str) -> float:
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / max(1, union)


def _load_embedding_backend(log: callable | None):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        if log:
            log("AI matcher fallback: sentence-transformers not installed, using keyword overlap.")
        return None

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        if log:
            log(f"AI matcher fallback: embedding model unavailable ({exc}), using keyword overlap.")
        return None

    if log:
        log("AI matcher: semantic embedding model loaded (all-MiniLM-L6-v2).")
    return model


def _semantic_scores(model, scene_text: str, clip_texts: list[str]) -> list[float]:
    scene_vec = model.encode([scene_text], normalize_embeddings=True)[0]
    clip_vecs = model.encode(clip_texts, normalize_embeddings=True)
    return [float((scene_vec * vec).sum()) for vec in clip_vecs]


def _select_best_index(
    base_scores: list[float],
    usage_count: list[int],
    recent_indices: list[int],
) -> int:
    best_idx = 0
    best_score = -math.inf
    for idx, score in enumerate(base_scores):
        diversity_penalty = usage_count[idx] * 0.08
        recency_penalty = 0.06 if idx in recent_indices[-2:] else 0.0
        final_score = score - diversity_penalty - recency_penalty
        if final_score > best_score:
            best_score = final_score
            best_idx = idx
    return best_idx


def list_video_clips(clips_folder: Path | None) -> list[Path]:
    if clips_folder is None or not clips_folder.exists():
        return []
    clips = [
        p
        for p in sorted(clips_folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return clips


def assign_clips(
    plan: list[PlannedSegment],
    clip_paths: list[Path],
    log: callable | None = None,
) -> list[TimelineClip]:
    if not plan or not clip_paths:
        return []

    clip_descriptions = [_clip_description(path) for path in clip_paths]
    embedder = _load_embedding_backend(log)

    timeline: list[TimelineClip] = []
    usage_count = [0 for _ in clip_paths]
    recent_indices: list[int] = []
    cursor = 0.0

    for scene_idx, segment in enumerate(plan, start=1):
        scene_text = (segment.text or "").strip() or "voiceover"
        scene_query = _scene_query(scene_text)
        if embedder is not None:
            scores = _semantic_scores(embedder, scene_query, clip_descriptions)
        else:
            scores = [_lexical_similarity(scene_query, clip_text) for clip_text in clip_descriptions]

        chosen_idx = _select_best_index(scores, usage_count, recent_indices)
        clip_path = clip_paths[chosen_idx]
        usage_count[chosen_idx] += 1
        recent_indices.append(chosen_idx)

        if log:
            keywords = suggest_scene_keywords(scene_text)
            log(
                f"Scene {scene_idx}: keywords={', '.join(keywords)} | "
                f"query='{scene_query}' | chosen={clip_path.name}"
            )

        end = cursor + segment.duration
        timeline.append(
            TimelineClip(
                source_path=clip_path,
                timeline_start=cursor,
                timeline_end=end,
            )
        )
        cursor = end
    return timeline
