from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .models import PlannedSegment, TimelineClip

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
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
    "okay",
    "ok",
    "seen",
    "thought",
    "check",
    "double",
    "all",
    "but",
    "had",
    "even",
    "days",
    "day",
    "six",
    "button",
    "follow",
    "hit",
    "dropped",
    "drop",
    "like",
    "subscribe",
    "watch",
    "really",
    "just",
    "im",
    "ive",
    "id",
}
NATURE_TERMS = {
    "nature",
    "wild",
    "wildlife",
    "forest",
    "desert",
    "ocean",
    "river",
    "mountain",
    "animal",
    "animals",
    "bird",
    "birds",
    "insect",
    "insects",
    "earth",
    "outdoor",
    "outdoors",
}
OFF_THEME_TERMS = {
    "office",
    "meeting",
    "business",
    "corporate",
    "finance",
    "laptop",
    "computer",
    "keyboard",
    "city",
    "urban",
    "traffic",
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
    query = " ".join(keywords)
    if "nature lives in you" in text.lower():
        # Reinforce stock search / matching toward nature-first visuals.
        query = f"nature forest life wellness {query}".strip()
    return query


def _movement_score(clip_text: str) -> float:
    movement_terms = {
        "running",
        "walking",
        "flying",
        "waves",
        "action",
        "dynamic",
        "motion",
        "drone",
        "timelapse",
        "closeup",
        "close-up",
    }
    tokens = _tokenize(clip_text)
    hits = len(tokens.intersection(movement_terms))
    return min(0.35, hits * 0.08)


def _keyword_relevance_score(scene_query: str, clip_text: str) -> float:
    query_keywords = suggest_scene_keywords(scene_query, max_keywords=6)
    clip_tokens = _tokenize(clip_text)
    if not query_keywords or not clip_tokens:
        return 0.0
    hits = sum(1 for kw in query_keywords if kw in clip_tokens)
    return min(0.32, hits * 0.07)


def _theme_alignment_score(scene_query: str, clip_text: str) -> float:
    scene_tokens = _tokenize(scene_query)
    clip_tokens = _tokenize(clip_text)
    if not scene_tokens or not clip_tokens:
        return 0.0

    nature_intent = bool(scene_tokens.intersection(NATURE_TERMS))
    if not nature_intent:
        return 0.0

    nature_hits = len(clip_tokens.intersection(NATURE_TERMS))
    off_theme_hits = len(clip_tokens.intersection(OFF_THEME_TERMS))
    return min(0.36, nature_hits * 0.09) - min(0.24, off_theme_hits * 0.06)


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


def _load_clip_backend(log: callable | None):
    """Load CLIP for actual frame-level visual matching (best quality, free, local).

    Returns (model, processor) or (None, None) if unavailable.
    Requires: pip install transformers torch
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch  # noqa: F401
    except ImportError:
        if log:
            log("CLIP backend unavailable (pip install transformers torch to enable).")
        return None, None

    try:
        model_name = "openai/clip-vit-base-patch32"
        if log:
            log(f"Loading CLIP visual matcher: {model_name}")
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        if log:
            log("CLIP visual matcher ready — matching frames to scene descriptions.")
        return model, processor
    except Exception as exc:
        if log:
            log(f"CLIP load failed ({exc}). Falling back to sentence-transformers.")
        return None, None


def _sample_keyframes(path: Path, n: int = 3) -> "list[np.ndarray]":
    """Sample n evenly-spaced frames from a video, or return the image itself."""
    import numpy as np
    frames: list[np.ndarray] = []
    try:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            from PIL import Image
            frames.append(np.array(Image.open(str(path)).convert("RGB")))
            return frames
        from moviepy.editor import VideoFileClip
        with VideoFileClip(str(path)) as vc:
            dur = float(vc.duration or 0.0)
            if dur <= 0:
                return frames
            # Sample at 25%, 50%, 75% (or fewer if clip is very short).
            positions = [dur * frac for frac in [0.25, 0.50, 0.75][:n]]
            for t in positions:
                t = max(0.0, min(t, dur - 0.05))
                try:
                    frames.append(vc.get_frame(t))
                except Exception:
                    pass
    except Exception:
        pass
    return frames


def _build_clip_image_embedding(model, processor, path: Path, cache_dir: Path) -> "np.ndarray | None":
    """
    CLIP image embedding for a clip, averaged over multiple keyframes.
    Cache key includes both file size and mtime so stale entries are never used.
    Uses model submodules directly to stay compatible across all transformers versions.
    """
    import numpy as np
    import torch
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        stat = path.stat()
        cache_key = f"{path.stem}_{stat.st_size}_{int(stat.st_mtime)}"
    except Exception:
        cache_key = path.stem
    cache_file = cache_dir / f"{cache_key}.npy"

    if cache_file.exists():
        try:
            return np.load(str(cache_file))
        except Exception:
            cache_file.unlink(missing_ok=True)

    raw_frames = _sample_keyframes(path, n=3)
    if not raw_frames:
        return None

    vecs: list[np.ndarray] = []
    for frame in raw_frames:
        try:
            pil_img = Image.fromarray(frame)
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=pixel_values)
                feat = model.visual_projection(vision_out.pooler_output)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vecs.append(feat[0].cpu().numpy())
        except Exception:
            continue

    if not vecs:
        return None

    # Average embeddings across frames, then re-normalise.
    mean_vec = np.mean(vecs, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 1e-6:
        mean_vec = mean_vec / norm

    np.save(str(cache_file), mean_vec)
    return mean_vec


def _precompute_clip_embeddings(
    model,
    processor,
    clip_paths: list[Path],
    cache_dir: Path,
    log: callable | None,
) -> "list[np.ndarray | None]":
    """Encode all clip images once before the matching loop — much faster than per-scene."""
    embeddings: list = []
    cached = missed = 0
    for path in clip_paths:
        vec = _build_clip_image_embedding(model, processor, path, cache_dir)
        embeddings.append(vec)
        if vec is not None:
            cached += 1
        else:
            missed += 1
    if log:
        log(f"CLIP clip embeddings: {cached} encoded, {missed} failed/skipped")
    return embeddings


def _clip_text_embedding(model, processor, text: str) -> "np.ndarray | None":
    """
    Encode a scene query as a CLIP text embedding.
    Uses model submodules directly to stay compatible across all transformers versions.
    """
    import numpy as np
    import torch
    try:
        tokens = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        with torch.no_grad():
            text_out = model.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
            )
            feat = model.text_projection(text_out.pooler_output)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()
    except Exception:
        return None


def _clip_scores_from_embeddings(
    text_vec: "np.ndarray",
    clip_embeddings: "list[np.ndarray | None]",
) -> list[float]:
    """Dot-product text vec against pre-computed clip vecs. Returns raw cosine similarities."""
    import numpy as np
    scores: list[float] = []
    for img_vec in clip_embeddings:
        scores.append(float(np.dot(text_vec, img_vec)) if img_vec is not None else 0.0)
    return scores


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
    ranked: list[tuple[float, int]] = []
    recent_window = recent_indices[-6:]
    last_idx = recent_indices[-1] if recent_indices else None
    recent_8 = recent_indices[-8:]
    for idx, score in enumerate(base_scores):
        diversity_penalty = usage_count[idx] * 0.32 + max(0, usage_count[idx] - 2) * 0.18

        if last_idx is not None and idx == last_idx:
            recency_penalty = 0.85
        elif len(recent_indices) >= 2 and idx == recent_indices[-2]:
            recency_penalty = 0.55
        elif idx in recent_window:
            recency_penalty = 0.30
        else:
            recency_penalty = 0.0

        # Extra cooldown when a clip has already appeared multiple times very recently.
        recent_hits = recent_8.count(idx)
        streak_penalty = 0.0
        if recent_hits >= 2:
            streak_penalty = 0.20 * (recent_hits - 1)

        final_score = score - diversity_penalty - recency_penalty - streak_penalty
        ranked.append((final_score, idx))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return 0

    top_score, top_idx = ranked[0]
    # Strong anti-repeat preference: if top pick is very recent and another clip is close,
    # prefer variety unless quality would drop significantly.
    if top_idx in recent_window:
        for alt_score, alt_idx in ranked[1:6]:
            if alt_idx not in recent_window and (top_score - alt_score) <= 0.28:
                return alt_idx

    # Especially avoid immediate A-A repeats unless the score gap is clearly large.
    if last_idx is not None and top_idx == last_idx:
        for alt_score, alt_idx in ranked[1:6]:
            if alt_idx != last_idx and (top_score - alt_score) <= 0.36:
                return alt_idx

    for alt_score, alt_idx in ranked[1:4]:
        if top_idx in recent_window and alt_idx not in recent_window and (top_score - alt_score) <= 0.16:
            return alt_idx
    return top_idx


def list_video_clips(clips_folder: Path | None) -> list[Path]:
    if clips_folder is None or not clips_folder.exists():
        return []
    clips = [
        p
        for p in sorted(clips_folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
    ]
    return clips


def _build_visual_plan(plan: list[PlannedSegment]) -> tuple[list[PlannedSegment], list[int]]:
    """Return (visual_plan, plan_indices) where plan_indices[j] is the plan index for visual_plan[j]."""
    if not plan:
        return [], []

    visual: list[PlannedSegment] = []
    plan_indices: list[int] = []
    for plan_idx, seg in enumerate(plan):
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_dur = max(0.45, seg_end - seg_start)

        # Keep scene changes frequent and transcript-aligned.
        if seg_dur <= 1.9:
            visual.append(
                PlannedSegment(
                    start=seg_start,
                    end=seg_end,
                    text=seg.text,
                    duration=seg_dur,
                    transition_after=seg.transition_after,
                    transition_seconds=seg.transition_seconds,
                    emphasis=seg.emphasis,
                    highlight_phrase=seg.highlight_phrase,
                    emphasis_words=seg.emphasis_words,
                    visual_query=seg.visual_query,
                    emotion=seg.emotion,
                    pacing=seg.pacing,
                    transition_type=seg.transition_type,
                    clip_length_seconds=max(0.9, min(float(seg.clip_length_seconds), 1.9)),
                )
            )
            plan_indices.append(plan_idx)
            continue

        # Split long segments so static gaps stay low.
        parts = max(2, int(math.ceil(seg_dur / 1.6)))
        part_dur = seg_dur / parts
        for i in range(parts):
            ps = seg_start + i * part_dur
            pe = seg_end if i == parts - 1 else ps + part_dur
            visual.append(
                PlannedSegment(
                    start=ps,
                    end=pe,
                    text=seg.text,
                    duration=max(0.45, pe - ps),
                    transition_after=True,
                    transition_seconds=max(0.10, min(float(seg.transition_seconds), 0.20)),
                    emphasis=seg.emphasis,
                    highlight_phrase=seg.highlight_phrase,
                    emphasis_words=seg.emphasis_words,
                    visual_query=seg.visual_query,
                    emotion=seg.emotion,
                    pacing=seg.pacing,
                    transition_type=seg.transition_type,
                    clip_length_seconds=max(0.9, min(float(seg.clip_length_seconds), 1.6)),
                )
            )
            plan_indices.append(plan_idx)

    return visual, plan_indices


def assign_clips(
    plan: list[PlannedSegment],
    clip_paths: list[Path],
    log: callable | None = None,
) -> list[TimelineClip]:
    if not plan or not clip_paths:
        return []

    visual_plan, plan_indices = _build_visual_plan(plan)
    if not visual_plan:
        return []

    clip_descriptions = [_clip_description(path) for path in clip_paths]

    # Resolve cache dir relative to the project root (parent of any clips folder).
    clip_cache_dir = clip_paths[0].parent / "_clip_cache"

    # Tier 1: CLIP visual matching — encode all clip frames once, then per-scene
    # just encode the text and dot-product. No per-scene video I/O.
    clip_model, clip_processor = _load_clip_backend(log)
    clip_embeddings: list = []
    if clip_model is not None:
        clip_embeddings = _precompute_clip_embeddings(
            clip_model, clip_processor, clip_paths, clip_cache_dir, log
        )

    # Tier 2: sentence-transformers text embedding (fallback when CLIP unavailable).
    embedder = None if clip_model is not None else _load_embedding_backend(log)

    if clip_model is not None:
        backend = "clip"
    elif embedder is not None:
        backend = "sentence-transformers"
    else:
        backend = "keyword"
    if log:
        log(f"B-roll matching backend: {backend}")

    timeline: list[TimelineClip] = []
    usage_count = [0 for _ in clip_paths]
    recent_indices: list[int] = []
    cursor = 0.0

    for scene_idx, (segment, pidx) in enumerate(zip(visual_plan, plan_indices), start=1):
        scene_text = (segment.text or "").strip() or "voiceover"
        scene_query = (segment.visual_query or "").strip() or _scene_query(scene_text)

        if clip_model is not None:
            text_vec = _clip_text_embedding(clip_model, clip_processor, scene_query)
            if text_vec is not None:
                base_scores = _clip_scores_from_embeddings(text_vec, clip_embeddings)
            else:
                base_scores = [0.0] * len(clip_paths)

            # CLIP cosine sims sit in ~0.15–0.35. Scale them to ~0–1 so metadata
            # bonuses (each capped at 0.35) don't completely dominate.
            min_s = min(base_scores) if base_scores else 0.0
            max_s = max(base_scores) if base_scores else 1.0
            spread = max(max_s - min_s, 1e-6)
            base_scores = [(s - min_s) / spread for s in base_scores]

        elif embedder is not None:
            base_scores = _semantic_scores(embedder, scene_query, clip_descriptions)
        else:
            base_scores = [_lexical_similarity(scene_query, desc) for desc in clip_descriptions]

        # Blend normalised visual score with lightweight metadata signals.
        scores = [
            base
            + _movement_score(desc) * 0.4
            + _keyword_relevance_score(scene_query, desc) * 0.4
            + _theme_alignment_score(scene_query, desc) * 0.4
            for base, desc in zip(base_scores, clip_descriptions)
        ]

        chosen_idx = _select_best_index(scores, usage_count, recent_indices)
        clip_path = clip_paths[chosen_idx]
        usage_count[chosen_idx] += 1
        recent_indices.append(chosen_idx)

        if log:
            keywords = suggest_scene_keywords(scene_text)
            score_str = f"{scores[chosen_idx]:.3f}" if scores else "?"
            log(
                f"Scene {scene_idx}: [{backend}={score_str}] "
                f"keywords={', '.join(keywords)} | "
                f"query='{scene_query}' | emotion={segment.emotion} | chosen={clip_path.name}"
            )

        end = cursor + segment.duration
        timeline.append(
            TimelineClip(
                source_path=clip_path,
                timeline_start=cursor,
                timeline_end=end,
                transition_after=segment.transition_after,
                transition_seconds=segment.transition_seconds,
                transition_type=segment.transition_type,
                emotion=segment.emotion,
                plan_idx=pidx,
            )
        )
        cursor = end
    return timeline
