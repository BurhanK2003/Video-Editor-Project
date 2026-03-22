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
    "subscribe",
    "watch",
    "really",
    "im",
    "ive",
    "id",
    "is",
    "am",
    "was",
    "up",
    "on",
    "at",
    "be",
    "get",
    "got",
    "as",
    "to",
    "by",
    "it",
    "do",
    "go",
    "or",
    "if",
    "when",
    "then",
    "every",
    "everything",
    "something",
    "anything",
    "nothing",
    "fact",
    "facts",
    "certainly",
    "maybe",
    "perhaps",
    "word",
    "words",
    "speak",
    "speaks",
    "doesnt",
    "dont",
    "didnt",
    "cant",
    "couldnt",
    "wont",
    "wouldnt",
    "shouldnt",
    "ago",
    "few",
    "theyll",
    "youre",
    "weve",
    "youll",
    "almost",
}
NON_VISUAL_TERMS = {
    "memory",
    "grief",
    "love",
    "loved",
    "ones",
    "silently",
    "funeral",
    "human",
    "emotion",
    "emotions",
    "separate",
    "right",
    "place",
    "want",
    "learn",
    "theyll",
    "youre",
    "doesnt",
    "not",
}
GENERIC_STYLE_TERMS = {
    "cinematic",
    "documentary",
    "natural",
    "outdoor",
    "outdoors",
    "wildlife",
    "light",
    "sunlight",
    "logo",
    "text",
    "glow",
    "broll",
    "4k",
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
HIGH_VALUE_SUBJECT_TERMS = {
    "beetle",
    "darkling",
    "ant",
    "spider",
    "lizard",
    "snake",
    "penguin",
    "elephant",
    "dolphin",
    "whale",
    "shark",
    "lion",
    "tiger",
    "bear",
    "eagle",
    "wolf",
    "fox",
    "giraffe",
    "zebra",
    "rhino",
    "gorilla",
    "monkey",
    "bird",
    "deer",
    "insect",
    "animal",
}
COMPOUND_SUBJECT_HEADS = {
    "beetle",
    "penguin",
    "elephant",
    "dolphin",
    "whale",
    "shark",
    "lion",
    "tiger",
    "bear",
    "eagle",
    "wolf",
    "fox",
    "giraffe",
    "zebra",
    "rhino",
    "gorilla",
    "monkey",
    "bird",
    "deer",
    "insect",
    "ant",
    "spider",
    "lizard",
    "snake",
}
MAX_MATCH_CANDIDATES = 120
MAX_USES_PER_CLIP = 2
EARLY_UNIQUE_SCENES = 10
WOMEN_TERMS = {
    "woman",
    "women",
    "female",
    "girl",
    "girls",
    "lady",
    "ladies",
    "mother",
    "mom",
    "wife",
    "bride",
    "her",
    "she",
}

_CLIP_MODEL_CACHE = None
_CLIP_PROCESSOR_CACHE = None
_CLIP_BACKEND_FAILED = False
_EMBEDDER_CACHE = None
_EMBEDDER_BACKEND_FAILED = False

NLIY_PHRASE = "nature lives in you"
NLIY_ALLOWED_NAME_FRAGMENTS = {
    "nature-lives-in-you-universe",
    "nature-mirror-nature-lives-in-you",
    # Backward-compatible filename variant already present in some libraries.
    "nature-mirror-nature-lives",
}


def suggest_scene_keywords(text: str, max_keywords: int = 4) -> list[str]:
    """Extract keywords from text, prioritizing noun-like words and word frequency.
    
    Higher-weight keywords (e.g., nouns, verbs) appear earlier in the list.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered = [w for w in words if len(w) > 2 and w not in STOP_WORDS and w not in NON_VISUAL_TERMS]
    if not filtered:
        return []

    frequency: dict[str, int] = {}
    for word in filtered:
        frequency[word] = frequency.get(word, 0) + 1

    # Prioritize longer words (likely more specific/meaningful) and higher frequency
    ordered = sorted(
        frequency.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0])
    )
    return [word for word, _ in ordered[:max_keywords]]


def _scene_query(text: str) -> str:
    keywords = suggest_scene_keywords(text, max_keywords=4)
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
    """Score how well a clip matches the scene's keywords.
    
    Weights earlier/more-important keywords more heavily.
    Penalizes clips that have no keyword matches.
    """
    query_keywords = suggest_scene_keywords(scene_query, max_keywords=6)
    clip_tokens = _tokenize(clip_text)
    
    if not query_keywords or not clip_tokens:
        return 0.0
    
    total_score = 0.0
    # Weight earlier keywords more heavily (they're more important)
    for position, kw in enumerate(query_keywords):
        if kw in clip_tokens:
            # Position weight: first keyword worth 0.5, others decay
            position_weight = 1.0 / (1 + position * 0.4)
            total_score += 0.15 * position_weight
    
    # Strong boost if multiple keywords match
    hit_count = sum(1 for kw in query_keywords if kw in clip_tokens)
    if hit_count >= 2:
        total_score += 0.15  # Bonus for diverse keyword coverage
    
    return min(0.75, total_score)


def _keyword_relevance_score_fast(keywords: list[str], clip_tokens: set[str]) -> float:
    """Fast version using pre-computed keywords and tokens. Avoids redundant tokenization."""
    if not keywords or not clip_tokens:
        return 0.0
    
    total_score = 0.0
    # Weight earlier keywords more heavily (they're more important)
    for position, kw in enumerate(keywords):
        if kw in clip_tokens:
            # Position weight: first keyword worth 0.5, others decay
            position_weight = 1.0 / (1 + position * 0.4)
            total_score += 0.15 * position_weight
    
    # Strong boost if multiple keywords match
    hit_count = sum(1 for kw in keywords if kw in clip_tokens)
    if hit_count >= 2:
        total_score += 0.15  # Bonus for diverse keyword coverage
    
    return min(0.75, total_score)


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


def _theme_alignment_score_fast(scene_tokens: set[str], clip_tokens: set[str]) -> float:
    """Fast version using pre-computed tokens. Avoids redundant tokenization."""
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
    return {w for w in words if len(w) > 2 and w not in STOP_WORDS and w not in NON_VISUAL_TERMS}


def _normalize_name_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")


def _nlily_allowed_indices(clip_paths: list[Path]) -> set[int]:
    allowed: set[int] = set()
    for idx, path in enumerate(clip_paths):
        stem_norm = _normalize_name_token(path.stem)
        if any(fragment in stem_norm for fragment in NLIY_ALLOWED_NAME_FRAGMENTS):
            allowed.add(idx)
            continue

        folder_norms = {_normalize_name_token(part) for part in path.parts}
        if _normalize_name_token(NLIY_PHRASE) in folder_norms and (
            "universe" in stem_norm or "mirror" in stem_norm
        ):
            allowed.add(idx)
    return allowed


def _singularize_token(token: str) -> str:
    t = (token or "").strip().lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _extract_subject_candidates_from_text(text: str, max_terms: int = 8) -> list[str]:
    words = [
        _singularize_token(w)
        for w in re.findall(r"[a-zA-Z']+", (text or "").lower())
    ]
    single_terms: list[str] = []
    for w in words:
        if len(w) < 3 or w in STOP_WORDS:
            continue
        if w in HIGH_VALUE_SUBJECT_TERMS:
            single_terms.append(w)

    phrase_terms: list[str] = []
    for i in range(len(words) - 1):
        left = words[i]
        right = words[i + 1]
        if right in COMPOUND_SUBJECT_HEADS and len(left) >= 4 and left not in STOP_WORDS:
            phrase_terms.append(f"{left} {right}")

    ordered: list[str] = []
    seen: set[str] = set()
    for term in [*phrase_terms, *single_terms]:
        if term in seen:
            continue
        seen.add(term)
        ordered.append(term)
        if len(ordered) >= max_terms:
            break
    return ordered


def _extract_scene_subject_terms(scene_text: str, scene_query: str) -> list[str]:
    extracted = _extract_subject_candidates_from_text(scene_text, max_terms=8)
    extracted.extend(_extract_subject_candidates_from_text(scene_query, max_terms=6))

    tokens: list[str] = []
    for raw in suggest_scene_keywords(scene_text, max_keywords=8) + suggest_scene_keywords(scene_query, max_keywords=8):
        token = _singularize_token(raw)
        if len(token) < 3 or token in STOP_WORDS or token in NON_VISUAL_TERMS:
            continue
        tokens.append(token)

    subject_terms: list[str] = []
    seen: set[str] = set()
    for term in extracted:
        if term in seen:
            continue
        subject_terms.append(term)
        seen.add(term)

    for token in tokens:
        if token in NON_VISUAL_TERMS or token in GENERIC_STYLE_TERMS:
            continue
        if token in HIGH_VALUE_SUBJECT_TERMS or len(token) >= 6:
            if token in seen:
                continue
            seen.add(token)
            subject_terms.append(token)

    # Remove generic style-only terms so subject matching stays concrete.
    subject_terms = [t for t in subject_terms if t not in GENERIC_STYLE_TERMS]
    return subject_terms[:5]


def _subject_alignment_score(scene_subject_terms: list[str], clip_singular: set[str]) -> float:
    """Strongly reward clips that contain concrete scene subjects, penalize misses."""
    if not scene_subject_terms:
        return 0.0

    required_entities = [
        term for term in scene_subject_terms
        if all(_singularize_token(w) in HIGH_VALUE_SUBJECT_TERMS for w in term.split())
    ]

    match_count = 0
    for term in scene_subject_terms:
        words = [w for w in term.split() if w]
        if not words:
            continue
        if all(_singularize_token(w) in clip_singular for w in words):
            match_count += 1

    # If a concrete species/entity is requested, strongly penalize clips that miss it.
    if required_entities and match_count <= 0:
        return -1.10

    if match_count <= 0:
        return -0.55
    return min(0.80, 0.45 + (match_count - 1) * 0.16)


def _women_presence_penalty(scene_tokens: set[str], clip_tokens: set[str]) -> float:
    """Penalize women-centric clips unless the scene text explicitly asks for that subject."""
    if not clip_tokens.intersection(WOMEN_TERMS):
        return 0.0
    if scene_tokens.intersection(WOMEN_TERMS):
        return 0.0
    return -0.80


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
    global _CLIP_MODEL_CACHE, _CLIP_PROCESSOR_CACHE, _CLIP_BACKEND_FAILED
    if _CLIP_MODEL_CACHE is not None and _CLIP_PROCESSOR_CACHE is not None:
        return _CLIP_MODEL_CACHE, _CLIP_PROCESSOR_CACHE
    if _CLIP_BACKEND_FAILED:
        return None, None

    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch  # noqa: F401
    except ImportError:
        _CLIP_BACKEND_FAILED = True
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
        _CLIP_MODEL_CACHE = model
        _CLIP_PROCESSOR_CACHE = processor
        if log:
            log("CLIP visual matcher ready — matching frames to scene descriptions.")
        return _CLIP_MODEL_CACHE, _CLIP_PROCESSOR_CACHE
    except Exception as exc:
        _CLIP_BACKEND_FAILED = True
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
        try:
            from moviepy.editor import VideoFileClip
        except Exception:
            from moviepy import VideoFileClip
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
    global _EMBEDDER_CACHE, _EMBEDDER_BACKEND_FAILED
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE
    if _EMBEDDER_BACKEND_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        _EMBEDDER_BACKEND_FAILED = True
        if log:
            log("AI matcher fallback: sentence-transformers not installed, using keyword overlap.")
        return None

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        _EMBEDDER_BACKEND_FAILED = True
        if log:
            log(f"AI matcher fallback: embedding model unavailable ({exc}), using keyword overlap.")
        return None

    _EMBEDDER_CACHE = model
    if log:
        log("AI matcher: semantic embedding model loaded (all-MiniLM-L6-v2).")
    return _EMBEDDER_CACHE


def _semantic_scores(model, scene_text: str, clip_texts: list[str]) -> list[float]:
    scene_vec = model.encode([scene_text], normalize_embeddings=True)[0]
    clip_vecs = model.encode(clip_texts, normalize_embeddings=True)
    return [float((scene_vec * vec).sum()) for vec in clip_vecs]


def _semantic_clip_vectors(model, clip_texts: list[str]):
    return model.encode(clip_texts, normalize_embeddings=True)


def _semantic_scores_with_clip_vectors(model, scene_text: str, clip_vecs) -> list[float]:
    scene_vec = model.encode([scene_text], normalize_embeddings=True)[0]
    return [float((scene_vec * vec).sum()) for vec in clip_vecs]


def _plan_focus_tokens(plan: list[PlannedSegment]) -> set[str]:
    tokens: set[str] = set()
    for seg in plan[:80]:
        text = (seg.text or "").strip()
        query = (seg.visual_query or "").strip()
        for word in suggest_scene_keywords(text, max_keywords=6):
            tokens.add(_singularize_token(word))
        for word in suggest_scene_keywords(query, max_keywords=6):
            tokens.add(_singularize_token(word))
        for subject in _extract_scene_subject_terms(text, query):
            for part in subject.split():
                tokens.add(_singularize_token(part))
    return {t for t in tokens if t and t not in STOP_WORDS}


def _prune_candidate_pool(
    plan: list[PlannedSegment],
    clip_paths: list[Path],
    clip_descriptions: list[str],
    clip_tokens_list: list[set[str]],
    clip_sources: list[str],
    log: callable | None,
    max_candidates: int = MAX_MATCH_CANDIDATES,
) -> tuple[list[Path], list[str], list[set[str]], list[str]]:
    if len(clip_paths) <= max_candidates:
        return clip_paths, clip_descriptions, clip_tokens_list, clip_sources

    focus_tokens = _plan_focus_tokens(plan)
    if not focus_tokens:
        # Keep newest clips if no textual signal is available.
        ranked_indices = sorted(
            range(len(clip_paths)),
            key=lambda i: clip_paths[i].stat().st_mtime if clip_paths[i].exists() else 0.0,
            reverse=True,
        )
        keep = set(ranked_indices[:max_candidates])
    else:
        scored: list[tuple[float, int]] = []
        for idx, clip_toks in enumerate(clip_tokens_list):
            singular_tokens = {_singularize_token(tok) for tok in clip_toks}
            overlap = len(singular_tokens.intersection(focus_tokens))
            subject_hits = len(singular_tokens.intersection(HIGH_VALUE_SUBJECT_TERMS))
            source_bonus = 0.08 if clip_sources[idx] == "stock" else 0.0
            score = overlap * 1.0 + subject_hits * 0.45 + source_bonus
            scored.append((score, idx))

        # Global top-k only: no hard source quotas.
        all_ranked = [idx for _s, idx in sorted(scored, key=lambda x: x[0], reverse=True)]
        keep = set(all_ranked[:max_candidates])

    selected_paths: list[Path] = []
    selected_desc: list[str] = []
    selected_tokens: list[set[str]] = []
    selected_sources: list[str] = []
    for idx, path in enumerate(clip_paths):
        if idx not in keep:
            continue
        selected_paths.append(path)
        selected_desc.append(clip_descriptions[idx])
        selected_tokens.append(clip_tokens_list[idx])
        selected_sources.append(clip_sources[idx])

    if log:
        local_kept = sum(1 for s in selected_sources if s == "local")
        stock_kept = sum(1 for s in selected_sources if s == "stock")
        log(
            "Clip candidate pruning: "
            f"{len(clip_paths)} -> {len(selected_paths)} "
            f"(local={local_kept}, stock={stock_kept})"
        )
    return selected_paths, selected_desc, selected_tokens, selected_sources


def _select_best_index(
    base_scores: list[float],
    usage_count: list[int],
    recent_indices: list[int],
    clip_sources: list[str] | None = None,
    allowed_indices: set[int] | None = None,
) -> int:
    """Select the best clip, heavily favoring variety and balancing local vs. stock footage.
    
    Uses pre-computed sets for O(1) membership tests instead of O(n) list lookups.
    
    Args:
        base_scores: Visual matching scores per clip
        usage_count: How many times each clip has been used
        recent_indices: Recent clip indices used (for recency penalty)
        clip_sources: Source type per clip ('local' or 'stock'), or None if not available
    """
    ranked: list[tuple[float, int]] = []
    
    # Convert to sets for O(1) membership testing (faster than list `in` checks)
    recent_window_set = set(recent_indices[-6:])
    recent_8 = recent_indices[-8:]
    recent_8_set = set(recent_8)
    recent_20_set = set(recent_indices[-20:])
    last_idx = recent_indices[-1] if recent_indices else None
    
    # Count local vs. stock usage in recent selections
    for idx, score in enumerate(base_scores):
        if allowed_indices is not None and idx not in allowed_indices:
            continue
        # Aggressive diversity penalty: exponential growth as usage increases
        # This strongly prevents clip repetition, especially for local clips
        base_diversity_penalty = (usage_count[idx] ** 1.5) * 0.22
        additional_reuse = max(0, usage_count[idx] - 3)
        diversity_penalty = base_diversity_penalty + additional_reuse * 0.35
        
        # Recency penalties - prevent immediate repeats
        if last_idx is not None and idx == last_idx:
            recency_penalty = 1.2  # Strong penalty for immediate repeat
        elif len(recent_indices) >= 2 and idx == recent_indices[-2]:
            recency_penalty = 0.75  # Strong penalty for recent use
        elif idx in recent_window_set:
            recency_penalty = 0.45  # Moderate penalty
        else:
            recency_penalty = 0.0
        
        # Streak penalty: if a clip appears multiple times in the recent window
        # Count manually from small list for efficiency
        recent_hits = recent_8.count(idx)
        streak_penalty = 0.0
        if recent_hits >= 2:
            streak_penalty = 0.35 * (recent_hits - 1)
        
        final_score = score - diversity_penalty - recency_penalty - streak_penalty
        ranked.append((final_score, idx))
    
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return 0
    
    top_score, top_idx = ranked[0]
    
    # Prefer variety: if top pick is very recent and another is close, pick the fresh one
    if top_idx in recent_window_set:
        for alt_score, alt_idx in ranked[1:8]:
            if alt_idx not in recent_window_set and (top_score - alt_score) <= 0.35:
                return alt_idx
    
    # Avoid immediate A-A repeats unless score gap is very large
    if last_idx is not None and top_idx == last_idx:
        for alt_score, alt_idx in ranked[1:8]:
            if alt_idx != last_idx and (top_score - alt_score) <= 0.50:
                return alt_idx
    
    # If top is recent and good alternatives exist, prefer fresh material
    for alt_score, alt_idx in ranked[1:6]:
        if top_idx in recent_window_set and alt_idx not in recent_window_set and (top_score - alt_score) <= 0.20:
            return alt_idx
    
    return top_idx


def list_video_clips(clips_folder: Path | None) -> list[Path]:
    if clips_folder is None or not clips_folder.exists():
        return []
    ignored_parts = {"_stock_cache", "_clip_cache", "__pycache__", "batch_logs"}
    clips = [
        p
        for p in sorted(clips_folder.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
        and not any(part in ignored_parts for part in p.parts)
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
    scene_shortlist: int = 28,
    avoid_clip_repetition: bool = True,
    clip_repeat_cooldown: int = 8,
) -> list[TimelineClip]:
    if not plan or not clip_paths:
        return []

    visual_plan, plan_indices = _build_visual_plan(plan)
    if not visual_plan:
        return []

    clip_descriptions = [_clip_description(path) for path in clip_paths]
    # Pre-compute all clip tokens once to avoid redundant tokenization
    clip_tokens_list = [_tokenize(desc) for desc in clip_descriptions]
    clip_singular_tokens_list = [{_singularize_token(tok) for tok in toks} for toks in clip_tokens_list]

    nliy_allowed_indices = _nlily_allowed_indices(clip_paths)
    if log and nliy_allowed_indices:
        names = ", ".join(clip_paths[i].name for i in sorted(nliy_allowed_indices))
        log(f"Nature-Lives-In-You scene rule active with clips: {names}")

    # Determine which clips are local vs. stock (stock clips typically in _stock_cache)
    clip_sources = ["stock" if "_stock_cache" in str(path) else "local" for path in clip_paths]
    clip_paths, clip_descriptions, clip_tokens_list, clip_sources = _prune_candidate_pool(
        plan=visual_plan,
        clip_paths=clip_paths,
        clip_descriptions=clip_descriptions,
        clip_tokens_list=clip_tokens_list,
        clip_sources=clip_sources,
        log=log,
    )
    clip_singular_tokens_list = [{_singularize_token(tok) for tok in toks} for toks in clip_tokens_list]

    # Resolve cache dir relative to the selected clip pool.
    clip_cache_dir = clip_paths[0].parent / "_clip_cache"

    local_indices = {idx for idx, source in enumerate(clip_sources) if source == "local"}
    stock_indices = {idx for idx, source in enumerate(clip_sources) if source == "stock"}

    # Tier 1: CLIP visual matching with lazy per-scene embedding (runtime-friendly).
    clip_model, clip_processor = _load_clip_backend(log)
    clip_embeddings: list = [None for _ in clip_paths]

    # Tier 2: sentence-transformers text embedding (fallback when CLIP unavailable).
    embedder = None if clip_model is not None else _load_embedding_backend(log)
    semantic_clip_vecs = _semantic_clip_vectors(embedder, clip_descriptions) if embedder is not None else None

    if clip_model is not None:
        backend = "clip"
    elif embedder is not None:
        backend = "sentence-transformers"
    else:
        backend = "keyword"
    if log:
        log(f"B-roll matching backend: {backend}")
        if clip_model is not None:
            log(
                "CLIP runtime mode: lazy embedding on scene shortlist "
                f"(k={max(8, min(int(scene_shortlist), len(clip_paths)))})"
            )
        if avoid_clip_repetition:
            log(f"Clip repetition guard: enabled (cooldown={max(2, int(clip_repeat_cooldown))})")

    timeline: list[TimelineClip] = []
    usage_count = [0 for _ in clip_paths]
    recent_indices: list[int] = []
    cursor = 0.0

    for scene_idx, (segment, pidx) in enumerate(zip(visual_plan, plan_indices), start=1):
        import numpy as np

        scene_text = (segment.text or "").strip() or "voiceover"
        scene_query = (segment.visual_query or "").strip() or _scene_query(scene_text)
        
        # Cache keywords for this scene (used in logging and scoring)
        scene_keywords = suggest_scene_keywords(scene_text)
        scene_query_tokens = _tokenize(scene_query)
        scene_subject_terms = _extract_scene_subject_terms(scene_text, scene_query)
        enforce_nliy = NLIY_PHRASE in scene_text.lower() or NLIY_PHRASE in scene_query.lower()

        movement_terms = {"running", "walking", "flying", "waves", "action", "dynamic", "motion", "drone", "timelapse", "closeup", "close-up"}

        # Fast metadata pass first: shortlist the most plausible candidates.
        fast_scores = []
        for clip_toks, clip_singular_toks in zip(clip_tokens_list, clip_singular_tokens_list):
            movement_score = min(0.35, len(clip_toks.intersection(movement_terms)) * 0.08)
            kw_score = _keyword_relevance_score_fast(scene_keywords, clip_toks)
            theme_score = _theme_alignment_score_fast(scene_query_tokens, clip_toks)
            subject_score = _subject_alignment_score(scene_subject_terms, clip_singular_toks)
            fast_scores.append(
                movement_score * 0.35
                + kw_score * 0.75
                + theme_score * 0.35
                + subject_score * 0.85
            )

        shortlist_size = max(8, min(int(scene_shortlist), len(clip_paths)))
        shortlist_indices = sorted(
            range(len(fast_scores)),
            key=lambda idx: fast_scores[idx],
            reverse=True,
        )[:shortlist_size]
        shortlist_set = set(shortlist_indices)

        if clip_model is not None:
            text_vec = _clip_text_embedding(clip_model, clip_processor, scene_query)
            base_scores = [0.0] * len(clip_paths)
            if text_vec is not None:
                raw_scores: dict[int, float] = {}
                for idx in shortlist_indices:
                    vec = clip_embeddings[idx]
                    if vec is None:
                        vec = _build_clip_image_embedding(
                            clip_model,
                            clip_processor,
                            clip_paths[idx],
                            clip_cache_dir,
                        )
                        clip_embeddings[idx] = vec
                    raw_scores[idx] = float(np.dot(text_vec, vec)) if vec is not None else 0.0

                if raw_scores:
                    min_s = min(raw_scores.values())
                    max_s = max(raw_scores.values())
                    spread = max(max_s - min_s, 1e-6)
                    for idx, value in raw_scores.items():
                        base_scores[idx] = (value - min_s) / spread

        elif embedder is not None:
            base_scores = _semantic_scores_with_clip_vectors(embedder, scene_query, semantic_clip_vecs)
        else:
            base_scores = [_lexical_similarity(scene_query, desc) for desc in clip_descriptions]

        # Blend normalised visual score with improved metadata signals.
        # Pre-compute scoring components for each clip efficiently.
        scores = []
        for idx, (base, clip_toks, clip_singular_toks, fast_score) in enumerate(
            zip(base_scores, clip_tokens_list, clip_singular_tokens_list, fast_scores)
        ):
            # Movement score: intersection with movement terms
            movement_score = min(0.35, len(clip_toks.intersection(movement_terms)) * 0.08)
            
            # Keyword relevance using pre-computed query keywords
            kw_score = _keyword_relevance_score_fast(scene_keywords, clip_toks)
            
            # Theme alignment
            theme_score = _theme_alignment_score_fast(scene_query_tokens, clip_toks)
            subject_score = _subject_alignment_score(scene_subject_terms, clip_singular_toks)
            women_penalty = _women_presence_penalty(scene_query_tokens, clip_toks)

            if clip_model is not None and idx not in shortlist_set:
                scores.append(float("-inf"))
                continue

            scores.append(
                base * 0.45
                + movement_score * 0.30
                + kw_score * 0.55
                + theme_score * 0.35
                + subject_score
                + fast_score * 0.20
                + women_penalty
            )

        preferred_pool: str | None = None
        local_best = max((scores[i] for i in local_indices), default=float("-inf"))
        stock_best = max((scores[i] for i in stock_indices), default=float("-inf"))
        if local_best != float("-inf") and stock_best != float("-inf"):
            if stock_best >= local_best + 0.06:
                preferred_pool = "stock"
            elif local_best >= stock_best + 0.06:
                preferred_pool = "local"

        shortlist_allowed = shortlist_set if clip_model is not None else None
        stock_allowed = stock_indices.intersection(shortlist_set) if clip_model is not None else stock_indices
        local_allowed = local_indices.intersection(shortlist_set) if clip_model is not None else local_indices

        all_allowed = set(shortlist_allowed) if shortlist_allowed is not None else set(range(len(scores)))

        def _pool_best(candidate_pool: set[int]) -> float:
            finite_scores = [scores[idx] for idx in candidate_pool if scores[idx] != float("-inf")]
            return max(finite_scores) if finite_scores else float("-inf")

        def _repeat_safe_pool(candidate_pool: set[int]) -> set[int]:
            if not avoid_clip_repetition:
                return set(candidate_pool)

            best_any = _pool_best(candidate_pool)
            if best_any == float("-inf"):
                return set(candidate_pool)

            # Hard cap to prevent excessive looping of the same clip.
            capped = {idx for idx in candidate_pool if usage_count[idx] < MAX_USES_PER_CLIP}
            if capped:
                candidate_pool = capped
                best_any = _pool_best(candidate_pool)

            # During the opening scenes, force fresh clips when available.
            if scene_idx <= EARLY_UNIQUE_SCENES:
                early_unused = {idx for idx in candidate_pool if usage_count[idx] == 0}
                if early_unused:
                    return early_unused

            # Use unseen clips first. Only allow reuse once all candidates have been seen.
            unused = {idx for idx in candidate_pool if usage_count[idx] == 0}
            if unused:
                best_unused = _pool_best(unused)
                if best_any - best_unused <= 0.50:
                    return unused

            cooldown = max(2, int(clip_repeat_cooldown))
            recent_block = set(recent_indices[-cooldown:])
            cooled = {idx for idx in candidate_pool if idx not in recent_block}
            if cooled:
                best_cooled = _pool_best(cooled)
                if best_any - best_cooled <= 0.30:
                    return cooled

            # If the pool is tiny and exhausted, allow fallback to keep pipeline moving.
            return set(candidate_pool)

        stock_allowed = _repeat_safe_pool(set(stock_allowed)) if stock_allowed else set()
        local_allowed = _repeat_safe_pool(set(local_allowed)) if local_allowed else set()
        global_allowed = _repeat_safe_pool(all_allowed)

        if enforce_nliy and nliy_allowed_indices:
            forced = set(nliy_allowed_indices)
            if clip_model is not None:
                forced = forced.intersection(shortlist_set)
            finite_forced = {idx for idx in forced if scores[idx] != float("-inf")}
            if finite_forced:
                global_allowed = finite_forced
                stock_allowed = finite_forced.intersection(stock_indices)
                local_allowed = finite_forced.intersection(local_indices)
                preferred_pool = None
            elif log:
                log("Nature-Lives-In-You rule had no finite candidates in shortlist; using normal matching fallback.")

        if preferred_pool == "stock" and stock_allowed:
            chosen_idx = _select_best_index(
                scores,
                usage_count,
                recent_indices,
                clip_sources,
                allowed_indices=stock_allowed,
            )
        elif preferred_pool == "local" and local_allowed:
            chosen_idx = _select_best_index(
                scores,
                usage_count,
                recent_indices,
                clip_sources,
                allowed_indices=local_allowed,
            )
        else:
            chosen_idx = _select_best_index(
                scores,
                usage_count,
                recent_indices,
                clip_sources,
                allowed_indices=global_allowed,
            )

        # Quality override: avoid extremely weak semantic picks caused by repetition rules.
        if avoid_clip_repetition and global_allowed:
            finite_allowed = [idx for idx in global_allowed if scores[idx] != float("-inf")]
            if finite_allowed:
                best_idx = max(finite_allowed, key=lambda idx: scores[idx])
                if scores[chosen_idx] < 0.10 and (scores[best_idx] - scores[chosen_idx]) >= 0.55:
                    chosen_idx = best_idx

        clip_path = clip_paths[chosen_idx]
        usage_count[chosen_idx] += 1
        recent_indices.append(chosen_idx)

        if log:
            score_str = f"{scores[chosen_idx]:.3f}" if scores else "?"
            log(
                f"Scene {scene_idx}: [{backend}={score_str}] "
                f"keywords={', '.join(scene_keywords)} | "
                f"subjects={', '.join(scene_subject_terms)} | "
                f"query='{scene_query}' | emotion={segment.emotion} | "
                f"shortlist={len(shortlist_set)} | "
                f"nliy-rule={'on' if (enforce_nliy and nliy_allowed_indices) else 'off'} | "
                f"source-check(local={local_best:.3f}, stock={stock_best:.3f}, preferred={preferred_pool or 'mixed'}) | "
                f"chosen={clip_path.name} ({clip_sources[chosen_idx]})"
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


def find_low_confidence_segments(
    plan: list[PlannedSegment],
    clip_paths: list[Path],
    log: callable | None = None,
    min_best_score: float = 0.60,
    max_segments: int = 10,
    scene_shortlist: int = 28,
) -> list[PlannedSegment]:
    """Return segments where local clip pool appears weak for the planned scene query.

    This is used to decide whether stock fetching is needed when local clips exist.
    """
    if not plan:
        return []
    if not clip_paths:
        return list(plan[: max(1, int(max_segments))])

    visual_plan, _plan_indices = _build_visual_plan(plan)
    if not visual_plan:
        return []

    clip_descriptions = [_clip_description(path) for path in clip_paths]
    clip_tokens_list = [_tokenize(desc) for desc in clip_descriptions]
    clip_singular_tokens_list = [{_singularize_token(tok) for tok in toks} for toks in clip_tokens_list]
    clip_sources = ["stock" if "_stock_cache" in str(path) else "local" for path in clip_paths]
    clip_paths, clip_descriptions, clip_tokens_list, clip_sources = _prune_candidate_pool(
        plan=visual_plan,
        clip_paths=clip_paths,
        clip_descriptions=clip_descriptions,
        clip_tokens_list=clip_tokens_list,
        clip_sources=clip_sources,
        log=log,
        max_candidates=80,
    )
    clip_singular_tokens_list = [{_singularize_token(tok) for tok in toks} for toks in clip_tokens_list]

    # Keep low-confidence scan lightweight: avoid CLIP frame encoding here.
    embedder = _load_embedding_backend(log)
    semantic_clip_vecs = _semantic_clip_vectors(embedder, clip_descriptions) if embedder is not None else None
    backend = "sentence-transformers" if embedder is not None else "keyword"

    low_confidence: list[PlannedSegment] = []
    seen_queries: set[str] = set()
    scores_seen: list[float] = []
    movement_terms = {"running", "walking", "flying", "waves", "action", "dynamic", "motion", "drone", "timelapse", "closeup", "close-up"}

    for segment in visual_plan:
        scene_text = (segment.text or "").strip() or "voiceover"
        scene_query = (segment.visual_query or "").strip() or _scene_query(scene_text)
        
        # Cache for this scene
        scene_keywords = suggest_scene_keywords(scene_text)
        scene_query_tokens = _tokenize(scene_query)
        scene_subject_terms = _extract_scene_subject_terms(scene_text, scene_query)

        if embedder is not None:
            base_scores = _semantic_scores_with_clip_vectors(embedder, scene_query, semantic_clip_vecs)
        else:
            base_scores = [_lexical_similarity(scene_query, desc) for desc in clip_descriptions]

        shortlist_size = max(8, min(int(scene_shortlist), len(clip_paths)))
        fast_scores = []
        for clip_toks, clip_singular_toks in zip(clip_tokens_list, clip_singular_tokens_list):
            movement_score = min(0.35, len(clip_toks.intersection(movement_terms)) * 0.08)
            kw_score = _keyword_relevance_score_fast(scene_keywords, clip_toks)
            theme_score = _theme_alignment_score_fast(scene_query_tokens, clip_toks)
            subject_score = _subject_alignment_score(scene_subject_terms, clip_singular_toks)
            fast_scores.append(
                movement_score * 0.35
                + kw_score * 0.75
                + theme_score * 0.35
                + subject_score * 0.85
            )

        shortlist_indices = sorted(
            range(len(fast_scores)),
            key=lambda idx: fast_scores[idx],
            reverse=True,
        )[:shortlist_size]
        shortlist_set = set(shortlist_indices)

        # Efficient scoring with pre-computed tokens
        scores = []
        for idx, (base, clip_toks, clip_singular_toks, fast_score) in enumerate(
            zip(base_scores, clip_tokens_list, clip_singular_tokens_list, fast_scores)
        ):
            if idx not in shortlist_set:
                continue
            movement_score = min(0.35, len(clip_toks.intersection(movement_terms)) * 0.08)
            kw_score = _keyword_relevance_score_fast(scene_keywords, clip_toks)
            theme_score = _theme_alignment_score_fast(scene_query_tokens, clip_toks)
            subject_score = _subject_alignment_score(scene_subject_terms, clip_singular_toks)
            women_penalty = _women_presence_penalty(scene_query_tokens, clip_toks)
            scores.append(
                base * 0.4
                + movement_score * 0.4
                + kw_score * 0.4
                + theme_score * 0.4
                + subject_score
                + fast_score * 0.2
                + women_penalty
            )
        best = max(scores) if scores else 0.0
        scores_seen.append(best)

        if best < float(min_best_score):
            query_key = scene_query.lower().strip()
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            low_confidence.append(segment)
            if len(low_confidence) >= max(1, int(max_segments)):
                break

    if log:
        avg_best = (sum(scores_seen) / len(scores_seen)) if scores_seen else 0.0
        log(
            "Local coverage scan: "
            f"backend={backend} | avg-best-score={avg_best:.3f} | "
            f"weak-scenes={len(low_confidence)}"
        )

    return low_confidence
