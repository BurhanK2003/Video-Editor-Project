from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np

from .clip_indexer import ClipIndexer, SceneEmbedding
from .models import PlannedSegment

FAISS_MIN_SCENES = 400
FAISS_TOP_K = 80
FAST_CPU_MAX_LOCAL_CLIPS = 140


@dataclass
class SelectedClip:
    source_path: Path
    source_kind: str  # local | online
    provider: str
    score: float
    scene_start: float
    scene_end: float
    window_start: float
    window_end: float
    query: str


@dataclass
class _OnlineCandidate:
    provider: str
    title: str
    page_url: str
    video_url: str
    thumb_url: str
    query: str


class _LocalSceneRetriever:
    def __init__(self, scenes: list[SceneEmbedding], log: callable | None = None) -> None:
        self.scenes = scenes
        self.log = log or (lambda _msg: None)
        self.index = None
        self.using_faiss = False

        if not scenes:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return

        self.embeddings = np.vstack([scene.embedding.astype(np.float32) for scene in scenes])
        self._maybe_build_faiss()

    def _maybe_build_faiss(self) -> None:
        if len(self.scenes) < FAISS_MIN_SCENES:
            return
        try:
            import faiss
        except Exception:
            return

        dim = int(self.embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        self.index = index
        self.using_faiss = True
        self.log(f"Local scene search: FAISS enabled (scenes={len(self.scenes)}, dim={dim})")

    def best(self, text_vectors: np.ndarray) -> tuple[float, SceneEmbedding] | None:
        if text_vectors.size == 0 or not self.scenes:
            return None

        if self.using_faiss and self.index is not None:
            top_k = min(FAISS_TOP_K, len(self.scenes))
            scores, ids = self.index.search(text_vectors.astype(np.float32), top_k)
            candidate_indices: set[int] = set()
            for row in ids:
                for idx in row.tolist():
                    if idx >= 0:
                        candidate_indices.add(int(idx))
            if not candidate_indices:
                return None

            best_pair: tuple[float, SceneEmbedding] | None = None
            for idx in candidate_indices:
                scene = self.scenes[idx]
                score = _cosine_max(text_vectors, scene.embedding)
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, scene)
            return best_pair

        # Brute-force cosine for smaller libraries.
        sims = np.dot(self.embeddings, text_vectors.T)
        if sims.size == 0:
            return None
        scores = np.max(sims, axis=1)
        best_idx = int(np.argmax(scores))
        return float(scores[best_idx]), self.scenes[best_idx]


def build_semantic_queries(text: str, global_keywords: list[str], max_queries: int = 3) -> list[str]:
    cleaned = " ".join((text or "").split()).strip()
    base_tokens = re.findall(r"[a-zA-Z']+", cleaned.lower())
    base_tokens = [t for t in base_tokens if len(t) > 2]

    local_keywords = [
        t for t in base_tokens
        if t not in {"the", "and", "that", "with", "from", "this", "have", "into", "your", "they"}
    ]
    head = " ".join(local_keywords[:6]).strip() or "nature wildlife"

    globals_hint = " ".join(global_keywords[:4]).strip()
    queries = [
        f"{head} documentary nature",
        f"calm meditative {head}",
        f"wildlife {head} cinematic",
    ]
    if globals_hint:
        queries.append(f"{head} {globals_hint}")

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        norm = " ".join(query.split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped[:max_queries]


def _global_keywords_from_plan(plan: list[PlannedSegment], max_keywords: int = 16) -> list[str]:
    corpus = " ".join((seg.text or "") for seg in plan).lower()
    tokens = re.findall(r"[a-zA-Z']+", corpus)
    stop = {
        "the", "and", "that", "with", "from", "this", "have", "into", "your", "they", "are", "was", "were",
        "for", "you", "our", "their", "there", "just", "very", "then", "than", "about", "because",
    }
    freq: dict[str, int] = {}
    for token in tokens:
        if len(token) < 3 or token in stop:
            continue
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda it: (-it[1], -len(it[0]), it[0]))
    return [word for word, _ in ranked[:max_keywords]]


def _cosine_max(text_vectors: np.ndarray, scene_vec: np.ndarray) -> float:
    if text_vectors.size == 0 or scene_vec.size == 0:
        return -1.0
    scores = np.dot(text_vectors, scene_vec)
    return float(np.max(scores))


def _best_local_candidate(
    text_vectors: np.ndarray,
    retriever: _LocalSceneRetriever,
    query_used: str,
) -> SelectedClip | None:
    best = retriever.best(text_vectors)

    if best is None:
        return None

    score, scene = best
    return SelectedClip(
        source_path=scene.clip_path,
        source_kind="local",
        provider="local",
        score=score,
        scene_start=scene.scene_start,
        scene_end=scene.scene_end,
        window_start=scene.best_window_start,
        window_end=scene.best_window_end,
        query=query_used,
    )


async def _fetch_json(session, url: str, headers: dict[str, str] | None = None) -> dict:
    try:
        async with session.get(url, headers=headers, timeout=20) as response:
            if response.status != 200:
                return {}
            text = await response.text()
            return json.loads(text)
    except Exception:
        return {}


async def _search_pexels(session, queries: list[str], api_key: str) -> list[_OnlineCandidate]:
    if not api_key:
        return []

    out: list[_OnlineCandidate] = []
    headers = {"Authorization": api_key}
    for query in queries:
        url = (
            "https://api.pexels.com/videos/search"
            f"?query={quote_plus(query)}&per_page=6&orientation=landscape"
        )
        payload = await _fetch_json(session, url, headers=headers)
        for item in payload.get("videos", [])[:6]:
            files = item.get("video_files", [])
            video_url = ""
            for vf in files:
                if str(vf.get("file_type") or "") == "video/mp4":
                    video_url = str(vf.get("link") or "")
                    if video_url:
                        break
            if not video_url:
                continue
            thumb = str(item.get("image") or "")
            out.append(
                _OnlineCandidate(
                    provider="pexels",
                    title=str(item.get("url") or ""),
                    page_url=str(item.get("url") or ""),
                    video_url=video_url,
                    thumb_url=thumb,
                    query=query,
                )
            )
    return out


async def _search_pixabay(session, queries: list[str], api_key: str) -> list[_OnlineCandidate]:
    if not api_key:
        return []

    out: list[_OnlineCandidate] = []
    for query in queries:
        url = (
            "https://pixabay.com/api/videos/"
            f"?key={quote_plus(api_key)}&q={quote_plus(query)}&per_page=6"
        )
        payload = await _fetch_json(session, url)
        for item in payload.get("hits", [])[:6]:
            variants = item.get("videos", {}) or {}
            video_url = ""
            for key in ("large", "medium", "small", "tiny"):
                if key in variants and variants[key].get("url"):
                    video_url = str(variants[key]["url"])
                    break
            if not video_url:
                continue
            thumb = str(item.get("videos", {}).get("tiny", {}).get("thumbnail") or item.get("picture_id") or "")
            if thumb and not thumb.startswith("http"):
                thumb = f"https://i.vimeocdn.com/video/{thumb}_295x166.jpg"
            out.append(
                _OnlineCandidate(
                    provider="pixabay",
                    title=str(item.get("tags") or ""),
                    page_url=str(item.get("pageURL") or ""),
                    video_url=video_url,
                    thumb_url=thumb,
                    query=query,
                )
            )
    return out


async def _download_image_rgb(session, url: str) -> np.ndarray | None:
    if not url:
        return None
    try:
        import cv2
    except Exception:
        return None

    try:
        async with session.get(url, timeout=20) as response:
            if response.status != 200:
                return None
            blob = await response.read()
        arr = np.frombuffer(blob, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


async def _best_online_candidate(
    session,
    indexer: ClipIndexer,
    text_vectors: np.ndarray,
    candidates: list[_OnlineCandidate],
) -> tuple[SelectedClip | None, list[_OnlineCandidate]]:
    if not candidates:
        return None, []

    images: list[np.ndarray] = []
    valid: list[_OnlineCandidate] = []
    for candidate in candidates:
        img = await _download_image_rgb(session, candidate.thumb_url)
        if img is None:
            continue
        images.append(img)
        valid.append(candidate)

    if not valid:
        return None, []

    vectors = indexer.encode_images(images)
    best_idx = -1
    best_score = -1.0
    for idx, vec in enumerate(vectors):
        score = _cosine_max(text_vectors, vec)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None, valid

    picked = valid[best_idx]
    selected = SelectedClip(
        source_path=Path(picked.video_url),
        source_kind="online",
        provider=picked.provider,
        score=float(best_score),
        scene_start=0.0,
        scene_end=0.0,
        window_start=0.0,
        window_end=3.0,
        query=picked.query,
    )
    return selected, valid


async def _download_online_clip(session, candidate: SelectedClip, cache_dir: Path) -> Path | None:
    url = str(candidate.source_path)
    if not url.startswith("http"):
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".mp4"
    file_name = f"online_{abs(hash(url)) % (10**12)}{suffix}"
    destination = cache_dir / file_name
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    part = destination.with_suffix(".part")
    try:
        async with session.get(url, timeout=60) as response:
            if response.status != 200:
                return None
            data = await response.read()
        with part.open("wb") as handle:
            handle.write(data)
        part.replace(destination)
        return destination
    except Exception:
        if part.exists():
            part.unlink(missing_ok=True)
        return None


async def _search_segment_parallel(
    segment: PlannedSegment,
    queries: list[str],
    text_vectors: np.ndarray,
    indexer: ClipIndexer,
    local_retriever: _LocalSceneRetriever,
    pexels_key: str,
    pixabay_key: str,
    online_cache_dir: Path,
    use_online: bool,
    log: callable | None,
) -> SelectedClip | None:
    if not queries:
        queries = ["calm nature wildlife"]
    if text_vectors.size == 0:
        text_vectors = indexer.encode_texts(queries)

    local_task = asyncio.to_thread(_best_local_candidate, text_vectors, local_retriever, queries[0])

    async def online_task():
        if not use_online:
            return None
        try:
            import aiohttp
        except Exception:
            return None

        async with aiohttp.ClientSession() as session:
            search_jobs = [
                _search_pexels(session, queries, pexels_key),
                _search_pixabay(session, queries, pixabay_key),
            ]
            results = await asyncio.gather(*search_jobs)
            merged: list[_OnlineCandidate] = []
            for batch in results:
                merged.extend(batch)
            best_online, _valid = await _best_online_candidate(session, indexer, text_vectors, merged)
            if best_online is None:
                return None
            downloaded = await _download_online_clip(session, best_online, online_cache_dir)
            if downloaded is None:
                return None

            # Score best window after download using the same local indexer pipeline.
            scenes = await asyncio.to_thread(indexer.index_clip, downloaded)
            if not scenes:
                return None
            best_scene = max(scenes, key=lambda s: _cosine_max(text_vectors, s.embedding))
            best_online.source_path = downloaded
            best_online.scene_start = best_scene.scene_start
            best_online.scene_end = best_scene.scene_end
            best_online.window_start = best_scene.best_window_start
            best_online.window_end = best_scene.best_window_end
            best_online.score = _cosine_max(text_vectors, best_scene.embedding)
            return best_online

    local_selected, online_selected = await asyncio.gather(local_task, online_task())

    if local_selected is None and online_selected is None:
        return None
    if local_selected is None:
        if log:
            log(f"Scene search: online picked ({online_selected.provider}) score={online_selected.score:.3f}")
        return online_selected
    if online_selected is None:
        if log:
            log(f"Scene search: local picked score={local_selected.score:.3f}")
        return local_selected

    if online_selected.score > local_selected.score + 0.05:
        if log:
            log(
                "Scene search: online preferred over local "
                f"(online={online_selected.score:.3f}, local={local_selected.score:.3f})"
            )
        return online_selected

    if log:
        log(
            "Scene search: local kept "
            f"(local={local_selected.score:.3f}, online={online_selected.score:.3f})"
        )
    return local_selected


async def select_best_clips_async(
    plan: list[PlannedSegment],
    clips_folder: Path | None,
    output_path: Path,
    allow_online: bool,
    pexels_key: str,
    pixabay_key: str,
    clip_model_name: str,
    log: callable | None = None,
) -> list[SelectedClip]:
    cpu_count = max(1, int(os.cpu_count() or 4))
    if clips_folder is None or not clips_folder.exists():
        local_clips: list[Path] = []
    else:
        indexer_probe = ClipIndexer(clips_folder=clips_folder, model_name=clip_model_name, log=log)
        local_clips = indexer_probe.list_local_clips()

    if cpu_count <= 8 and len(local_clips) > FAST_CPU_MAX_LOCAL_CLIPS:
        local_clips = sorted(
            local_clips,
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )[:FAST_CPU_MAX_LOCAL_CLIPS]
        if log:
            log(
                "Runtime optimization: limiting advanced CLIP index to "
                f"{len(local_clips)} most recent local clips (CPU={cpu_count})."
            )

    if not local_clips and not allow_online:
        return []

    working_folder = clips_folder if clips_folder and clips_folder.exists() else (output_path.parent / "_temp_local_index")
    working_folder.mkdir(parents=True, exist_ok=True)
    indexer = ClipIndexer(clips_folder=working_folder, model_name=clip_model_name, log=log)

    # If local clips exist but indexer root differs, explicitly pass paths.
    local_scenes = await asyncio.to_thread(indexer.build_local_scene_index, local_clips if local_clips else None)
    local_retriever = _LocalSceneRetriever(local_scenes, log=log)

    # Phase 2 speedup: encode all unique segment queries in one batch.
    global_keywords = _global_keywords_from_plan(plan)
    max_queries_per_segment = 2 if cpu_count <= 8 else 3
    segment_queries: list[list[str]] = []
    query_bank: list[str] = []
    seen_queries: set[str] = set()
    for segment in plan:
        queries = build_semantic_queries(
            segment.text or segment.visual_query,
            global_keywords,
            max_queries=max_queries_per_segment,
        )
        if not queries:
            queries = ["calm nature wildlife"]
        segment_queries.append(queries)
        for query in queries:
            if query in seen_queries:
                continue
            seen_queries.add(query)
            query_bank.append(query)

    query_vectors_map: dict[str, np.ndarray] = {}
    if query_bank:
        encoded = indexer.encode_texts(query_bank)
        for idx, query in enumerate(query_bank):
            query_vectors_map[query] = encoded[idx]

    if log:
        log(
            "Phase 2 optimization: "
            f"batched_query_vectors={len(query_bank)}, local_scenes={len(local_scenes)}, "
            f"faiss={'on' if local_retriever.using_faiss else 'off'}"
        )

    selected: list[SelectedClip] = []
    online_cache = output_path.parent / "_stock_cache" / "async_online"
    allow_online_runtime = bool(allow_online)
    if allow_online and cpu_count <= 8 and len(plan) >= 12:
        allow_online_runtime = False
        if log:
            log("Runtime optimization: skipping advanced per-scene online search on slower CPU.")

    for seg_idx, segment in enumerate(plan):
        queries = segment_queries[seg_idx] if seg_idx < len(segment_queries) else ["calm nature wildlife"]
        text_vectors = np.vstack([query_vectors_map[q] for q in queries if q in query_vectors_map]).astype(np.float32)
        pick = await _search_segment_parallel(
            segment=segment,
            queries=queries,
            text_vectors=text_vectors,
            indexer=indexer,
            local_retriever=local_retriever,
            pexels_key=pexels_key,
            pixabay_key=pixabay_key,
            online_cache_dir=online_cache,
            use_online=allow_online_runtime,
            log=log,
        )
        if pick is None:
            continue
        selected.append(pick)

    return selected


def select_best_clips(
    plan: list[PlannedSegment],
    clips_folder: Path | None,
    output_path: Path,
    allow_online: bool,
    pexels_key: str,
    pixabay_key: str,
    clip_model_name: str,
    log: callable | None = None,
) -> list[SelectedClip]:
    return asyncio.run(
        select_best_clips_async(
            plan=plan,
            clips_folder=clips_folder,
            output_path=output_path,
            allow_online=allow_online,
            pexels_key=pexels_key,
            pixabay_key=pixabay_key,
            clip_model_name=clip_model_name,
            log=log,
        )
    )
