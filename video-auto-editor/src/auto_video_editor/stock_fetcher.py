from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import PlannedSegment

STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "from",
    "have",
    "into",
    "just",
    "like",
    "more",
    "over",
    "some",
    "than",
    "that",
    "their",
    "there",
    "they",
    "this",
    "those",
    "very",
    "voiceover",
    "with",
    "would",
}
DEFAULT_QUERIES = ["background", "nature", "business", "people"]
MIN_DOWNLOADS_PER_KEYWORD = 3
MAX_RESULTS_PER_KEYWORD = 5
USER_AGENT = "local-auto-video-editor/1.0"


def fetch_stock_clips(
    plan: list[PlannedSegment],
    output_path: Path,
    width: int,
    height: int,
    keywords_override: str,
    log: callable,
) -> list[Path]:
    pexels_key = _validated_api_key("PEXELS_API_KEY", log)
    pixabay_key = _validated_api_key("PIXABAY_API_KEY", log)
    if not pexels_key and not pixabay_key:
        log("Stock fetch skipped: no provider API keys configured.")
        return []

    queries = _build_queries(plan, keywords_override)
    target_count = _target_clip_count(plan=plan, queries=queries, keywords_override=keywords_override)
    cache_dir = output_path.parent / "_stock_cache"
    downloaded: list[Path] = []
    seen_paths: set[Path] = set()

    log(f"Stock keyword count: {len(queries)} | Target downloaded clips: {target_count}")

    if pexels_key and len(downloaded) < target_count:
        downloaded.extend(
            _download_from_pexels(
                api_key=pexels_key,
                queries=queries,
                cache_dir=cache_dir / "pexels",
                width=width,
                height=height,
                target_count=target_count - len(downloaded),
                seen_paths=seen_paths,
                log=log,
            )
        )

    if pixabay_key and len(downloaded) < target_count:
        downloaded.extend(
            _download_from_pixabay(
                api_key=pixabay_key,
                queries=queries,
                cache_dir=cache_dir / "pixabay",
                width=width,
                height=height,
                target_count=target_count - len(downloaded),
                seen_paths=seen_paths,
                log=log,
            )
        )

    return downloaded


def _build_queries(plan: list[PlannedSegment], keywords_override: str) -> list[str]:
    if keywords_override.strip():
        return _dedupe(
            part.strip()
            for part in re.split(r"[,\n]+", keywords_override)
            if part.strip()
        )

    derived = []
    for segment in plan[:8]:
        query = _segment_to_query(segment.text)
        if query:
            derived.append(query)

    return _dedupe(derived + DEFAULT_QUERIES)


def _segment_to_query(text: str) -> str | None:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    filtered = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
    if not filtered:
        return None
    return " ".join(filtered[:3])


def _download_from_pexels(
    api_key: str,
    queries: list[str],
    cache_dir: Path,
    width: int,
    height: int,
    target_count: int,
    seen_paths: set[Path],
    log: callable,
) -> list[Path]:
    results: list[Path] = []
    for query in queries:
        log(f"Searching Pexels for: {query}")
        params = urlencode({"query": query, "per_page": MAX_RESULTS_PER_KEYWORD, "orientation": "landscape"})
        request = Request(
            f"https://api.pexels.com/videos/search?{params}",
            headers={"Authorization": api_key, "User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            log(f"Pexels request failed for '{query}': {exc}")
            continue

        found = payload.get("videos", [])
        log(f"Pexels results for '{query}': {len(found)}")

        for video in found:
            if len(results) >= target_count:
                break
            video_url = _select_pexels_video_url(video, width, height)
            if not video_url:
                continue
            destination = cache_dir / f"pexels_{video['id']}.mp4"
            clip_path = _download_video(video_url, destination, seen_paths, log)
            if clip_path is not None:
                results.append(clip_path)
    return results


def _download_from_pixabay(
    api_key: str,
    queries: list[str],
    cache_dir: Path,
    width: int,
    height: int,
    target_count: int,
    seen_paths: set[Path],
    log: callable,
) -> list[Path]:
    results: list[Path] = []
    for query in queries:
        log(f"Searching Pixabay for: {query}")
        params = urlencode({"key": api_key, "q": query, "per_page": MAX_RESULTS_PER_KEYWORD})
        request = Request(
            f"https://pixabay.com/api/videos/?{params}",
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            log(f"Pixabay request failed for '{query}': {exc}")
            continue

        found = payload.get("hits", [])
        log(f"Pixabay results for '{query}': {len(found)}")

        for video in found:
            if len(results) >= target_count:
                break
            video_url = _select_pixabay_video_url(video, width, height)
            if not video_url:
                continue
            destination = cache_dir / f"pixabay_{video['id']}.mp4"
            clip_path = _download_video(video_url, destination, seen_paths, log)
            if clip_path is not None:
                results.append(clip_path)
    return results


def _select_pexels_video_url(video: dict, width: int, height: int) -> str | None:
    video_files = video.get("video_files", [])
    best_url = None
    best_score = None
    for item in video_files:
        link = item.get("link")
        if not link or item.get("file_type") != "video/mp4":
            continue
        file_width = int(item.get("width") or 0)
        file_height = int(item.get("height") or 0)
        score = _resolution_score(file_width, file_height, width, height)
        if best_score is None or score > best_score:
            best_score = score
            best_url = link
    return best_url


def _select_pixabay_video_url(video: dict, width: int, height: int) -> str | None:
    variants = video.get("videos", {})
    best_url = None
    best_score = None
    for key in ("large", "medium", "small", "tiny"):
        item = variants.get(key)
        if not item:
            continue
        link = item.get("url")
        if not link:
            continue
        file_width = int(item.get("width") or 0)
        file_height = int(item.get("height") or 0)
        score = _resolution_score(file_width, file_height, width, height)
        if best_score is None or score > best_score:
            best_score = score
            best_url = link
    return best_url


def _resolution_score(file_width: int, file_height: int, target_width: int, target_height: int) -> int:
    fit_bonus = 1_000_000 if file_width >= target_width and file_height >= target_height else 0
    return fit_bonus + (file_width * file_height)


def _download_video(url: str, destination: Path, seen_paths: set[Path], log: callable) -> Path | None:
    if destination in seen_paths:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen_paths.add(destination)

    if destination.exists() and destination.stat().st_size > 0:
        log(f"Using cached stock clip: {destination.name}")
        return destination

    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination)
        log(f"Downloaded stock clip: {destination.name}")
        return destination
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        log(f"Stock clip download failed for {destination.name}: {exc}")
        return None


def _read_setting(name: str) -> str:
    direct_value = os.getenv(name, "").strip()
    if direct_value:
        return direct_value

    for candidate in _env_file_candidates():
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ""


def _validated_api_key(name: str, log: callable) -> str:
    value = _read_setting(name)
    lowered = value.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        log(f"Ignoring {name}: value looks like a URL, not an API key.")
        return ""
    return value


def _env_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    for root in search_roots:
        current = root.resolve()
        for directory in (current, *current.parents):
            candidate = directory / ".env"
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    return candidates


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _target_clip_count(plan: list[PlannedSegment], queries: list[str], keywords_override: str) -> int:
    if keywords_override.strip() and queries:
        return min(max(len(plan), MIN_DOWNLOADS_PER_KEYWORD * len(queries)), 30)
    return min(max(3, len(plan)), 8)