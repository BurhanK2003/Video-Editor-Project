from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
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
DEFAULT_QUERIES = ["wildlife", "nature outdoors", "animals"]
MIN_DOWNLOADS_PER_KEYWORD = 1
MAX_RESULTS_PER_KEYWORD = 5
MAX_DOWNLOADS_PER_KEYWORD = 3
USER_AGENT = "local-auto-video-editor/1.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEED_FETCH_PER_PROVIDER = 2
MAX_PROVIDER_QUERY_SLICE = 6


@dataclass(frozen=True)
class StockProviderAdapter:
    name: str
    cache_subdir: str
    api_key_env: str | None
    fetch_fn: Callable[..., list[Path]]


def _build_provider_adapters() -> list[StockProviderAdapter]:
    """Provider adapter scaffold: register providers in one place."""
    return [
        StockProviderAdapter(
            name="Pexels Video",
            cache_subdir="pexels",
            api_key_env="PEXELS_API_KEY",
            fetch_fn=_download_from_pexels,
        ),
        StockProviderAdapter(
            name="Pixabay Video",
            cache_subdir="pixabay",
            api_key_env="PIXABAY_API_KEY",
            fetch_fn=_download_from_pixabay,
        ),
        # New provider: keyless Openverse images for source diversity.
        StockProviderAdapter(
            name="Openverse Images",
            cache_subdir="openverse_images",
            api_key_env=None,
            fetch_fn=_download_images_from_openverse,
        ),
        StockProviderAdapter(
            name="Pexels Images",
            cache_subdir="pexels_images",
            api_key_env="PEXELS_API_KEY",
            fetch_fn=_download_images_from_pexels,
        ),
        StockProviderAdapter(
            name="Pixabay Images",
            cache_subdir="pixabay_images",
            api_key_env="PIXABAY_API_KEY",
            fetch_fn=_download_images_from_pixabay,
        ),
        StockProviderAdapter(
            name="YouTube Creative Commons",
            cache_subdir="youtube_cc",
            api_key_env=None,
            fetch_fn=_download_videos_from_youtube_cc,
        ),
    ]


def fetch_stock_clips(
    plan: list[PlannedSegment],
    output_path: Path,
    width: int,
    height: int,
    keywords_override: str,
    log: callable,
) -> list[Path]:
    provider_adapters = _build_provider_adapters()
    provider_keys: dict[str, str] = {}
    enabled_adapters: list[StockProviderAdapter] = []
    for adapter in provider_adapters:
        if not adapter.api_key_env:
            enabled_adapters.append(adapter)
            continue
        key = _validated_api_key(adapter.api_key_env, log)
        provider_keys[adapter.api_key_env] = key
        if key:
            enabled_adapters.append(adapter)

    if not enabled_adapters:
        log("Stock fetch skipped: no provider API keys configured and no keyless providers enabled.")
        return []

    queries = _build_queries(plan, keywords_override)
    target_count = _target_clip_count(plan=plan, queries=queries, keywords_override=keywords_override)
    # Fetch a small headroom to improve per-scene best-match quality while keeping runtime bounded.
    candidate_target = max(target_count, min(target_count + 4, int(target_count * 1.35)))
    cache_dir = output_path.parent / "_stock_cache"
    downloaded: list[Path] = []
    seen_paths: set[Path] = set()

    if keywords_override.strip():
        log(f"Manual stock hint(s): {keywords_override.strip()}")
    log(
        f"Stock keyword count: {len(queries)} | "
        f"Target clips: {target_count} | candidate pool target: {candidate_target}"
    )
    if queries:
        preview = "; ".join(queries[:4])
        log(f"Stock query preview: {preview}")

    def _run_provider(
        adapter: StockProviderAdapter,
        *,
        provider_queries: list[str],
        provider_target: int,
    ) -> None:
        nonlocal downloaded
        if provider_target <= 0:
            return
        key = provider_keys.get(adapter.api_key_env or "", "") if adapter.api_key_env else ""
        try:
            if adapter.api_key_env:
                downloaded.extend(
                    adapter.fetch_fn(
                        api_key=key,
                        queries=provider_queries,
                        cache_dir=cache_dir / adapter.cache_subdir,
                        width=width,
                        height=height,
                        target_count=provider_target,
                        seen_paths=seen_paths,
                        log=log,
                    )
                )
            else:
                downloaded.extend(
                    adapter.fetch_fn(
                        queries=provider_queries,
                        cache_dir=cache_dir / adapter.cache_subdir,
                        width=width,
                        height=height,
                        target_count=provider_target,
                        seen_paths=seen_paths,
                        log=log,
                    )
                )
        except Exception as exc:
            log(f"Provider '{adapter.name}' failed: {exc}")

    # Phase 1: quick seed from each provider for source diversity with low latency.
    seed_queries = queries[: max(2, min(MAX_PROVIDER_QUERY_SLICE, len(queries)))]
    for adapter in enabled_adapters:
        if len(downloaded) >= candidate_target:
            break
        _run_provider(
            adapter,
            provider_queries=seed_queries,
            provider_target=min(SEED_FETCH_PER_PROVIDER, max(0, candidate_target - len(downloaded))),
        )

    # Phase 2: expand by score potential until candidate pool target is met.
    if len(downloaded) < candidate_target:
        for adapter in enabled_adapters:
            if len(downloaded) >= candidate_target:
                break
            _run_provider(
                adapter,
                provider_queries=queries[: max(2, min(len(queries), MAX_PROVIDER_QUERY_SLICE + 4))],
                provider_target=max(0, candidate_target - len(downloaded)),
            )

    if enabled_adapters:
        provider_names = ", ".join(adapter.name for adapter in enabled_adapters)
        log(f"Enabled stock providers: {provider_names}")

    # Keep pool bounded for downstream matcher runtime.
    return downloaded[:candidate_target]


def _download_videos_from_youtube_cc(
    queries: list[str],
    cache_dir: Path,
    width: int,
    height: int,
    target_count: int,
    seen_paths: set[Path],
    log: callable,
) -> list[Path]:
    del width, height
    try:
        import yt_dlp
    except Exception:
        log("YouTube CC provider skipped: install yt-dlp to enable Creative Commons downloads.")
        return []

    results: list[Path] = []
    for query in queries[:6]:
        if len(results) >= target_count:
            break

        search_term = f"ytsearch5:{query} b roll creative commons"
        probe_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            "extract_flat": False,
            "ignoreerrors": True,
            "socket_timeout": 15,
        }
        try:
            with yt_dlp.YoutubeDL(probe_opts) as ydl:
                info = ydl.extract_info(search_term, download=False) or {}
        except Exception as exc:
            log(f"YouTube CC search failed for '{query}': {exc}")
            continue

        entries = info.get("entries") or []
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if len(results) >= target_count:
                break
            if not isinstance(entry, dict):
                continue

            video_id = str(entry.get("id") or "").strip()
            webpage_url = str(entry.get("webpage_url") or entry.get("url") or "").strip()
            license_text = str(entry.get("license") or "").strip().lower()
            title = str(entry.get("title") or "").strip()

            if not video_id or not webpage_url:
                continue

            # Keep this provider restricted to explicit CC metadata when available.
            if license_text and "creative commons" not in license_text and "cc" not in license_text:
                continue

            destination = cache_dir / f"ytcc_{video_id}.mp4"
            if destination in seen_paths:
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size > 0:
                seen_paths.add(destination)
                _write_metadata_sidecar(
                    destination,
                    {
                        "provider": "youtube-cc",
                        "query": query,
                        "title": title,
                        "tags": "youtube,creative-commons",
                    },
                )
                log(f"Using cached stock clip: {destination.name}")
                results.append(destination)
                continue

            outtmpl = str(destination.with_suffix("")) + ".%(ext)s"
            dl_opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best[ext=mp4]/best",
                "outtmpl": outtmpl,
                "merge_output_format": "mp4",
                "ignoreerrors": True,
                "socket_timeout": 20,
            }
            try:
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.download([webpage_url])
            except Exception as exc:
                log(f"YouTube CC download failed for '{title or video_id}': {exc}")
                continue

            produced = destination if destination.exists() else None
            if produced is None:
                candidates = sorted(destination.parent.glob(f"ytcc_{video_id}.*"))
                for candidate in candidates:
                    if candidate.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"} and candidate.stat().st_size > 0:
                        produced = candidate
                        break

            if produced is None:
                continue

            seen_paths.add(produced)
            _write_metadata_sidecar(
                produced,
                {
                    "provider": "youtube-cc",
                    "query": query,
                    "title": title,
                    "tags": "youtube,creative-commons",
                },
            )
            log(f"Downloaded stock clip: {produced.name}")
            results.append(produced)
            break

    return results


def _build_queries(plan: list[PlannedSegment], keywords_override: str) -> list[str]:
    manual_terms = [
        part.strip()
        for part in re.split(r"[,\n]+", keywords_override)
        if part.strip()
    ]

    derived = []
    for segment in plan[:20]:
        if segment.visual_query.strip():
            # Use the planner's cinematic query directly — already well-formed.
            derived.append(segment.visual_query.strip())

    base_queries = _dedupe(derived + DEFAULT_QUERIES)

    if not manual_terms:
        return base_queries

    manual_phrase = " ".join(manual_terms)
    blended: list[str] = []
    for query in base_queries[:14]:
        blended.append(f"{manual_phrase} {query}".strip())

    # Keep some pure AI queries too so user hints guide search without collapsing diversity.
    blended.extend(base_queries[:8])
    blended.append(f"cinematic {manual_phrase} wildlife nature 4k".strip())
    blended.append(manual_phrase)
    return _dedupe(blended)


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
        if len(results) >= target_count:
            log(f"Pexels download target reached ({target_count}). Stopping further keyword searches.")
            break
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

        downloaded_for_query = 0
        for video in found:
            if len(results) >= target_count:
                break
            if downloaded_for_query >= MAX_DOWNLOADS_PER_KEYWORD:
                break
            video_url = _select_pexels_video_url(video, width, height)
            if not video_url:
                continue
            destination = cache_dir / f"pexels_{video['id']}.mp4"
            metadata = {
                "provider": "pexels",
                "query": query,
                "title": str(video.get("url") or ""),
                "tags": "",
            }
            clip_path = _download_video(video_url, destination, seen_paths, log, metadata)
            if clip_path is not None:
                results.append(clip_path)
                downloaded_for_query += 1
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
        if len(results) >= target_count:
            log(f"Pixabay download target reached ({target_count}). Stopping further keyword searches.")
            break
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

        downloaded_for_query = 0
        for video in found:
            if len(results) >= target_count:
                break
            if downloaded_for_query >= MAX_DOWNLOADS_PER_KEYWORD:
                break
            video_url = _select_pixabay_video_url(video, width, height)
            if not video_url:
                continue
            destination = cache_dir / f"pixabay_{video['id']}.mp4"
            metadata = {
                "provider": "pixabay",
                "query": query,
                "title": str(video.get("pageURL") or ""),
                "tags": str(video.get("tags") or ""),
            }
            clip_path = _download_video(video_url, destination, seen_paths, log, metadata)
            if clip_path is not None:
                results.append(clip_path)
                downloaded_for_query += 1
    return results


def _download_images_from_pexels(
    api_key: str,
    queries: list[str],
    cache_dir: Path,
    width: int,
    height: int,
    target_count: int,
    seen_paths: set[Path],
    log: callable,
) -> list[Path]:
    del width, height
    results: list[Path] = []
    for query in queries:
        if len(results) >= target_count:
            break
        params = urlencode({"query": query, "per_page": 3, "orientation": "portrait"})
        request = Request(
            f"https://api.pexels.com/v1/search?{params}",
            headers={"Authorization": api_key, "User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue

        for photo in payload.get("photos", []):
            if len(results) >= target_count:
                break
            image_url = str(((photo.get("src") or {}).get("large2x") or (photo.get("src") or {}).get("portrait") or "")).strip()
            if not image_url:
                continue
            destination = cache_dir / f"pexels_photo_{photo['id']}.jpg"
            metadata = {
                "provider": "pexels-image",
                "query": query,
                "title": str(photo.get("url") or ""),
                "tags": str(photo.get("alt") or ""),
            }
            clip_path = _download_binary(image_url, destination, seen_paths, log, metadata, asset_label="stock image")
            if clip_path is not None:
                results.append(clip_path)
                break
    return results


def _download_images_from_pixabay(
    api_key: str,
    queries: list[str],
    cache_dir: Path,
    width: int,
    height: int,
    target_count: int,
    seen_paths: set[Path],
    log: callable,
) -> list[Path]:
    del width, height
    results: list[Path] = []
    for query in queries:
        if len(results) >= target_count:
            break
        params = urlencode({"key": api_key, "q": query, "per_page": 3, "image_type": "photo", "orientation": "vertical"})
        request = Request(
            f"https://pixabay.com/api/?{params}",
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue

        for photo in payload.get("hits", []):
            if len(results) >= target_count:
                break
            image_url = str(photo.get("largeImageURL") or photo.get("webformatURL") or "").strip()
            if not image_url:
                continue
            destination = cache_dir / f"pixabay_photo_{photo['id']}.jpg"
            metadata = {
                "provider": "pixabay-image",
                "query": query,
                "title": str(photo.get("pageURL") or ""),
                "tags": str(photo.get("tags") or ""),
            }
            clip_path = _download_binary(image_url, destination, seen_paths, log, metadata, asset_label="stock image")
            if clip_path is not None:
                results.append(clip_path)
                break
    return results


def _download_images_from_openverse(
    queries: list[str],
    cache_dir: Path,
    width: int,
    height: int,
    target_count: int,
    seen_paths: set[Path],
    log: callable,
) -> list[Path]:
    del width, height
    results: list[Path] = []
    for query in queries:
        if len(results) >= target_count:
            break

        params = urlencode(
            {
                "q": query,
                "page_size": 4,
                "license_type": "commercial",
                "extension": "jpg,png,webp",
            }
        )
        request = Request(
            f"https://api.openverse.org/v1/images/?{params}",
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            log(f"Openverse request failed for '{query}': {exc}")
            continue

        found = payload.get("results", [])
        if isinstance(found, list):
            log(f"Openverse image results for '{query}': {len(found)}")
        else:
            continue

        for photo in found:
            if len(results) >= target_count:
                break
            if not isinstance(photo, dict):
                continue

            image_url = str(
                photo.get("url")
                or photo.get("thumbnail")
                or photo.get("source")
                or ""
            ).strip()
            if not image_url:
                continue

            photo_id = str(photo.get("id") or "").strip() or hashlib.md5(image_url.encode("utf-8")).hexdigest()[:12]
            destination = cache_dir / f"openverse_{photo_id}.jpg"
            metadata = {
                "provider": "openverse-image",
                "query": query,
                "title": str(photo.get("title") or photo.get("source") or ""),
                "tags": ",".join(str(t) for t in (photo.get("tags") or [])[:8]),
            }
            clip_path = _download_binary(image_url, destination, seen_paths, log, metadata, asset_label="stock image")
            if clip_path is not None:
                results.append(clip_path)
                break

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


def _download_video(
    url: str,
    destination: Path,
    seen_paths: set[Path],
    log: callable,
    metadata: dict,
) -> Path | None:
    return _download_binary(url, destination, seen_paths, log, metadata, asset_label="stock clip")


def _download_binary(
    url: str,
    destination: Path,
    seen_paths: set[Path],
    log: callable,
    metadata: dict,
    asset_label: str,
) -> Path | None:
    if destination in seen_paths:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen_paths.add(destination)

    if destination.exists() and destination.stat().st_size > 0:
        _write_metadata_sidecar(destination, metadata)
        log(f"Using cached {asset_label}: {destination.name}")
        return destination

    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination)
        _write_metadata_sidecar(destination, metadata)
        log(f"Downloaded {asset_label}: {destination.name}")
        return destination
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        log(f"{asset_label.capitalize()} download failed for {destination.name}: {exc}")
        return None


def _write_metadata_sidecar(destination: Path, metadata: dict) -> None:
    sidecar = destination.with_suffix(destination.suffix + ".json")
    payload = {
        "provider": str(metadata.get("provider") or ""),
        "query": str(metadata.get("query") or ""),
        "title": str(metadata.get("title") or ""),
        "tags": str(metadata.get("tags") or ""),
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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
    if queries:
        # Keep download volume tied to scene count, not just manual keyword count.
        keyword_scaled_target = max(len(plan), min(8, len(plan) + 2))
        hard_cap = max(8, MAX_DOWNLOADS_PER_KEYWORD * len(queries))
        return min(max(len(plan), keyword_scaled_target), hard_cap)
    return min(max(3, len(plan)), 8)