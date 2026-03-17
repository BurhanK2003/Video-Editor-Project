from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from .matcher import suggest_scene_keywords
from .models import PlannedSegment, TranscriptSegment, WordToken


# ---------------------------------------------------------------------------
# Free AI query generation (Ollama → Groq → keyword fallback)
# ---------------------------------------------------------------------------

_QUERY_CACHE: dict[str, str] = {}
_QUERY_CACHE_FILE = Path(".llm_query_cache.json")
_CACHE_LOADED = False
_SCENE_PLAN_CACHE: dict[str, dict] = {}
_SCENE_PLAN_CACHE_FILE = Path(".llm_scene_plan_cache.json")
_SCENE_PLAN_CACHE_LOADED = False

VALID_EMOTIONS = {"curiosity", "suspense", "excitement"}
VALID_PACING = {"slow", "medium", "fast"}
VALID_TRANSITIONS = {"jump_cut", "zoom_in", "whip", "fade"}


def _load_query_cache() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    try:
        if _QUERY_CACHE_FILE.exists():
            _QUERY_CACHE.update(json.loads(_QUERY_CACHE_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass


def _save_query_cache() -> None:
    try:
        _QUERY_CACHE_FILE.write_text(
            json.dumps(_QUERY_CACHE, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _load_scene_plan_cache() -> None:
    global _SCENE_PLAN_CACHE_LOADED
    if _SCENE_PLAN_CACHE_LOADED:
        return
    _SCENE_PLAN_CACHE_LOADED = True
    try:
        if _SCENE_PLAN_CACHE_FILE.exists():
            payload = json.loads(_SCENE_PLAN_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                _SCENE_PLAN_CACHE.update(payload)
    except Exception:
        pass


def _save_scene_plan_cache() -> None:
    try:
        _SCENE_PLAN_CACHE_FILE.write_text(
            json.dumps(_SCENE_PLAN_CACHE, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _call_ollama(prompt: str) -> str | None:
    """Call local Ollama (http://localhost:11434). Free, no API key, runs offline."""
    payload = json.dumps({
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 32},
    }).encode()
    try:
        req = Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode())
            return (result.get("response") or "").strip()
    except Exception:
        return None


def _call_ollama_json(prompt: str, *, timeout: int = 12, num_predict: int = 320) -> dict | None:
    """Call Ollama in strict JSON mode and parse a JSON object response."""
    payload = json.dumps(
        {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.25, "num_predict": int(max(64, num_predict))},
        }
    ).encode()
    try:
        req = Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=max(4, int(timeout))) as resp:
            result = json.loads(resp.read().decode())
            response_text = str(result.get("response") or "").strip()
    except Exception:
        return None

    if not response_text:
        return None

    try:
        parsed = json.loads(response_text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # Defensive extraction if model leaks wrapper text around JSON.
    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _call_groq(prompt: str) -> str | None:
    """Call Groq free-tier API (requires GROQ_API_KEY env var, llama-3.3-70b-versatile)."""
    import os
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.4,
    }).encode()
    try:
        req = Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urlopen(req, timeout=12) as resp:
            result = json.loads(resp.read().decode())
            return (result["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def _llm_cinematic_query(text: str, emotion: str) -> str | None:
    """
    Use a free LLM (Ollama locally, Groq cloud) to write a specific cinematic
    stock footage search query for the given narration line.

    Results are cached by (text, emotion) so repeated runs are instant.
    Falls back to None if no LLM is reachable (caller uses keyword fallback).
    """
    _load_query_cache()
    cache_key = f"{text.lower().strip()}|{emotion}"
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    prompt = (
        f'You are helping find stock footage for a short video.\n'
        f'The narrator says: "{text}"\n'
        f'Scene emotion: {emotion}\n'
        f"Write ONE cinematic stock footage search query, max 8 words, "
        f"no quotes. Focus on VISUALS not words. Be specific and concrete. "
        f"Reply with ONLY the query, nothing else."
    )

    result = _call_ollama(prompt) or _call_groq(prompt)
    if result:
        # Sanitise: strip quotes, truncate to 10 words max.
        cleaned = re.sub(r'["\'\n]', "", result).strip()
        cleaned = " ".join(cleaned.split()[:10])
        if len(cleaned) > 4:
            _QUERY_CACHE[cache_key] = cleaned
            _save_query_cache()
            return cleaned
    return None


def _normalized_emotion(value: str | None, fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in VALID_EMOTIONS else fallback


def _normalized_pacing(value: str | None, fallback: str = "fast") -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in VALID_PACING else fallback


def _normalized_transition(value: str | None, fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in VALID_TRANSITIONS else fallback


def _normalized_clip_length(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return fallback
    return max(1.0, min(3.5, parsed))


def _normalized_emphasis_words(raw: object, fallback_text: str) -> list[str]:
    words: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            token = re.sub(r"[^A-Za-z']+", "", str(item or "")).lower().strip()
            if token:
                words.append(token)
    if not words:
        words = _important_words(fallback_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        deduped.append(word)
    return deduped[:4]


def _normalized_highlight_phrase(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    return " ".join(text.split()[:6])


def _normalized_visual_query(raw: object, fallback: str) -> str:
    query = re.sub(r"[\r\n\t]+", " ", str(raw or "")).strip()
    if not query:
        return fallback
    query = re.sub(r"[\"']", "", query)
    query = " ".join(query.split()[:12]).strip()
    return query if len(query) >= 5 else fallback


def _ollama_structured_scene_plan(
    text: str,
    *,
    fallback_query: str,
    fallback_emotion: str,
    fallback_transition: str,
    fallback_clip_length: float,
    fallback_pacing: str = "fast",
) -> dict | None:
    """Return a validated scene profile from Ollama strict JSON mode."""
    _load_scene_plan_cache()
    cache_key = (
        f"scene:v1|{text.lower().strip()}|{fallback_emotion}|{fallback_transition}|"
        f"{fallback_pacing}|{fallback_clip_length:.2f}"
    )
    cached = _SCENE_PLAN_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return cached

    prompt = (
        "You are a short-form video scene planner. Return JSON only.\n"
        "Given one narration chunk, produce a concrete stock-footage scene profile.\n"
        "Allowed values:\n"
        "- emotion: curiosity | suspense | excitement\n"
        "- pacing: slow | medium | fast\n"
        "- transition_type: jump_cut | zoom_in | whip | fade\n"
        "- clip_length_seconds: number between 1.0 and 3.5\n"
        "JSON schema:\n"
        '{"visual_query":"string max 12 words",'
        '"emotion":"string",'
        '"pacing":"string",'
        '"transition_type":"string",'
        '"clip_length_seconds":1.6,'
        '"emphasis_words":["word1","word2"],'
        '"highlight_phrase":"short optional phrase"}\n'
        f"Narration chunk: {text}\n"
        f"Current best visual query: {fallback_query}\n"
        f"Current emotion: {fallback_emotion}\n"
        "Output only one JSON object."
    )
    raw = _call_ollama_json(prompt, timeout=12, num_predict=300)
    if not isinstance(raw, dict):
        return None

    profile = {
        "visual_query": _normalized_visual_query(raw.get("visual_query"), fallback_query),
        "emotion": _normalized_emotion(raw.get("emotion"), fallback_emotion),
        "pacing": _normalized_pacing(raw.get("pacing"), fallback_pacing),
        "transition_type": _normalized_transition(raw.get("transition_type"), fallback_transition),
        "clip_length_seconds": _normalized_clip_length(raw.get("clip_length_seconds"), fallback_clip_length),
        "emphasis_words": _normalized_emphasis_words(raw.get("emphasis_words"), text),
        "highlight_phrase": _normalized_highlight_phrase(raw.get("highlight_phrase")),
    }
    _SCENE_PLAN_CACHE[cache_key] = profile
    _save_scene_plan_cache()
    return profile


def _ollama_critic_pass(plan_payload: list[dict]) -> dict[int, dict]:
    """Critic pass that revises repetitive/weak scene specs via strict JSON edits."""
    if not plan_payload:
        return {}

    compact = []
    for idx, seg in enumerate(plan_payload[:40]):
        compact.append(
            {
                "index": idx,
                "text": str(seg.get("text") or "")[:120],
                "visual_query": str(seg.get("visual_query") or ""),
                "emotion": str(seg.get("emotion") or "curiosity"),
                "transition_type": str(seg.get("transition_type") or "jump_cut"),
                "clip_length_seconds": float(seg.get("clip_length_seconds") or 1.6),
            }
        )

    prompt = (
        "You are a scene-plan critic for short-form videos. Return JSON only.\n"
        "Task: find repetitive, vague, or low-energy visual plans and propose minimal edits.\n"
        "Rules:\n"
        "- Keep the original meaning of narration.\n"
        "- Edit only scenes that need improvement.\n"
        "- Use allowed values: emotion {curiosity,suspense,excitement}, transition_type {jump_cut,zoom_in,whip,fade}.\n"
        "- clip_length_seconds must be between 1.0 and 3.5.\n"
        "Output schema:\n"
        '{"edits":[{"index":3,"visual_query":"...","emotion":"curiosity",'
        '"transition_type":"whip","clip_length_seconds":1.4,"reason":"..."}]}\n'
        f"Current scene plan JSON:\n{json.dumps(compact, ensure_ascii=True)}"
    )
    raw = _call_ollama_json(prompt, timeout=14, num_predict=420)
    if not isinstance(raw, dict):
        return {}

    edits: dict[int, dict] = {}
    raw_edits = raw.get("edits")
    if not isinstance(raw_edits, list):
        return edits

    for item in raw_edits:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        if idx < 0 or idx >= len(plan_payload):
            continue

        original = plan_payload[idx]
        edits[idx] = {
            "visual_query": _normalized_visual_query(item.get("visual_query"), str(original.get("visual_query") or "")),
            "emotion": _normalized_emotion(item.get("emotion"), str(original.get("emotion") or "curiosity")),
            "transition_type": _normalized_transition(item.get("transition_type"), str(original.get("transition_type") or "jump_cut")),
            "clip_length_seconds": _normalized_clip_length(item.get("clip_length_seconds"), float(original.get("clip_length_seconds") or 1.6)),
            "reason": str(item.get("reason") or "").strip(),
        }
    return edits


BRAND_PHRASE = "Nature Lives In You"
IMPORTANT_WORDS = {
    "one",
    "only",
    "secret",
    "shocking",
    "surprising",
    "must",
    "never",
    "always",
    "important",
    "nature",
    "lives",
    "you",
}
EMOTION_WORDS = {
    "curiosity": {"why", "how", "discover", "hidden", "secret", "unknown"},
    "suspense": {"wait", "before", "until", "risk", "danger", "warning"},
    "excitement": {"amazing", "incredible", "wow", "powerful", "boost", "fast"},
}
TRANSITIONS = ("jump_cut", "zoom_in", "whip", "fade")
NATURE_THEME_TERMS = ("nature", "wildlife", "documentary", "outdoors")
ABSTRACT_VISUAL_STOPWORDS = {
    "forget",
    "listen",
    "feel",
    "think",
    "know",
    "understand",
    "remember",
    "believe",
    "watch",
    "hear",
}
VISUAL_HINTS_BY_TOKEN = {
    "forget": ["contemplative", "reflection", "close-up"],
    "listen": ["listening", "headphones", "ambient"],
    "feel": ["emotional", "slow motion"],
    "think": ["contemplative", "portrait"],
    "remember": ["nostalgic", "soft light"],
    "breathe": ["deep breath", "calm", "nature walk"],
    "calm": ["serene", "gentle motion"],
    "focus": ["close-up", "eyes", "detail"],
}
NOISE_KEYWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "there",
    "theres",
    "it's",
    "its",
    "you",
    "your",
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
    "now",
    "today",
    "really",
    "just",
    "im",
    "ive",
    "id",
}

SCENE_INTENTS: list[tuple[re.Pattern[str], dict[str, str | float]]] = [
    (
        re.compile(r"\brain\b|\braindrops?\b|\bcalm\b", re.IGNORECASE),
        {
            "query": "cinematic close-up raindrops leaves window slow motion nature 4k",
            "emotion": "curiosity",
            "transition_type": "fade",
            "clip_length_seconds": 1.8,
        },
    ),
    (
        re.compile(r"\bwalk\b|\bforest\b|\bhome\b", re.IGNORECASE),
        {
            "query": "cinematic forest trail walking sunlight leaves pov nature 4k",
            "emotion": "curiosity",
            "transition_type": "jump_cut",
            "clip_length_seconds": 1.6,
        },
    ),
    (
        re.compile(r"\bwithin\b|\binside\b|\breflection\b|\breflections\b", re.IGNORECASE),
        {
            "query": "cinematic reflection water mirror human silhouette petals nature 4k",
            "emotion": "suspense",
            "transition_type": "fade",
            "clip_length_seconds": 1.9,
        },
    ),
    (
        re.compile(r"\bbirds?\b|\bdeer\b|\binsects?\b|\bwildlife\b|\banimals?\b", re.IGNORECASE),
        {
            "query": "cinematic wildlife birds deer insects macro natural habitat 4k",
            "emotion": "excitement",
            "transition_type": "jump_cut",
            "clip_length_seconds": 1.3,
        },
    ),
    (
        re.compile(r"\baerial\b|\bmountains?\b|\brivers?\b|\bocean\b|\bclouds?\b", re.IGNORECASE),
        {
            "query": "cinematic aerial forest mountains river drone landscape nature 4k",
            "emotion": "curiosity",
            "transition_type": "fade",
            "clip_length_seconds": 1.8,
        },
    ),
    (
        re.compile(r"nature lives in you", re.IGNORECASE),
        {
            "query": "cinematic logo text glow forest mist sunlight nature lives in you 4k",
            "emotion": "curiosity",
            "transition_type": "fade",
            "clip_length_seconds": 2.0,
        },
    ),
    (
        re.compile(r"\bsubscribe\b|\bfollow\b|\bcomment\b|\btell us\b", re.IGNORECASE),
        {
            "query": "cinematic hands touching water sunlight leaves call to action nature 4k",
            "emotion": "excitement",
            "transition_type": "zoom_in",
            "clip_length_seconds": 1.4,
        },
    ),
]


def _normalize_brand_phrase(text: str) -> str:
    return re.sub(
        r"nature\s+lives\s+in\s+you",
        BRAND_PHRASE,
        text,
        flags=re.IGNORECASE,
    )


def _clean_word(word: str) -> str:
    return re.sub(r"[^A-Za-z']+", "", word).strip()


def _important_words(text: str) -> list[str]:
    words = [_clean_word(w).lower() for w in text.split()]
    picked = [w for w in words if w and (w in IMPORTANT_WORDS or len(w) >= 9)]
    unique: list[str] = []
    seen: set[str] = set()
    for w in picked:
        if w in seen:
            continue
        seen.add(w)
        unique.append(w)
    return unique[:2]


def _apply_caps_style(text: str, emphasis_words: list[str]) -> str:
    if not text.strip():
        return text
    styled = text

    if BRAND_PHRASE.lower() in styled.lower():
        styled = re.sub(
            r"nature\s+lives\s+in\s+you",
            "NATURE LIVES IN YOU",
            styled,
            flags=re.IGNORECASE,
        )

    for word in emphasis_words:
        if not word:
            continue
        styled = re.sub(
            rf"\b{re.escape(word)}\b",
            word.upper(),
            styled,
            flags=re.IGNORECASE,
        )
    return styled


def _chunk_size(words_per_second: float) -> int:
    # Keep captions readable and semantically useful; avoid hyper-fragmented 2-word subtitles.
    if words_per_second >= 3.6:
        return 4
    if words_per_second >= 2.8:
        return 5
    if words_per_second >= 2.0:
        return 6
    return 7


def _chunk_words(words: list[WordToken], target_size: int) -> list[list[WordToken]]:
    chunks: list[list[WordToken]] = []
    current: list[WordToken] = []
    for word in words:
        current.append(word)
        token = (word.text or "").strip()
        hit_punctuation = token.endswith((".", "!", "?", ",", ";", ":"))
        if len(current) >= max(3, target_size) or (hit_punctuation and len(current) >= 3):
            chunks.append(current)
            current = []

    if current:
        if chunks and len(current) <= 2:
            chunks[-1].extend(current)
        else:
            chunks.append(current)

    normalized: list[list[WordToken]] = []
    for chunk in chunks:
        if len(chunk) > 8:
            normalized.append(chunk[:8])
            remainder = chunk[8:]
            if remainder:
                normalized.append(remainder)
        else:
            normalized.append(chunk)
    chunks = normalized

    if len(chunks) >= 2 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def _clip_length_seconds(words_per_second: float) -> float:
    if words_per_second >= 3.2:
        return 1.2
    if words_per_second >= 2.4:
        return 1.6
    if words_per_second >= 1.6:
        return 2.1
    return 2.7


def _emotion_for_text(text: str) -> str:
    lower = text.lower()
    for emotion, keys in EMOTION_WORDS.items():
        if any(k in lower for k in keys):
            return emotion
    return "curiosity"


def _transition_for_segment(
    text: str,
    emotion: str,
    words_per_second: float,
    prev_transitions: list[str],
) -> str:
    lower = text.lower()
    if any(k in lower for k in ("feel", "reflect", "remember", "calm", "breathe")):
        candidate = "fade"
    elif emotion == "excitement" or words_per_second >= 2.9:
        candidate = "whip"
    elif "!" in text or any(k in lower for k in ("one", "secret", "must", "never")):
        candidate = "zoom_in"
    else:
        candidate = "jump_cut"

    if len(prev_transitions) >= 2 and prev_transitions[-1] == prev_transitions[-2] == candidate:
        for fallback in TRANSITIONS:
            if fallback != candidate:
                return fallback
    return candidate


def _cinematic_query(text: str, emotion: str) -> str:
    # Try free LLM first — much more creative and specific than keyword extraction.
    llm_result = _llm_cinematic_query(text, emotion)
    if llm_result:
        return llm_result

    # Keyword fallback (no LLM available).
    raw_keywords = suggest_scene_keywords(text, max_keywords=6)
    filtered: list[str] = []
    visual_hints: list[str] = []
    for kw in raw_keywords:
        token = re.sub(r"[^a-zA-Z]+", "", kw).lower()
        if len(token) < 3 or token in NOISE_KEYWORDS:
            continue
        if token in ABSTRACT_VISUAL_STOPWORDS:
            visual_hints.extend(VISUAL_HINTS_BY_TOKEN.get(token, []))
            continue
        visual_hints.extend(VISUAL_HINTS_BY_TOKEN.get(token, []))
        filtered.append(token)

    lower_text = text.lower()
    if "listen" in lower_text and "nature" in lower_text:
        visual_hints.extend(["person", "listening", "forest trail"])
    if "forget" in lower_text:
        visual_hints.extend(["contemplative", "portrait", "soft light"])

    # Put content-specific words first; keep nature anchors as fallback context.
    deduped: list[str] = []
    seen: set[str] = set()
    for token in [*visual_hints, *filtered, *NATURE_THEME_TERMS]:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)

    base = " ".join(deduped[:7]) if deduped else "nature wildlife documentary"

    if BRAND_PHRASE.lower() in text.lower():
        return "cinematic forest mist sunlight nature life close-up 4k"

    if emotion == "suspense":
        return f"cinematic dramatic {base} moody lighting 4k"
    if emotion == "excitement":
        return f"cinematic action {base} dynamic movement 4k"
    return f"cinematic {base} natural light movement 4k"


def _scene_intent_profile(text: str) -> dict[str, str | float]:
    for pattern, profile in SCENE_INTENTS:
        if pattern.search(text):
            return dict(profile)
    return {}


def _words_from_segment(seg: TranscriptSegment) -> list[WordToken]:
    if seg.words:
        return [w for w in seg.words if (w.text or "").strip()]

    raw = [w for w in (seg.text or "").split() if w.strip()]
    if not raw:
        return []

    total = max(0.2, float(seg.end - seg.start))
    step = total / len(raw)
    tokens: list[WordToken] = []
    cursor = float(seg.start)
    for i, word in enumerate(raw):
        end = float(seg.end) if i == len(raw) - 1 else min(float(seg.end), cursor + step)
        tokens.append(WordToken(start=cursor, end=end, text=word))
        cursor = end
    return tokens


def build_plan(segments: list[TranscriptSegment], log: callable | None = None) -> list[PlannedSegment]:
    plan: list[PlannedSegment] = []
    recent_transitions: list[str] = []
    attention_toggle = False

    scene_payload: list[dict] = []

    for seg in segments:
        seg_text = _normalize_brand_phrase((seg.text or "").strip())
        words = _words_from_segment(seg)
        if not words:
            continue

        words_per_second = len(words) / max(0.2, float(seg.end - seg.start))
        chunk_len = _chunk_size(words_per_second)
        clip_len = _clip_length_seconds(words_per_second)

        chunks = _chunk_words(words, chunk_len)
        consumed_words = 0
        for chunk in chunks:
            if not chunk:
                continue
            i = consumed_words
            consumed_words += len(chunk)

            chunk_text = _normalize_brand_phrase(" ".join(w.text for w in chunk).strip())
            emphasis_words = _important_words(chunk_text)
            intent = _scene_intent_profile(chunk_text)
            emotion = str(intent.get("emotion") or _emotion_for_text(chunk_text))
            transition_type = _transition_for_segment(
                text=chunk_text,
                emotion=emotion,
                words_per_second=words_per_second,
                prev_transitions=recent_transitions,
            )
            if intent.get("transition_type"):
                transition_type = str(intent["transition_type"])

            segment_clip_len = float(intent.get("clip_length_seconds") or clip_len)
            base_query = str(intent.get("query") or _cinematic_query(chunk_text, emotion))

            structured = _ollama_structured_scene_plan(
                chunk_text,
                fallback_query=base_query,
                fallback_emotion=emotion,
                fallback_transition=transition_type,
                fallback_clip_length=segment_clip_len,
                fallback_pacing="fast",
            )
            if structured:
                emotion = str(structured.get("emotion") or emotion)
                transition_type = str(structured.get("transition_type") or transition_type)
                segment_clip_len = float(structured.get("clip_length_seconds") or segment_clip_len)
                emphasis_words = _normalized_emphasis_words(structured.get("emphasis_words"), chunk_text)
                highlight_phrase = _normalized_highlight_phrase(structured.get("highlight_phrase"))
                visual_query = _normalized_visual_query(structured.get("visual_query"), base_query)
                pacing = _normalized_pacing(str(structured.get("pacing") or ""), "fast")
            else:
                visual_query = base_query
                highlight_phrase = BRAND_PHRASE if BRAND_PHRASE.lower() in chunk_text.lower() else ""
                pacing = "fast"

            recent_transitions.append(transition_type)

            transition_after = segment_clip_len >= 1.1 and consumed_words < len(words)
            if transition_type == "fade":
                transition_seconds = 0.22
            elif transition_type == "zoom_in":
                transition_seconds = 0.18
            elif transition_type == "whip":
                transition_seconds = 0.12
            else:
                transition_seconds = 0.10

            start = max(float(seg.start), float(chunk[0].start))
            end = min(float(seg.end), float(chunk[-1].end))
            if end <= start:
                continue

            styled_text = _apply_caps_style(chunk_text, emphasis_words)
            attention_toggle = not attention_toggle
            alternating_emphasis = attention_toggle and len(chunk_text.split()) <= 4
            plan.append(
                PlannedSegment(
                    start=start,
                    end=end,
                    text=styled_text,
                    duration=max(0.45, end - start),
                    transition_after=transition_after,
                    transition_seconds=transition_seconds,
                    emphasis=bool(emphasis_words)
                    or BRAND_PHRASE.lower() in styled_text.lower()
                    or alternating_emphasis,
                    highlight_phrase=highlight_phrase,
                    emphasis_words=emphasis_words,
                    word_tokens=[
                        WordToken(start=float(w.start), end=float(w.end), text=w.text)
                        for w in chunk
                    ],
                    visual_query=visual_query,
                    emotion=emotion,
                    pacing=pacing,
                    transition_type=transition_type,
                    clip_length_seconds=segment_clip_len,
                )
            )

            scene_payload.append(
                {
                    "text": styled_text,
                    "visual_query": visual_query,
                    "emotion": emotion,
                    "transition_type": transition_type,
                    "clip_length_seconds": segment_clip_len,
                }
            )

    critic_edits = _ollama_critic_pass(scene_payload)
    if critic_edits:
        for idx, edit in critic_edits.items():
            if idx < 0 or idx >= len(plan):
                continue
            original = plan[idx]
            plan[idx] = PlannedSegment(
                start=original.start,
                end=original.end,
                text=original.text,
                duration=original.duration,
                transition_after=original.transition_after,
                transition_seconds=original.transition_seconds,
                emphasis=original.emphasis,
                highlight_phrase=original.highlight_phrase,
                emphasis_words=original.emphasis_words,
                word_tokens=original.word_tokens,
                visual_query=_normalized_visual_query(edit.get("visual_query"), original.visual_query),
                emotion=_normalized_emotion(str(edit.get("emotion") or ""), original.emotion),
                pacing=original.pacing,
                transition_type=_normalized_transition(str(edit.get("transition_type") or ""), original.transition_type),
                clip_length_seconds=_normalized_clip_length(edit.get("clip_length_seconds"), original.clip_length_seconds),
            )
            if log and str(edit.get("reason") or "").strip():
                log(f"Scene critic adjusted segment {idx + 1}: {str(edit.get('reason')).strip()}")

        if log:
            log(f"Scene critic pass applied: {len(critic_edits)} segment edits")

    return plan
