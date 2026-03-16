from __future__ import annotations

import re

from .matcher import suggest_scene_keywords
from .models import PlannedSegment, TranscriptSegment, WordToken


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


def build_plan(segments: list[TranscriptSegment]) -> list[PlannedSegment]:
    plan: list[PlannedSegment] = []
    recent_transitions: list[str] = []
    attention_toggle = False

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
            recent_transitions.append(transition_type)

            segment_clip_len = float(intent.get("clip_length_seconds") or clip_len)
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
                    highlight_phrase=BRAND_PHRASE if BRAND_PHRASE.lower() in styled_text.lower() else "",
                    emphasis_words=emphasis_words,
                    word_tokens=[
                        WordToken(start=float(w.start), end=float(w.end), text=w.text)
                        for w in chunk
                    ],
                    visual_query=str(intent.get("query") or _cinematic_query(styled_text, emotion)),
                    emotion=emotion,
                    pacing="fast",
                    transition_type=transition_type,
                    clip_length_seconds=segment_clip_len,
                )
            )

    return plan
