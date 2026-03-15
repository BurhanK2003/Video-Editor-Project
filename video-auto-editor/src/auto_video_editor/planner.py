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
    # Viral style pacing: faster voice means smaller caption chunks.
    if words_per_second >= 3.4:
        return 2
    if words_per_second >= 2.7:
        return 3
    if words_per_second >= 1.9:
        return 4
    return 5


def _chunk_words(words: list[WordToken], target_size: int) -> list[list[WordToken]]:
    chunks: list[list[WordToken]] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + target_size]
        i += target_size
        if len(chunk) == 1 and chunks:
            chunks[-1].extend(chunk)
            continue
        if len(chunk) > 6:
            chunk = chunk[:6]
        chunks.append(chunk)

    # Enforce minimum 2 words where possible.
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
    keywords = suggest_scene_keywords(text, max_keywords=3)
    base = " ".join(keywords) if keywords else "nature"

    if BRAND_PHRASE.lower() in text.lower():
        return "cinematic forest mist sunlight nature life close-up 4k"

    if emotion == "suspense":
        return f"cinematic dramatic {base} moody lighting 4k"
    if emotion == "excitement":
        return f"cinematic action {base} dynamic movement 4k"
    return f"cinematic {base} natural light movement 4k"


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
            emotion = _emotion_for_text(chunk_text)
            transition_type = _transition_for_segment(
                text=chunk_text,
                emotion=emotion,
                words_per_second=words_per_second,
                prev_transitions=recent_transitions,
            )
            recent_transitions.append(transition_type)

            transition_after = clip_len >= 1.2 and consumed_words < len(words)
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
                    visual_query=_cinematic_query(styled_text, emotion),
                    emotion=emotion,
                    pacing="fast",
                    transition_type=transition_type,
                    clip_length_seconds=clip_len,
                )
            )

    return plan
