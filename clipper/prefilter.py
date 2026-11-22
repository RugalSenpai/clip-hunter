"""Component 3: Free heuristic arousal pre-filter over transcript blocks."""

from __future__ import annotations

import math
import re

from .models import TranscriptBlock

AROUSAL_PHRASES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bno\s+way\b", re.IGNORECASE), 3.0),
    (re.compile(r"\bwhat\s+the\s+(?:fuck|hell|heck|f\*+)\b", re.IGNORECASE), 3.5),
    (re.compile(r"\blet'?s\s+go+\b", re.IGNORECASE), 3.0),
    (re.compile(r"\bi\s+can'?t\b", re.IGNORECASE), 2.0),
    (re.compile(r"\boh\s+my\s+god\b|\bomg\b", re.IGNORECASE), 2.8),
    (re.compile(r"\bholy\s+(?:shit|cow|moly|crap)\b", re.IGNORECASE), 3.0),
    (re.compile(r"\bare\s+you\s+kidding\b|\bno\s+shot\b|\bain'?t\s+no\s+way\b", re.IGNORECASE), 3.0),
    (re.compile(r"\bwhat\s+just\s+happened\b|\bwait\s+a\s+second\b", re.IGNORECASE), 2.5),
    (re.compile(r"\b(?:no+|bro+|yo+|wait+)\b", re.IGNORECASE), 0.7),
    (re.compile(r"\[(?:laughs?|laughter|applause|screaming|cheering)\]", re.IGNORECASE), 2.5),
    (re.compile(r"\b(?:hahaha+|lmao|lmfao)\b", re.IGNORECASE), 2.0),
]

FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:thank\s+you\s+for\s+the\s+(?:sub|donation|gifted|bits))\b", re.IGNORECASE),
    re.compile(r"\b(?:sponsor|use\s+code|link\s+in\s+the\s+description|like\s+and\s+subscribe)\b", re.IGNORECASE),
]


def score_block(block: TranscriptBlock) -> float:
    """Compute a zero-cost arousal score for a transcript block."""
    if not block.segments:
        block.prefilter_score = 0.0
        return 0.0

    full_text = " ".join(seg.text for seg in block.segments)
    words = full_text.split()
    word_count = max(1, len(words))

    phrase_score = 0.0
    phrase_hits = 0
    for pattern, weight in AROUSAL_PHRASES:
        hits = len(pattern.findall(full_text))
        if hits:
            phrase_hits += hits
            phrase_score += min(hits, 12) * weight

    exclamation_count = full_text.count("!")
    question_burst = full_text.count("?!") + full_text.count("!?") + full_text.count("???")
    caps_words = sum(1 for w in words if len(w) >= 3 and w.isalpha() and w.isupper())
    caps_density = caps_words / word_count

    # Rapid speech burst: peak words/sec in any 10-second sub-window
    peak_wps = 0.0
    for seg in block.segments:
        dur = max(0.5, seg.end - seg.start)
        wps = len(seg.text.split()) / dur
        if wps > peak_wps:
            peak_wps = min(wps, 8.0)

    # Penalize sponsor / donation-heavy blocks slightly
    filler_hits = sum(len(p.findall(full_text)) for p in FILLER_PATTERNS)

    # Loudness / chat signals from live mode if present
    loudness_bonus = float(block.signals.get("loudness_spikes", 0)) * 2.5
    chat_bonus = float(block.signals.get("chat_spikes", 0)) * 3.0

    raw_score = (
        phrase_score
        + min(exclamation_count, 20) * 0.9
        + question_burst * 2.0
        + caps_density * 25.0
        + peak_wps * 0.8
        + loudness_bonus
        + chat_bonus
        - filler_hits * 2.0
    )

    # Normalize gently by log(duration_min) so shorter tail blocks aren't unfairly buried
    dur_min = max(1.0, (block.end - block.start) / 60.0)
    normalized = max(0.0, raw_score / math.sqrt(dur_min / 10.0))

    block.prefilter_score = round(normalized, 3)
    block.signals.update(
        {
            "phrase_hits": phrase_hits,
            "exclamations": exclamation_count,
            "caps_density": round(caps_density, 4),
            "peak_wps": round(peak_wps, 2),
            "word_count": word_count,
        }
    )
    return block.prefilter_score


def shortlist_blocks(
    blocks: list[TranscriptBlock],
    top_fraction: float = 0.20,
    min_blocks: int = 2,
    max_blocks: int | None = None,
) -> list[TranscriptBlock]:
    """Score every block and return the top 10-20% highest-arousal blocks in chronological order."""
    if not blocks:
        return []

    for block in blocks:
        score_block(block)

    target_count = max(min_blocks, math.ceil(len(blocks) * top_fraction))
    if max_blocks is not None:
        target_count = min(target_count, max_blocks)
    target_count = min(target_count, len(blocks))

    ranked = sorted(blocks, key=lambda b: b.prefilter_score, reverse=True)
    chosen = ranked[:target_count]
    chosen.sort(key=lambda b: b.index)
    return chosen
