"""Component 4: Pluggable AI Judge (Session LLM default + Local/Cloud OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import Candidate, TranscriptBlock, VALID_CATEGORIES, parse_timestamp

JUDGE_SYSTEM_PROMPT = """You are a harsh viral short-form video editor selecting clips for YouTube Shorts and Instagram Reels from a gaming stream transcript.

TIMING RULES (STRICT):
- Chat spike -> the moment is 10-15 seconds BEFORE the spike.
- Loud voice / exclamation in the transcript -> that line IS the moment (pre-roll only 2-5s; 15s of build-up kills the hook).
- Both signals together -> start just before the vocal line (~10s ahead of chat peak).
- Start ON the hook (first 3 seconds must grab); end right after the payoff.
- Boundaries: 20-60 seconds. `end` = start of the first line AFTER the moment.

SELECTION RULES (APPLY HARSHLY):
Categories: first_reaction, discovery, freakout, fail, hot_take, funny.

THROW OUT (score 0, omit from output):
- Needs prior context / refers to something earlier
- Starts with filler ("uh so yeah", "anyway", "okay so")
- Chat/donation/sponsor reads, waiting screens, dead air
- Junk repeated lines, "[Music]", "Thank you."
- Anything hateful, sexual, or about real people's private lives

SCORING:
- 9-10: anyone scrolling would stop and share
- 7-8: solid, clear hook in first 3s
- 5-6: fine but nobody shares it (include only if nothing better)
- <5: drop

Most 10-minute blocks have 0-2 real candidates; many have none. Emit `[]` if none qualify.

OUTPUT FORMAT:
Return ONLY a valid JSON array (no markdown fences, no commentary) matching:
[
  {
    "start": "HH:MM:SS",
    "end": "HH:MM:SS",
    "category": "first_reaction",
    "score": 8.5,
    "hook": "SHORT ALL-CAPS HOOK MAX 8 WORDS",
    "title": "Sentence case post title under 70 characters",
    "caption": "1-2 short lines + a comment-driving question. No links.",
    "hashtags": ["#gaming", "#clips", "#streamer", "#viral"],
    "why": "brief reason why it hooks immediately with no context needed"
  }
]
Field rules:
- hook: on-screen title, max ~8 words, names the streamer or subject, makes you need to see it.
- title: post title, sentence case, under 70 characters.
- caption: 1-2 short lines + a comment-driving question. No links.
- hashtags: 4-6 tags including the game/topic tag and the streamer's name tag.
"""


def build_block_prompt(
    block: TranscriptBlock,
    streamer: str = "the streamer",
    game_tag: str | None = None,
    extra_context: str = "",
) -> str:
    """Format a transcript block + signals into a judge user prompt."""
    lines = [
        f"Streamer: {streamer}",
        f"Block #{block.index} ({block.to_dict()['start']} to {block.to_dict()['end']})",
    ]
    if game_tag:
        lines.append(f"Always include hashtag: {game_tag}")
    if block.signals:
        lines.append(f"Detected signals: {json.dumps(block.signals)}")
    if extra_context:
        lines.append(extra_context)
    lines.append("\nTRANSCRIPT:")
    lines.append(block.format_transcript())
    lines.append("\nReturn ONLY the JSON array of candidates (or [] if none).")
    return "\n".join(lines)


def _extract_json_array(raw_text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from raw LLM output, tolerating think tags, code fences, and wrapper objects."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    # Try parsing whole JSON first (handles {"clips": [...]}, single {...}, or [...])
    for candidate_slice in (cleaned,):
        try:
            parsed = json.loads(candidate_slice)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                for key in ("candidates", "clips", "moments", "results", "items"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        return [item for item in val if isinstance(item, dict)]
                if "start" in parsed and "end" in parsed:
                    return [parsed]
        except json.JSONDecodeError:
            pass

    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return []
    try:
        parsed = json.loads(cleaned[start_idx : end_idx + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def validate_candidates(
    raw_items: list[dict[str, Any]],
    min_score: float = 0.0,
) -> list[Candidate]:
    """Validate and normalize raw candidate dicts against the Section 6 schema."""
    valid: list[Candidate] = []
    for item in raw_items:
        try:
            cand = Candidate.from_dict(item)
        except (KeyError, ValueError, TypeError):
            continue

        if cand.score < min_score:
            continue
        if cand.end_sec <= cand.start_sec:
            continue
        if cand.category not in VALID_CATEGORIES:
            cand.category = "funny"
        if not cand.hook:
            cand.hook = cand.title[:50].upper() or "INSANE MOMENT"
        if len(cand.title) > 70:
            cand.title = cand.title[:67].rstrip() + "..."
        if not cand.hashtags:
            cand.hashtags = ["#gaming", "#clips", "#shorts", "#viral"]
        valid.append(cand)
    return valid


class Judge(ABC):
    """Abstract judge backend."""

    @abstractmethod
    def judge_block(
        self,
        block: TranscriptBlock,
        streamer: str = "the streamer",
        game_tag: str | None = None,
    ) -> list[Candidate]:
        """Judge a single transcript block and return validated candidates."""


class OpenAICompatibleJudge(Judge):
    """Judge backed by any OpenAI-compatible endpoint (Ollama, LM Studio, OpenAI, Groq, Gemini)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 120,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("CLIPPER_LLM_BASE_URL")
            or "http://localhost:11434/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("CLIPPER_LLM_MODEL") or "qwen2.5:7b"
        self.api_key = api_key or os.environ.get("CLIPPER_LLM_API_KEY") or "local"
        self.timeout_sec = timeout_sec
        self.temperature = temperature

    def judge_block(
        self,
        block: TranscriptBlock,
        streamer: str = "the streamer",
        game_tag: str | None = None,
    ) -> list[Candidate]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_block_prompt(block, streamer=streamer, game_tag=game_tag),
                },
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise RuntimeError(
                f"LLM HTTP {exc.code} ({url}, model={self.model}): {err_body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"LLM connection failed ({url}, model={self.model}): {exc.reason}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            return []
        content = choices[0].get("message", {}).get("content", "")
        raw_items = _extract_json_array(content)
        return validate_candidates(raw_items)


def list_available_models(
    base_url: str,
    api_key: str = "local",
    timeout_sec: int = 5,
) -> list[str]:
    """Query GET /models on an OpenAI-compatible endpoint (Ollama, LM Studio, OpenAI)."""
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key or 'local'}"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    items = data.get("data") or data.get("models") or []
    names: list[str] = []
    for item in items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name") or item.get("model")
            if mid:
                names.append(str(mid))
        elif isinstance(item, str):
            names.append(item)
    return names


def test_llm_connection(
    base_url: str,
    model: str,
    api_key: str = "local",
    timeout_sec: int = 15,
) -> str:
    """Send a tiny ping completion to verify the LLM endpoint & model respond."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 12,
        "messages": [{"role": "user", "content": "Reply with OK"}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'local'}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    reply = choices[0].get("message", {}).get("content", "").strip() if choices else ""
    return reply or "OK"


class SessionLLMJudge(Judge):
    """Free zero-API-key judge: writes shortlist.json for the session LLM and reads candidates.json."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.shortlist_path = self.run_dir / "shortlist.json"
        self.candidates_path = self.run_dir / "candidates.json"

    def write_shortlist(
        self,
        blocks: list[TranscriptBlock],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "instructions": JUDGE_SYSTEM_PROMPT,
            "metadata": metadata or {},
            "blocks": [b.to_dict() for b in blocks],
        }
        self.shortlist_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.shortlist_path

    def load_candidates(self, path: str | Path | None = None, min_score: float = 0.0) -> list[Candidate]:
        target = Path(path) if path else self.candidates_path
        raw = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "candidates" in raw:
            raw = raw["candidates"]
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON list in {target}")
        return validate_candidates(raw, min_score=min_score)

    def judge_block(
        self,
        block: TranscriptBlock,
        streamer: str = "the streamer",
        game_tag: str | None = None,
    ) -> list[Candidate]:
        if not self.candidates_path.exists():
            return []
        all_cands = self.load_candidates()
        return [
            c
            for c in all_cands
            if c.start_sec >= block.start - 5.0 and c.start_sec <= block.end + 5.0
        ]


def save_candidates(candidates: list[Candidate], path: str | Path) -> Path:
    """Persist validated candidates to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([c.to_dict() for c in candidates], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target
