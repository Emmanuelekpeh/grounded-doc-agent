"""Thin LLM wrapper. Default provider is a local Ollama server (CPU friendly, no API key).

Env vars:
  PAPERWORLD_PROVIDER   ollama (default) | anthropic
  PAPERWORLD_MODEL      model name for the chosen provider (default: qwen2.5:3b / claude-sonnet-5-5)
  OLLAMA_HOST           default http://localhost:11434
  PAPERWORLD_NUM_CTX    Ollama context window in tokens (default 8192; 6 passages + prompt need ~4k)
  PAPERWORLD_TIMEOUT    seconds to wait for a completion (default 900; CPU inference is slow)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

PROVIDER = os.environ.get("PAPERWORLD_PROVIDER", "ollama").lower()
DEFAULT_MODEL = os.environ.get(
    "PAPERWORLD_MODEL", "claude-sonnet-5-5" if PROVIDER == "anthropic" else "qwen2.5:3b")


def _ollama(system: str, user: str, max_tokens: int) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    body = {
        "model": DEFAULT_MODEL,
        "stream": False,
        "format": "json",  # constrain decoding to valid JSON; small models otherwise wrap/ramble
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "num_predict": max_tokens,
                    "num_ctx": int(os.environ.get("PAPERWORLD_NUM_CTX", "8192"))},
    }
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("PAPERWORLD_TIMEOUT", "900"))) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"Ollama error {e.code}: {detail} (is '{DEFAULT_MODEL}' pulled? `ollama list`)") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {host}: {e.reason}. Start it or use --no-llm.") from e
    return data["message"]["content"]


def _anthropic(system: str, user: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install the SDK: pip install anthropic (or use the ollama provider / --no-llm)") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY (or use PAPERWORLD_PROVIDER=ollama / --no-llm)")
    msg = anthropic.Anthropic().messages.create(
        model=DEFAULT_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def complete(system: str, user: str, max_tokens: int = 1800) -> str:
    if PROVIDER == "anthropic":
        return _anthropic(system, user, max_tokens)
    if PROVIDER == "ollama":
        return _ollama(system, user, max_tokens)
    raise RuntimeError(f"Unknown PAPERWORLD_PROVIDER {PROVIDER!r}; use ollama or anthropic")
