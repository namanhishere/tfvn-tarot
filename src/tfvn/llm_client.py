"""OpenAI-protocol LLM client with prompt-hash caching (Wave 2 / Wave 3 generation).

Resolves ``{LLM_BASE_URL}/chat/completions`` with Bearer auth. Compatible with
any OpenAI-protocol gateway (OpenRouter, vLLM, llama.cpp server, self-hosted
relay). Every request is cached by SHA-256 of (model, messages, temperature,
max_tokens) so a cache hit never re-bills the endpoint — the plan requires
"ALL generations cached by prompt hash".

Env keys (all optional; sensible defaults):
  LLM_BASE_URL        default https://llm.namanhishere.com/v1
  LLM_API_KEY         required for live calls; missing -> offline/dry-run mode
  LLM_MODEL           default deepseek-v4-flash
  LLM_MODEL_SONNET    second-tier model for judge/critique calls (Wave 3 mix:
                      Judge + critique on Sonnet, bulk generation on Haiku)
  LLM_TIMEOUT         per-request timeout in seconds (default 120)
  GEN_CACHE_DIR       cache dir for prompt-hash caching (default .cache/gen)

Per-call model override: ``chat(..., model=...)`` / ``chat_json(..., model=...)``
select a different model than the client default for that call (cache key
includes the model, so the two tiers never collide).

Usage::

    from tfvn.llm_client import LLMClient, load_env
    load_env()                      # reads project .env if present
    client = LLMClient()
    text = client.chat([{"role": "user", "content": "..."}], temperature=0.7)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_BASE_URL = "https://llm.namanhishere.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

_JSON_SCHEMA_HINT = (
    "Respond with ONLY a single JSON object (no markdown fences, no prose before "
    "or after). Every key must be double-quoted."
)


class LLMError(RuntimeError):
    """Raised on transport or API errors after retries are exhausted."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env(env_path: Optional[Path] = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ if unset.

    Minimal dependency-free loader: strips ``export `` prefixes, ignores blank
    lines and ``#`` comments, never overrides an already-set variable (real env
    wins over file). Call once at process start.
    """
    path = env_path or (_project_root() / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class LLMClient:
    """Thin OpenAI-protocol chat client with deterministic prompt-hash caching."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        cache_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        thinking: Optional[dict] = None,
    ) -> None:
        self.base_url = (base_url or _env("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or _env("LLM_API_KEY") or None
        self.model = model or _env("LLM_MODEL") or DEFAULT_MODEL
        self.model_sonnet = _env("LLM_MODEL_SONNET") or ""  # optional 2nd tier
        self.timeout = float(timeout if timeout is not None else _env("LLM_TIMEOUT", "120"))
        self.cache_dir = Path(cache_dir or _env("GEN_CACHE_DIR", ".cache/gen"))
        self.seed = seed if seed is not None else int(_env("GEN_SEED", "42") or "42")
        self._rng = random.Random(self.seed)
        if thinking is not None:
            self.thinking: Optional[dict] = thinking
        elif _env("LLM_THINKING", "disabled").lower() == "enabled":
            self.thinking = None
        else:
            self.thinking = {"type": "disabled"}

    # ------------------------------------------------------------- transport --

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("LLM_API_KEY is not set (set it in .env or the environment)")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:400]
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2.0 * (attempt + 1) + self._rng.uniform(0, 1))
                    continue
                raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
            except json.JSONDecodeError as e:
                # Non-JSON / empty response body (some gateways return 200 with a
                # blank body under load) — transient: retry with backoff, and a
                # hard LLMError only after all attempts are exhausted so one
                # flaky reply cannot kill a long generation run.
                last_err = e
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1) + self._rng.uniform(0, 1))
                    continue
                raise LLMError(f"non-JSON response from {url}: {last_err}") from last_err
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise LLMError(f"transport error to {url}: {last_err}") from last_err
        raise LLMError(f"request to {url} failed after retries: {last_err}")

    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    # ----------------------------------------------------------------- cache --

    def _cache_key(
        self, messages: Sequence[Dict[str, str]], temperature: float, max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        blob = json.dumps(
            {
                "model": model or self.model,
                "messages": messages,
                "temperature": round(float(temperature), 4),
                "max_tokens": int(max_tokens),
                "thinking": self.thinking,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cached(self, key: str) -> Optional[str]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))["content"]
            except Exception:
                return None
        return None

    def _store(self, key: str, content: str) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{key}.json"
            path.write_text(
                json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # cache is best-effort; never fail generation because of it

    # ---------------------------------------------------------------- public --

    def available_models(self) -> List[str]:
        """GET /models — used by the connection smoke test."""
        if not self.api_key:
            raise LLMError("LLM_API_KEY is not set")
        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", [])]

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        json_object: bool = False,
        use_cache: bool = True,
        model: Optional[str] = None,
    ) -> str:
        """One chat completion. Returns assistant content text.

        ``json_object=True`` requests ``response_format={"type": "json_object"}``
        and appends a JSON-only instruction to the last user message; if the
        gateway rejects the response_format field (400), it retries without it
        (some proxies do not implement structured output).

        ``model`` overrides the client default for this call only (used by the
        Wave 3 mix: Judge/critique on Sonnet, bulk generation on Haiku).
        """
        msgs: List[Dict[str, str]] = []
        for i, m in enumerate(messages):
            msgs.append(dict(m))
        if json_object and msgs and msgs[-1]["role"] == "user":
            msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + "\n\n" + _JSON_SCHEMA_HINT}

        key = self._cache_key(msgs, temperature, max_tokens, model=model)
        if use_cache:
            hit = self._cached(key)
            if hit is not None:
                return hit

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": msgs,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(self._chat_url(), payload)
        except LLMError as e:
            if json_object and "400" in str(e):
                payload.pop("response_format", None)
                data = self._post(self._chat_url(), payload)
            elif self.thinking is not None and "400" in str(e):
                payload.pop("thinking", None)
                data = self._post(self._chat_url(), payload)
            else:
                raise
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected chat response shape: {json.dumps(data)[:300]}") from e
        if not content or not content.strip():
            raise LLMError(
                "empty assistant content (reasoning model consumed the token "
                "budget? finish_reason="
                f"{data.get('choices', [{}])[0].get('finish_reason')!r})"
            )
        if use_cache:
            self._store(key, content)
        return content

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Chat completion parsed as JSON. Falls back to extracting the first
        ``{...}`` object from the raw text if the model ignores the format hint."""
        raw = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_object=True,
            use_cache=use_cache,
            model=model,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"model did not return JSON: {raw[:300]!r}")
