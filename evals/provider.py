"""Generation providers for the eval harness.

A provider wraps whatever produces Vietnamese readings: a running llama-server,
a local HF transformers model, or a deterministic mock for tests.

Spec strings understood by :func:`provider_from_spec`:
  - ``mock``                     deterministic echo mock (tests only)
  - ``llama-server@URL``         HTTP against llama-server (OpenAI-compatible)
  - ``hf@MODEL_OR_PATH[:DEV]``   local transformers (device default cpu)
"""

from __future__ import annotations

import json
import os.path
import re
import urllib.request


class ProviderError(RuntimeError):
    pass


class GenerationProvider:
    """Base interface: generate() required; loglikelihood optional."""

    name = "base"

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        raise NotImplementedError

    def supports_loglikelihood(self) -> bool:
        return False

    def loglikelihood(self, prefix: str, candidate: str) -> float:
        """Return sum log P(candidate | prefix). Only meaningful when
        supports_loglikelihood() is True."""
        raise NotImplementedError(f"{self.name} does not support loglikelihood")


class MockProvider(GenerationProvider):
    """Deterministic provider for tests. Replies are cycled; a reply containing
    ``{prompt}`` is formatted with the prompt."""

    name = "mock"

    def __init__(self, replies=None) -> None:
        self._replies = list(replies or ["Xin chào."])
        self._i = 0

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        r = self._replies[self._i % len(self._replies)]
        self._i += 1
        return r.format(prompt=prompt)

    def loglikelihood(self, prefix: str, candidate: str) -> float:
        # Deterministic pseudo-score for tests only (not a real probability).
        shared = len(os.path.commonprefix([prefix[-64:], candidate]))
        return float(shared - len(candidate))


class LlamaServerProvider(GenerationProvider):
    """HTTP client against llama-server's OpenAI-compatible endpoint."""

    name = "llama-server"

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict, retries: int = 2) -> dict:
        import time

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                self.base_url + path,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(5 * (attempt + 1))
        raise ProviderError(f"POST {path} failed after {retries + 1} attempts: "
                            f"{last_err}") from last_err

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = self._post("/v1/chat/completions", {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(
                f"unexpected llama-server response: {json.dumps(data)[:300]}") from e


class TransformersProvider(GenerationProvider):
    """Local HF transformers generation + true loglikelihood scoring."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self._tok = None
        self._model = None

    @property
    def name(self) -> str:
        return f"hf:{self.model_path}"

    def supports_loglikelihood(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto"
        ).to(self.device).eval()

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        import torch

        self._ensure_loaded()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        text = self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][ids["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True)

    def loglikelihood(self, prefix: str, candidate: str) -> float:
        import torch

        self._ensure_loaded()
        p_ids = self._tok(prefix, return_tensors="pt").input_ids.to(self.device)
        full_ids = self._tok(prefix + candidate, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            logits = self._model(full_ids).logits
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = full_ids[:, 1:]
        tok_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        n_prefix = p_ids.shape[1]
        return float(tok_lp[0][max(n_prefix - 1, 0):].sum())


def provider_from_spec(spec: str) -> GenerationProvider:
    spec = spec.strip()
    if spec == "mock":
        return MockProvider()
    m = re.match(r"^llama-server@(\S+)$", spec)
    if m:
        return LlamaServerProvider(m.group(1))
    m = re.match(r"^hf@([^:]+)(?::(\S+))?$", spec)
    if m:
        return TransformersProvider(m.group(1), device=m.group(2) or "cpu")
    raise ValueError(f"unknown provider spec: {spec!r}")
