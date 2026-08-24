import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest

from evals.provider import MockProvider, provider_from_spec


def test_mock_cycles_and_formats():
    p = MockProvider(["a {prompt} b", "c"])
    assert p.generate("X") == "a X b"
    assert p.generate("Y") == "c"
    assert p.generate("Z") == "a Z b"


def test_spec_parsing():
    assert provider_from_spec("mock").name == "mock"
    ls = provider_from_spec("llama-server@http://127.0.0.1:8080")
    assert ls.base_url == "http://127.0.0.1:8080"
    hf = provider_from_spec("hf@Qwen/Qwen3-1.7B")
    assert hf.device == "cpu" and hf.model_path == "Qwen/Qwen3-1.7B"
    hf2 = provider_from_spec("hf@/models/x:cuda")
    assert hf2.device == "cuda"
    with pytest.raises(ValueError):
        provider_from_spec("bogus")


def test_mock_loglikelihood_deterministic():
    p = MockProvider()
    s1 = p.loglikelihood("tiền tố cố định", "đáp án A")
    s2 = p.loglikelihood("tiền tố cố định", "đáp án A")
    assert s1 == s2
