"""Cross-process end-to-end test (F3-grade): serve.sh -> llama-server ->
FastAPI /reading over real HTTP.

Skipped unless the smoke quant and llama-server binary exist:
    RUN_E2E=1 .venv/bin/python -m pytest tests/test_e2e_serving.py -q
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUANT = Path("/tmp/quants/model.q5_k_m_imx.gguf")
LLAMA_BIN = Path.home() / "llama.cpp/build/bin/llama-server"
APP_PORT = 8078
LLAMA_PORT = 8079

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1" or not QUANT.exists() or not LLAMA_BIN.exists(),
    reason="RUN_E2E=1 and smoke quant required")


def _wait_http(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


@pytest.fixture(scope="module")
def stack():
    llama = subprocess.Popen(
        [str(LLAMA_BIN), "-m", str(QUANT), "--host", "127.0.0.1",
         "--port", str(LLAMA_PORT), "-c", "4096"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _wait_http(f"http://127.0.0.1:{LLAMA_PORT}/health", 180), \
        "llama-server did not come up"
    app_env = {**os.environ,
               "TAROT_LLAMA_SERVER": f"http://127.0.0.1:{LLAMA_PORT}",
               "PYTHONPATH": f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"}
    uvicorn = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tfvn.serve:app",
         "--host", "127.0.0.1", "--port", str(APP_PORT)],
        cwd=str(ROOT), env=app_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _wait_http(f"http://127.0.0.1:{APP_PORT}/health", 90), \
        "tfvn.serve did not come up"
    yield {"llama": f"http://127.0.0.1:{LLAMA_PORT}",
           "app": f"http://127.0.0.1:{APP_PORT}"}

    for proc in (uvicorn, llama):
        proc.send_signal(signal.SIGTERM)
    for proc in (uvicorn, llama):
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_full_stack_health(stack):
    health = json.load(urllib.request.urlopen(stack["app"] + "/health"))
    assert health["ok"] and health["deck_size"] == 78 and health["n_ctx"] == 4096


def test_full_stack_reading_generation(stack):
    """Real draw -> real generation -> real validators over two HTTP hops."""
    out = _post(stack["app"] + "/reading", {
        "question_vi": "Công việc của tôi sắp tới sẽ thế nào?",
        "seed": 42, "n_cards": 1})
    assert "draw" in out and len(out["draw"]) == 1
    assert isinstance(out["reading_vi"], str) and out["reading_vi"]
    # whatever the base model produces, the validator contract must hold
    assert isinstance(out["validation_warning"], bool)
    assert isinstance(out["validator_failures"], list)


def test_full_stack_clarification_and_crisis(stack):
    r = _post(stack["app"] + "/reading",
              {"question_vi": "Bói đi", "seed": 7, "n_cards": 3})
    assert r.get("stop_reading") and "lĩnh vực" in r.get("clarification", "")

    c = _post(stack["app"] + "/reading",
              {"question_vi": "Tôi không muốn sống nữa", "seed": 1})
    assert c.get("crisis_route") and "115" in c["message_vi"]
