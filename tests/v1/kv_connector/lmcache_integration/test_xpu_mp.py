# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Released XPU wheel: MP cache reuse preserves serving results."""

import asyncio
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="Released LMCache XPU wheel integration"
)

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CHUNK_SIZE = 64
SERVER_ARGS = [
    "--revision",
    REVISION,
    "--host",
    "127.0.0.1",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "2048",
    "--max-num-seqs",
    "1",
    "--max-num-batched-tokens",
    "2048",
    "--gpu-memory-utilization",
    "0.25",
    "--enforce-eager",
    "--no-enable-prefix-caching",
    "--attention-backend",
    "TRITON_ATTN",
]
SERVER_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_STANDALONE_COMPILE": "0",
}


def _mp_args(port, prefix_caching=False):
    connector = {
        "kv_connector": "LMCacheMPConnector",
        "kv_connector_module_path": (
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_xpu_mp_connector"
        ),
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": port,
            "lmcache.mp.use_vmm_api": False,
        },
    }
    args = SERVER_ARGS + ["--kv-transfer-config", json.dumps(connector)]
    if prefix_caching:
        args = [
            "--enable-prefix-caching" if arg == "--no-enable-prefix-caching" else arg
            for arg in args
        ]
    return args


def _complete(server, prompt):
    with server.get_client() as client:
        return client.completions.create(
            model=MODEL,
            prompt=prompt,
            max_tokens=16,
            temperature=0,
            seed=0,
            logprobs=5,
            extra_body={"kv_transfer_params": {"cached_token_stats": {}}},
        ).model_dump()


def _check_result(result, reference, *, cache_hit, vllm_cache_hit=False):
    actual = result["choices"][0]
    expected = reference["choices"][0]
    assert actual["text"] == expected["text"]
    assert actual["logprobs"]["tokens"] == expected["logprobs"]["tokens"]
    assert (
        actual["logprobs"]["token_logprobs"] == expected["logprobs"]["token_logprobs"]
    )
    stats = result["kv_transfer_params"]["cached_token_stats"]
    if vllm_cache_hit:
        assert stats["num_vllm_cached_tokens"] >= CHUNK_SIZE
    else:
        assert stats["num_vllm_cached_tokens"] == 0
    if cache_hit:
        assert stats["num_lmcache_cached_tokens"] >= CHUNK_SIZE
    else:
        assert stats["num_lmcache_cached_tokens"] == 0


@pytest.fixture(scope="module")
def reference():
    import lmcache.xpu_ops

    assert lmcache.xpu_ops.__file__.endswith(".so")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompt = tokenizer.encode(
        (
            "The research notes say Paris is the capital of France. "
            "The river Seine runs through Paris. "
            "This fact remains the same in every note. "
        )
        * 20
        + "\nAccording to the notes, the capital of France is",
        add_special_tokens=False,
    )
    assert len(prompt) > 2 * CHUNK_SIZE
    with RemoteOpenAIServer(MODEL, SERVER_ARGS, env_dict=SERVER_ENV) as server:
        return prompt, _complete(server, prompt)


@contextmanager
def _cache_server(tmp_path, transport):
    rpc_port, http_port, metrics_port = (get_open_port() for _ in range(3))
    command = [
        "lmcache",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(rpc_port),
        "--http-port",
        str(http_port),
        "--prometheus-port",
        str(metrics_port),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--supported-transfer-mode",
        "engine_driven",
        "--l1-size-gb",
        "0.5",
        "--no-l1-use-lazy",
        "--eviction-policy",
        "LRU",
    ]
    if transport == "shm":
        command += ["--shm-name", f"lmcache-xpu-{rpc_port}"]
    log_path = tmp_path / "lmcache-server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{http_port}/healthcheck", timeout=1
                    )
                    if response.ok and response.json()["status"] == "healthy":
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            else:
                raise AssertionError(
                    f"LMCache server did not become healthy:\n{log_path.read_text()}"
                )
            yield rpc_port
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)


@pytest.mark.parametrize("transport", ["shm", "inline"])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_mp_cache_reuse_preserves_output_after_engine_restart(
    reference, tmp_path, transport, prefix_caching
):
    """A warm MP cache must serve real KV data independently of the engine."""
    prompt, baseline = reference
    with _cache_server(tmp_path, transport) as port:
        args = _mp_args(port, prefix_caching)
        with RemoteOpenAIServer(MODEL, args, env_dict=SERVER_ENV) as server:
            _check_result(_complete(server, prompt), baseline, cache_hit=False)
            _check_result(
                _complete(server, prompt),
                baseline,
                cache_hit=True,
                vllm_cache_hit=prefix_caching,
            )
        with RemoteOpenAIServer(MODEL, args, env_dict=SERVER_ENV) as server:
            _check_result(_complete(server, prompt), baseline, cache_hit=True)


async def _wait_request_counts(server, expected):
    deadline = time.monotonic() + 30
    actual = None
    while time.monotonic() < deadline:
        response = await asyncio.to_thread(
            requests.get, server.url_for("metrics"), timeout=1
        )
        response.raise_for_status()
        counts = {
            sample.name: sample.value
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
            if sample.name in ("vllm:num_requests_running", "vllm:num_requests_waiting")
        }
        actual = (
            counts.get("vllm:num_requests_running"),
            counts.get("vllm:num_requests_waiting"),
        )
        if actual == expected:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"Request counts were {actual}, expected {expected}")


@pytest.mark.asyncio
async def test_mp_survives_aborting_a_queued_request(reference, tmp_path):
    """Aborting before a cache lookup must release the request without a crash."""
    prompt, baseline = reference
    with _cache_server(tmp_path, "inline") as port:
        env = {**SERVER_ENV, "VLLM_SERVER_DEV_MODE": "1"}
        with RemoteOpenAIServer(MODEL, _mp_args(port), env_dict=env) as server:
            async with server.get_async_client() as client:

                async def generate():
                    return await client.completions.create(
                        model=MODEL,
                        prompt=prompt,
                        max_tokens=1400,
                        temperature=0,
                        seed=0,
                        extra_body={"ignore_eos": True},
                    )

                tasks = [asyncio.create_task(generate())]
                try:
                    await _wait_request_counts(server, (1, 0))
                    tasks.append(asyncio.create_task(generate()))
                    await _wait_request_counts(server, (1, 1))
                    response = await asyncio.to_thread(
                        requests.post,
                        server.url_for("abort_requests"),
                        json={},
                        timeout=10,
                    )
                    response.raise_for_status()
                    assert response.json()["aborted"] == 2
                    await _wait_request_counts(server, (0, 0))
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            _check_result(_complete(server, prompt), baseline, cache_hit=True)
