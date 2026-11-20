# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU MP bridge for the released LMCache 0.5.8rc1 wheel.

Use vLLM's single-group request tracking with the wheel's packed-token protocol
and engine-driven transfer API. The wheel's connector uses the older KV-cache
spec interface.
"""

from typing import TYPE_CHECKING, Any, Literal

import torch
import zmq
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.common.token_codec import pack_token_ids

from vllm.config import KVTransferConfig, VllmConfig
from vllm.config.kv_transfer import KVCacheAccess
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPConnectorMetadata,
    LMCacheMPConnectorUpstream,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    reformat_block_ids,
)
from vllm.platforms import current_platform
from vllm.v1.core.single_type_kv_cache_manager import get_kv_cache_kind
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpecKind,
    KVQuantMode,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class _SchedulerAdapter(LMCacheMPSchedulerAdapter):
    def __init__(self, **kwargs: Any):
        self._lookup_salts: dict[str, str] = {}
        super().__init__(**kwargs)

    def maybe_submit_lookup_request(
        self, request_id: str, token_ids: list[int], cache_salt: str = ""
    ) -> bool:
        self._lookup_salts[request_id] = cache_salt
        return super().maybe_submit_lookup_request(
            request_id, pack_token_ids(token_ids), cache_salt=cache_salt
        )

    def free_lookup_locks(self, token_ids: list[int], **kwargs: Any) -> None:
        kwargs.setdefault("cache_salt", self._lookup_salts[kwargs["request_id"]])
        super().free_lookup_locks(pack_token_ids(token_ids), **kwargs)

    def end_session(self, request_id: str) -> None:
        super().end_session(request_id)
        self._lookup_salts.pop(request_id, None)


class _LoadStoreOp(LoadStoreOp):
    def __len__(self) -> int:
        return len(self.flat_block_ids)


class LMCacheMPConnector(LMCacheMPConnectorUpstream):
    """Engine-driven XPU MP transfer for one FP16/BF16 full-attention group."""

    @classmethod
    def kv_cache_access(cls, kv_transfer_config: KVTransferConfig) -> KVCacheAccess:
        return KVCacheAccess.IN_PROCESS

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        if not current_platform.is_xpu():
            raise ValueError("This LMCache MP connector requires XPU")
        groups = kv_cache_config.kv_cache_groups
        if (
            len(groups) != 1
            or get_kv_cache_kind(
                groups[0].manager_config, groups[0].kv_cache_specs.values()
            )
            != KVCacheSpecKind.FULL_ATTENTION
        ):
            raise ValueError(
                "XPU LMCache MP requires one full-attention KV-cache group"
            )
        if any(
            not isinstance(spec, AttentionSpec)
            or spec.kv_quant_mode != KVQuantMode.NONE
            or spec.dtype not in (torch.float16, torch.bfloat16)
            for spec in groups[0].kv_cache_specs.values()
        ):
            raise ValueError("XPU LMCache MP requires unquantized FP16/BF16 KV caches")
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise ValueError(
                "XPU LMCache MP does not support decode context parallelism"
            )

        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        config = vllm_config.kv_transfer_config
        assert config is not None
        extra = dict(config.kv_connector_extra_config)
        extra["lmcache.mp.use_vmm_api"] = False
        extra["lmcache.mp.mp_transfer_mode"] = "engine_driven"
        host = config.get_from_extra_config("lmcache.mp.host", "tcp://localhost")
        port = config.get_from_extra_config("lmcache.mp.port", 5555)
        parallel = vllm_config.parallel_config
        strategy = ParallelStrategy(
            vllm_world_size=parallel.world_size,
            vllm_worker_id=parallel.rank,
            tp_size=parallel.tensor_parallel_size,
            pp_size=parallel.pipeline_parallel_size,
            n_servers=1,
            dcp_size=1,
        )
        self.vllm_block_size = vllm_config.cache_config.block_size
        options = {
            "context": zmq.Context.instance(),
            "model_name": vllm_config.model_config.model,
            "vllm_block_size": self.vllm_block_size,
            "parallel_strategy": strategy,
            "extra_config": extra,
        }
        if role == KVConnectorRole.SCHEDULER:
            self.scheduler_adapter = _SchedulerAdapter(
                server_urls=[f"{host}:{port}"], **options
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
        elif role == KVConnectorRole.WORKER:
            self.worker_adapter = LMCacheMPWorkerAdapter(
                server_url=f"{host}:{port}", **options
            )
        else:
            raise ValueError(f"Unknown KVConnectorRole: {role}")

    def _transfer(self, direction: Literal["STORE", "RETRIEVE"]) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        requests = [req for req in metadata.requests if req.direction == direction]
        if not requests:
            return
        event = self.worker_adapter.create_recorded_event()
        submit = (
            self.worker_adapter.batched_submit_retrieve_requests
            if direction == "RETRIEVE"
            else self.worker_adapter.batched_submit_store_requests
        )
        submit(
            [req.request_id for req in requests],
            [req.op for req in requests],
            event,
            cache_salts=[req.cache_salt for req in requests],
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._transfer("RETRIEVE")

    def wait_for_save(self) -> None:
        self._transfer("STORE")

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        tracker = self._get_or_create_request_tracker(request)
        tracker.num_vllm_hit_blocks = num_computed_tokens // self.vllm_block_size
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def _add_metadata(
        self,
        metadata: LMCacheMPConnectorMetadata,
        tracker: LMCacheMPRequestTracker,
        direction: Literal["STORE", "RETRIEVE"],
        start: int,
        end: int,
        skip: int = 0,
    ) -> None:
        metadata.add_request_metadata(
            LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction=direction,
                op=_LoadStoreOp(
                    token_bytes=pack_token_ids(tracker.all_token_ids),
                    block_ids=[tracker.allocated_block_ids[start:end]],
                    start=start * self.vllm_block_size,
                    end=end * self.vllm_block_size,
                    skip_first_n_tokens=skip,
                ),
                cache_salt=tracker.cache_salt,
            )
        )

    def _store_metadata(
        self, metadata: LMCacheMPConnectorMetadata, tracker: LMCacheMPRequestTracker
    ) -> None:
        computed = tracker.num_scheduled_tokens // self.vllm_block_size + max(
            tracker.num_vllm_hit_blocks, tracker.num_lmcache_hit_blocks
        )
        available = min(
            len(tracker.all_token_ids) // self.vllm_block_size,
            len(tracker.allocated_block_ids),
            computed,
        )
        chunk = self.scheduler_adapter.num_blocks_per_chunk()
        start = tracker.num_stored_blocks
        end = start + (available - start) // chunk * chunk
        if end > start:
            self._add_metadata(metadata, tracker, "STORE", start, end)
            tracker.increase_num_stored_blocks(end - start)

    def _process_retrieve_requests(self, metadata: LMCacheMPConnectorMetadata) -> None:
        chunk = self.scheduler_adapter.num_blocks_per_chunk()
        for tracker in self.request_trackers.values():
            if tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue
            start = tracker.num_vllm_hit_blocks // chunk * chunk
            end = tracker.num_lmcache_hit_blocks
            if end > start:
                skip = (tracker.num_vllm_hit_blocks - start) * self.vllm_block_size
                self._add_metadata(metadata, tracker, "RETRIEVE", start, end, skip)
            tracker.state = LMCacheMPRequestState.READY

    def _process_new_requests(
        self, scheduler_output: "SchedulerOutput", metadata: LMCacheMPConnectorMetadata
    ) -> None:
        for request in scheduler_output.scheduled_new_reqs:
            tracker = self._get_request_tracker(request.req_id)
            tracker.increase_num_scheduled_tokens(
                scheduler_output.num_scheduled_tokens[request.req_id]
            )
            self._store_metadata(metadata, tracker)

    def _process_cached_requests(
        self, scheduler_output: "SchedulerOutput", metadata: LMCacheMPConnectorMetadata
    ) -> None:
        requests = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(requests.req_ids):
            tracker = self._get_request_tracker(request_id)
            tracker.append_block_ids(reformat_block_ids(requests.new_block_ids[index]))
            tracker.increase_num_scheduled_tokens(
                scheduler_output.num_scheduled_tokens[request_id]
            )
            self._store_metadata(metadata, tracker)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        tracker = self.request_trackers.get(request.request_id)
        if tracker is None:
            # The worker reports completion even for requests aborted before lookup.
            self.scheduler_adapter.end_session(request.request_id)
            return True, None
        vllm_hits = tracker.num_vllm_hit_blocks * self.vllm_block_size
        lmcache_hits = tracker.num_lmcache_hit_blocks * self.vllm_block_size
        pending, params = super().request_finished(request, block_ids)
        if (
            request.kv_transfer_params is not None
            and "cached_token_stats" in request.kv_transfer_params
        ):
            assert params is not None
            params["cached_token_stats"] = {
                "num_vllm_cached_tokens": vllm_hits,
                "num_lmcache_cached_tokens": lmcache_hits,
            }
        return pending, params

    def shutdown(self) -> None:
        if self.role == KVConnectorRole.SCHEDULER:
            self.scheduler_adapter.shutdown()
        else:
            self.worker_adapter.shutdown()
