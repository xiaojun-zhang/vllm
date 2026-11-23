# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt LMCache's released MP connector to the XPU image."""

from typing import TYPE_CHECKING, Any

import torch
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector as ExternalLMCacheMPConnector,
)

from vllm.config import KVTransferConfig, VllmConfig
from vllm.config.kv_transfer import KVCacheAccess
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.platforms import current_platform
from vllm.v1.core.single_type_kv_cache_manager import get_kv_cache_kind
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpecKind,
    KVQuantMode,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class LMCacheMPConnector(ExternalLMCacheMPConnector):
    """Engine-driven XPU MP transfer for one FP16/BF16 full-attention group."""

    @classmethod
    def kv_cache_access(cls, kv_transfer_config: KVTransferConfig) -> KVCacheAccess:
        # Workers copy KV tensors to host buffers for the separate MP server.
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
        super().__init__(vllm_config, role, kv_cache_config)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if (
            params is not None
            and "cached_token_stats" in params
            and request.request_id not in self.request_trackers
        ):
            # LMCache 0.5.8 reads stats from the tracker before cleaning it up.
            self._get_or_create_request_tracker(request)
        return super().request_finished(request, block_ids)
