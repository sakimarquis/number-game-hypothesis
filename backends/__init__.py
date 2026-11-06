"""Backend factory for the selected (model, backend) binding."""

import os

from configs.backend_registry import API_BACKENDS, BACKEND_CHOICES
from configs.settings import BackendConfig
from configs.layout import model_root


def create_backend(backend_config: BackendConfig, concurrency: int = 8):
    model_info = backend_config.model_info
    backend = backend_config.backend
    binding = backend_config.binding
    if backend in API_BACKENDS:
        from backends.api import ApiBackend
        return ApiBackend(model_info.id, provider=backend,
                          concurrency=concurrency,
                          enable_thinking=backend_config.enable_thinking,
                          reasoning_effort=backend_config.reasoning_effort,
                          top_logprobs=backend_config.top_logprobs,
                          thinking_protocol=binding.thinking_protocol,
                          supports_json_schema=binding.supports_json_schema)
    if backend == "google_cloud":
        from backends.google_cloud import GoogleCloudBackend
        return GoogleCloudBackend(
            model_info.id,
            api_key_env=binding.api_key_env,
            concurrency=concurrency,
            enable_thinking=backend_config.enable_thinking,
            execution=binding.execution,
            batch_priority=binding.batch_priority,
            supports_json_schema=binding.supports_json_schema,
            batch_root=os.path.join(model_root(backend_config.result_key), "batch"),
        )
    if backend == "hf":
        from backends.hf import HfBackend
        return HfBackend(
            model_info.id,
            enable_thinking=backend_config.enable_thinking,
            batch_size=binding.effective_batch_size(enable_thinking=backend_config.enable_thinking),
        )
    if backend == "tinker":
        from backends.tinker import TinkerBackend
        return TinkerBackend(
            model_info.id,
            renderer_name=binding.renderer,
            thinking_renderer_name=binding.thinking_renderer,
            enable_thinking=backend_config.enable_thinking,
            concurrency=concurrency,
            tokenizer_trust_remote_code=binding.tokenizer_trust_remote_code,
        )
    raise ValueError(
        f"Unknown backend: {backend!r}. "
        f"Expected one of {set(BACKEND_CHOICES)}."
    )
