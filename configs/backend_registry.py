"""Dependency-light backend identities and OpenAI-compatible provider policy."""

BACKEND_CHOICES = (
    "hf",
    "local",
    "tinker",
    "openai",
    "openrouter",
    "oai",
    "google_cloud",
)

API_PROVIDER_CONFIGS = {
    "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "request_timeout_seconds": 180,
        "sdk_max_retries": 0,
        "text_trial_retry_sweeps": 2,
    },
    # Generic Chat Completions gateway. Unlike `local`, this sends no
    # vLLM/SGLang extensions and assumes no logprob support.
    "oai": {
        "base_url": None,
        "base_url_env": "OAI_BASE_URL",
        "api_key_env": "OAI_API_KEY",
        "require_base_url": True,
        "request_timeout_seconds": 300,
        "sdk_max_retries": 0,
        "text_trial_retry_sweeps": 2,
        "text_trial_encoding": "yes_index",
    },
    # Local OpenAI-compatible server (vLLM or SGLang). Default 30000 follows
    # SGLang; vLLM's default needs LOCAL_BASE_URL=http://localhost:8000/v1.
    "local": {
        "base_url": "http://localhost:30000/v1",
        "base_url_env": "LOCAL_BASE_URL",
        "api_key_env": "LOCAL_API_KEY",
        "default_api_key": "EMPTY",
    },
}

API_BACKENDS = frozenset(API_PROVIDER_CONFIGS)
HOSTED_OAI_BACKENDS = frozenset({"openai", "openrouter"})
TOGGLE_THINKING_BACKENDS = frozenset({"hf", "tinker", "local", "google_cloud"})
# Backends whose transport never returns response logprobs.
NO_LOGPROB_BACKENDS = frozenset({"oai", "google_cloud"})
