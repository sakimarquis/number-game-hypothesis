# Inference backends

Backends execute model requests; [scoring/eval_modes.py](../scoring/eval_modes.py) owns evaluation protocols and parsing. The asynchronous [Backend](base.py) interface provides `generate()` and, where supported, binary logprob scoring. Experiment semantics are in [tasks/README.md](../tasks/README.md); setup and cache replacement rules are in the [root README](../README.md).

## Choose a backend

Model defaults and supported bindings live in [configs/model_aliases.yml](../configs/model_aliases.yml). Use `--backend` to select another configured binding, and `--thinking` to enable its thinking/reasoning mode.

| Backend | Execution | Logprob scoring | Connection |
| --- | --- | --- | --- |
| `openai` | Hosted Chat Completions | When the resolved binding supports it and reasoning effort is `none`. | `OPENAI_API_KEY`. |
| `openrouter` | OpenRouter Chat Completions | Depends on model binding and reasoning configuration. | `OPENROUTER_API_KEY`. |
| `oai` | Generic Chat Completions gateway | Unavailable. | `OAI_BASE_URL`, `OAI_API_KEY`. |
| `google_cloud` | Gemini Developer API; Batch by default | Unavailable. | Binding's `api_key_env`; currently `GCP_Gemini`. |
| `local` | vLLM/SGLang Chat Completions | Server logprobs. | `LOCAL_BASE_URL`; default `http://localhost:30000/v1`. |
| `hf` | Direct Transformers/PyTorch | Yes/No token logits. | Local model loading. |
| `tinker` | Tinker sampling client | Teacher-forced candidate logprobs. | Tinker SDK credentials. |

```powershell
uv run run_number_game.py --model gemma4-31b --backend local --thinking
uv run run_number_game.py --model gemini36-flash --allow-text-posterior
uv run run_number_game.py --model gemini36-flash --online --allow-text-posterior
```

The runner uses configured `POSTERIOR.method` when logprobs are available and `text_trials` otherwise. The root README explains the required text-trial cost preflight. Most text-trial backends return a full binary array; OAI returns `{"yes": [target, ...]}`, which the scorer converts to the same vector. These are backend-specific formats, not interchangeable parser fallbacks.

## Thinking and batching

Each evaluation mode supplies its own `thinking_budget` from [configs/number_game.yml](../configs/number_game.yml); a model's `thinking_generation_tokens` in [configs/model_aliases.yml](../configs/model_aliases.yml) overrides the hypothesis-generation budgets for its thinking runs. `max_tokens` is the requested answer budget; thinking adds headroom. Direct callers must pass a budget when thinking is enabled. An additive output cap alone does not guarantee that a provider will leave enough tokens for the answer.

| Path | Budget behavior |
| --- | --- |
| HF | A logits processor forces the tokenizer's thinking-close sequence at the cap. |
| Local text protocol | Two calls: capped thinking, then a separate answer continuation. |
| Local channel protocol | One combined stream; the serving reasoning parser separates channels. |
| OpenAI / OpenRouter | Provider-managed reasoning; OpenRouter adjusts the total using its effort allocation policy. |
| Gemini | `MINIMAL` without `--thinking`, `MEDIUM` with it; MEDIUM reserves at least 2048 thinking tokens, but allocation remains provider-controlled. |
| Tinker | Additive total budget without forced thinking closure; a thought that reaches the cap leaves an empty answer. |

HF derives thinking boundaries from the tokenizer. Local bindings explicitly declare `thinking_protocol: text` or `channel`; the client does not infer this from a model name. Tinker remote tokenizer code is enabled only by the binding's `tokenizer_trust_remote_code` setting. Tinker sampling accepts only `temperature`, `top_p`, and `top_k`; configured `min_p`, `presence_penalty`, and `repetition_penalty` are not applied.

For logits with thinking, `POSTERIOR.thinking_mode: shared_prefix` reasons once per evidence prefix and reuses it for all targets. `per_target` reasons separately and requires backend support; Tinker does not support it. `CONCURRENCY` limits online requests; HF instead uses binding-specific GPU `batch_size`. Free-text modes retry only failed JSON parses.

## Resume and failure handling

### Gemini Batch

The `google_cloud` implementation uses the Gemini Developer API, not Vertex. It leaves Gemini 3.x sampling controls unset. Batch execution submits condition-wide requests through the File API and monitors at most four conditions concurrently. Hypothesis parse retries and posterior recovery submit only the affected rows; provider-failed rows have bounded transport retries.

Jobs are staged under `results/{model}/batch/{logical-key}-{request-hash}/`. Saved manifests let reruns resume submitted jobs. Completed sidecars are removed after the canonical raw cache is saved; failed, cancelled, or expired jobs are not silently rerouted online. `--online` explicitly switches execution without changing result paths or semantic cache identity.

### Online OAI and OpenRouter

Text trials checkpoint completed responses under the condition's `data/.posterior_text_trials_progress/`. Reruns resume only missing conceptual `(prefix, trial)` requests, including recovery rounds. The manifest must match the cache identity, prefixes, trial count, domain, and text format; incompatible progress requires explicit replacement with `--refresh-raw-cache`.

Retryable transport failures receive bounded failed-key sweeps while successful responses remain checkpointed. Exhausted malformed attempts are archived before their prefix retry state resets. Analysis ignores progress sidecars; successful publication of `posterior.json` removes them. Request timeouts and retry counts are owned by [configs/backend_registry.py](../configs/backend_registry.py).

### Diagnose failures

Empty or partial answers can result from exhausted output budgets, especially on provider-managed thinking paths. Hypothesis parse failures are recorded in `data/hypothesis_eval_parse_failures.json` or `data/hypothesis_generation_parse_failures.json`. Posterior vector validity and recovery semantics are defined in [tasks/README.md](../tasks/README.md#posterior-predict-membership).

Thinking runs (`--thinking` or a reasoning effort) also write `thinking_diagnostics.per_prefix` into each raw cache: for every prefix, one record per generation request in request order (each hypothesis-mode attempt, each text-trial request across recovery phases, or the one shared-prefix thought). Each record has `thinking_tokens` (`null` when the provider does not report it), `forced_close` (thinking hit its budget and was closed by our code), and `truncated` (the answer stopped at its output cap). Sources per path:

| Path | `thinking_tokens` | `forced_close` | `truncated` |
| --- | --- | --- | --- |
| HF | Tokens before the close boundary | Budget processor forced the close | No EOS within `max_new_tokens` |
| Tinker | Re-encoded parsed thought; all sampled tokens for shared thoughts or a response cut mid-thought | Shared thought stopped on `length` | Stopped on `length` |
| OpenAI-compatible | `reasoning_tokens` from usage; stage-1 completion tokens on two-stage or shared thinking | Stage-1 thought stopped on `length` | `finish_reason == "length"` |
| Gemini | `thoughts_token_count` | Never (provider-managed) | `finish_reason == MAX_TOKENS` |

Per-target HF posterior thinking records nothing. Caches written before this field existed have no thinking diagnostics.

## Extend a backend

Implement the asynchronous methods in [base.py](base.py), declare capabilities explicitly, and wire the backend into [create_backend](../backends/__init__.py), the backend registry, and model bindings. Implement `score_binary()` and shared-thinking hooks only when supported. Standard Chat Completions gateways should use the existing `oai` provider.
