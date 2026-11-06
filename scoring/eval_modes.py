"""Scoring pipelines for each evaluation mode.

Each scorer takes (backend, condition, prefixes, task context) and returns a
ScoredPrefixes with values aligned to the prefixes. Value types differ by eval mode:
  posterior:             np.ndarray of shape (domain,)
  hypothesis_eval:       {hypothesis_label: probability}
  hypothesis_generation: [{hypothesis: str, confidence: float}]
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from configs.layout import condition_batch_key
from configs.settings import GREEDY, ExperimentCondition, GenerationSamplingConfig
from scoring.text_trial_progress import INITIAL_PHASE, RECOVERY_PHASES, TextTrialProgress
from tasks.prompt_space import PromptSpaceSpec, build_prompt_space_spec
from tasks.prompts import (
    compose_preamble,
    compose_binary_messages,
    compose_posterior_vector_messages,
    compose_posterior_yes_index_messages,
    compose_posterior_reasoning_messages,
    compose_hypothesis_eval_messages,
    compose_hypothesis_generation_messages,
)
from tasks.stimuli import TaskContext
from utils import save_json

_NUMBER_PREFIX_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
TEXT_TRIAL_FORMAT = "full_domain_binary_vector_v2"


class TextTrialTransportFailure(RuntimeError):
    """All bounded transport retries failed; durable peers remain checkpointed."""


@dataclass(frozen=True)
class ScoredPrefixes:
    """Scorer values plus optional diagnostics, aligned with the scored prefixes.

    `thinking` holds one GenerateResult.thinking_record() per generation request
    made for each prefix (every attempt or trial, or the shared thought).
    """

    values: list[Any]
    prefix_diagnostics: list[dict[str, Any]] | None = None
    thinking: list[list[dict]] | None = None


def posterior_vector_max_tokens(domain: int) -> int:
    """Answer budget for a full-domain JSON array of binary integers."""
    return max(64, 4 * domain + 16)


def _posterior_vector_schema(domain: int) -> dict:
    return {
        "type": "array",
        "minItems": domain,
        "maxItems": domain,
        "items": {"type": "integer", "enum": [0, 1]},
    }


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()


def _strip_fences(text: str) -> str:
    """Extract content from markdown code fences, or return stripped text."""
    m = re.search(r"```\w*\n?(.*?)\n?```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _extract_balanced_json(text: str, opener: str) -> str | None:
    """Return the first balanced JSON object/array embedded in arbitrary text."""
    closer = "}" if opener == "{" else "]"
    start = text.find(opener)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _json_candidates(text: str, opener: str) -> list[str]:
    """Try the full text first, then the first embedded balanced JSON snippet."""
    stripped = _strip_fences(_strip_thinking(text))
    embedded = _extract_balanced_json(stripped, opener)
    return [c for c in (stripped, embedded) if c]


def _make_parse_failure_event(
    prefix: list[int],
    text: str,
    status: str,
    attempt: int,
    max_retries: int,
    parse_error=None,
) -> dict:
    return {
        "prefix": list(prefix),
        "status": status,
        "attempt": attempt,
        "max_retries": max_retries,
        "parse_error": f"{type(parse_error).__name__}: {parse_error}" if parse_error else None,
        "raw": text,
    }


def _retry_json_messages(messages: list[dict], instruction: str) -> list[dict]:
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "The previous response was not valid JSON for this task. "
                f"{instruction} Output only JSON, with no explanations, markdown, or extra text."
            ),
        },
    ]


def _publish_parse_failures(path: str | None, events: list[dict]) -> None:
    """Write (or clear) the parse-failure report and print its one-line summary."""
    if not events:
        if path is not None and os.path.exists(path):
            os.remove(path)
        return
    summary = {
        "n_events": len(events),
        "n_retry_events": sum(event["status"] == "retry" for event in events),
        "n_exhausted": sum(event["status"] == "exhausted" for event in events),
        "n_prefixes": len({tuple(event["prefix"]) for event in events}),
    }
    if path is not None:
        save_json(path, {"summary": summary, "events": events})
    status = f", {summary['n_exhausted']} exhausted" if summary["n_exhausted"] else ", all recovered"
    suffix = f" -> {path}" if path else ""
    print(f"    parse failures: {summary['n_retry_events']} retry events across "
          f"{summary['n_prefixes']} prefixes{status}{suffix}")


def _coerce_llm_float(value) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a numeric probability: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        match = _NUMBER_PREFIX_RE.match(s)
        if match:
            return float(match.group())
    raise ValueError(f"Could not parse float from {value!r}")


def _finite_probability(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and 0 <= value <= 1
    )


def _repair_json_obj(text: str) -> dict[str, float] | None:
    """Best-effort extraction of {key: probability} from malformed LLM output."""
    pairs = re.findall(rf'"(\d+)"\s*:\s*({_NUMBER_PREFIX_RE.pattern})', text)
    if pairs:
        return {k: _coerce_llm_float(v) for k, v in pairs}
    return None


def _parse_json_obj(text: str) -> dict[str, float] | None:
    """Parse a non-empty JSON object from LLM output, with repair fallback."""
    candidates = _json_candidates(text, "{")
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj:
                return {str(k): _coerce_llm_float(v) for k, v in obj.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    for candidate in candidates:
        repaired = _repair_json_obj(candidate)
        if repaired is not None:
            return repaired
    return None


def _validate_hypothesis_eval(parsed: dict[str, float], labels: list[str]) -> dict[str, float]:
    """Require one finite probability in [0, 1] for every numbered prompt candidate."""
    expected_keys = {str(i) for i in range(1, len(labels) + 1)}
    if set(parsed) != expected_keys:
        missing = sorted(expected_keys - set(parsed), key=int)
        unknown = sorted(set(parsed) - expected_keys)
        raise ValueError(f"Candidate keys do not match prompt: missing={missing}, unknown={unknown}")
    if not all(_finite_probability(value) for value in parsed.values()):
        raise ValueError("Candidate probabilities must be finite numbers in [0, 1]")
    if sum(parsed.values()) <= 0:
        raise ValueError("Candidate probabilities must have positive total mass")
    return parsed


def _normalized_label_mass(parsed: dict[str, float], labels: list[str]) -> dict[str, float]:
    """Map validated numbered keys to labels, summing duplicate labels, and normalize."""
    result: dict[str, float] = {}
    for key, value in parsed.items():
        label = labels[int(key) - 1]
        result[label] = result.get(label, 0.0) + value
    total = sum(result.values())
    return {label: value / total for label, value in result.items()}


def _repair_json_array(text: str) -> str:
    """Repair common malformed hypothesis-generation arrays."""
    text = re.sub(r',\s*confidence"\s*:', '", "confidence":', text)
    last_brace = text.rfind('}')
    if last_brace >= 0 and last_brace < len(text) - 1:
        text = text[:last_brace + 1] + ']'
    return text


def _extract_hyp_conf(arr: list) -> list[dict]:
    """Normalize generated items; any malformed item rejects the whole list (returns [])."""
    items = []
    for item in arr:
        if not isinstance(item, dict) or "hypothesis" not in item or "confidence" not in item:
            return []
        hypothesis = item["hypothesis"]
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            return []
        confidence = _coerce_llm_float(item["confidence"])
        if not _finite_probability(confidence):
            return []
        items.append({"hypothesis": hypothesis.strip(), "confidence": confidence})
    return items


def _parse_json_array(text: str) -> list[dict] | None:
    """Parse a non-empty JSON array of {hypothesis, confidence} from LLM output."""
    for raw_candidate in _json_candidates(text, "["):
        for candidate in (raw_candidate, _repair_json_array(raw_candidate)):
            try:
                arr = json.loads(candidate)
                if isinstance(arr, list):
                    items = _extract_hyp_conf(arr)
                    if items:
                        return items
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return None


def _validate_hypothesis_generation(parsed: list[dict], n_hypotheses: int) -> list[dict]:
    """Require a bounded candidate list with some usable confidence mass."""
    if len(parsed) > n_hypotheses:
        raise ValueError(f"Expected at most {n_hypotheses} hypotheses, got {len(parsed)}")
    if sum(item["confidence"] for item in parsed) <= 0:
        raise ValueError("Generated hypotheses must have positive total confidence")
    return parsed


def valid_cached_result(condition: ExperimentCondition, result, spec: PromptSpaceSpec) -> bool:
    """Whether one stored prefix result satisfies the same contract scoring publishes.

    Hypothesis modes publish an empty value for exhausted parses; analysis counts it as exhausted.
    """
    if condition.eval_mode == "posterior":
        return (
            isinstance(result, list)
            and len(result) == condition.domain
            and all(_finite_probability(value) for value in result)
        )
    if condition.eval_mode == "hypothesis_eval":
        return isinstance(result, dict) and (not result or (
            set(result) <= set(spec.unique_labels)
            and all(_finite_probability(value) for value in result.values())
            and sum(result.values()) > 0
        ))
    if result == []:
        return True
    try:
        items = _extract_hyp_conf(result) if isinstance(result, list) else []
        return bool(items) and items == result and bool(
            _validate_hypothesis_generation(items, condition.mode_config.n_hypotheses)
        )
    except (TypeError, ValueError):
        return False


def _preamble(condition: ExperimentCondition, prefix: list[int], ctx: TaskContext) -> tuple[str, PromptSpaceSpec]:
    spec = build_prompt_space_spec(ctx, prefix)
    return compose_preamble(condition, prefix, spec), spec


def text_trial_messages(condition: ExperimentCondition, prefix: list[int], ctx: TaskContext,
                        encoding: str) -> list[dict]:
    """Full-domain judgment request; `yes_index` asks for accepted targets instead of a binary array."""
    preamble, _ = _preamble(condition, prefix, ctx)
    compose = compose_posterior_yes_index_messages if encoding == "yes_index" else compose_posterior_vector_messages
    return compose(preamble, condition.domain)


def _target_messages(preamble: str, domain: int) -> list[list[dict]]:
    return [compose_binary_messages(preamble, target) for target in range(1, domain + 1)]


def _log_prefix_range(prefix: list[int], probs: np.ndarray, suffix: str = "") -> None:
    print(f"    prefix {prefix}: [{probs.min():.3f}, {probs.max():.3f}]{suffix}")


async def _generate_all(
    backend, messages_list, *, max_tokens, generation_config, thinking_budget,
    batch_key=None, response_json_schema=None,
    completion_callback=None, isolate_retryable_errors=False,
):
    """Generate every prompt, returning results (or isolated retryable errors) in order.

    With `isolate_retryable_errors`, each request completes independently so durable
    callbacks see every success even when peers fail transiently.
    """
    kwargs = {
        "max_tokens": max_tokens,
        "thinking_budget": thinking_budget,
        "generation_config": generation_config,
    }
    if response_json_schema is not None and backend.supports_json_schema:
        kwargs["response_json_schema"] = response_json_schema

    if isolate_retryable_errors:
        async def one(index, messages):
            try:
                generated = (await backend.generate([messages], **kwargs))[0]
            except Exception as exc:
                if backend.is_retryable_generation_error(exc):
                    return exc
                raise
            if completion_callback is not None:
                completion_callback(index, generated)
            return generated

        return await asyncio.gather(*[one(index, messages) for index, messages in enumerate(messages_list)])

    results = await backend.generate(messages_list, batch_key=batch_key, **kwargs)
    if completion_callback is not None:
        for index, result in enumerate(results):
            completion_callback(index, result)
    return results


async def score_posterior(
    backend,
    condition: ExperimentCondition,
    prefixes: list[list[int]],
    ctx: TaskContext,
    *,
    progress_path: str | None = None,
    progress_identity: dict | None = None,
) -> ScoredPrefixes:
    """Binary scoring: p(yes) per target per prefix.

    Dispatch precedence: configured method → shared-prefix thinking → one flat
    dispatch over every prefix-target pair.
    """
    cfg = condition.mode_config
    if cfg.method == "text_trials":
        return await _score_posterior_text_trials(
            backend, condition, prefixes, ctx,
            progress_path=progress_path, progress_identity=progress_identity,
        )

    if backend.enable_thinking:
        if cfg.thinking_mode == "shared_prefix":
            return await _score_posterior_logits_shared_thinking(backend, condition, prefixes, ctx)
        if not backend.supports_per_target_thinking:
            raise ValueError(
                f"{type(backend).__name__} does not support per-target posterior thinking; "
                "set POSTERIOR.thinking_mode to 'shared_prefix'."
            )

    # Flattening across prefixes avoids idle gaps between per-prefix client prep.
    return ScoredPrefixes(await _score_posterior_logits_flat(backend, condition, prefixes, ctx))


async def _score_posterior_logits_shared_thinking(backend, condition, prefixes, ctx) -> ScoredPrefixes:
    """One thinking rollout per prefix, then target-wise Yes/No logits."""
    out: list[np.ndarray] = []
    thinking: list[list[dict]] = []
    for prefix in prefixes:
        preamble, _ = _preamble(condition, prefix, ctx)
        scores, n_fallback, record = await backend.score_binary_with_shared_thinking(
            compose_posterior_reasoning_messages(preamble),
            _target_messages(preamble, condition.domain),
            thinking_budget=condition.mode_config.thinking_budget,
        )
        probs = np.asarray(scores, dtype=np.float32)
        out.append(probs)
        thinking.append([record])
        _log_prefix_range(prefix, probs, f", {n_fallback} fallbacks" if n_fallback else "")
    return ScoredPrefixes(out, thinking=thinking)


async def _score_posterior_logits_flat(backend, condition, prefixes, ctx) -> list[np.ndarray]:
    """One score_binary dispatch over all prefix-target pairs."""
    all_msgs: list[list[dict]] = []
    for prefix in prefixes:
        preamble, _ = _preamble(condition, prefix, ctx)
        all_msgs.extend(_target_messages(preamble, condition.domain))

    print(f"    [posterior] dispatching {len(all_msgs)} requests across {len(prefixes)} prefixes...")
    scores, n_fallback = await backend.score_binary(all_msgs, thinking_budget=condition.mode_config.thinking_budget)
    scores_np = np.asarray(scores, dtype=np.float32)

    out = [scores_np[i * condition.domain:(i + 1) * condition.domain] for i in range(len(prefixes))]
    for prefix, probs in zip(prefixes, out):
        _log_prefix_range(prefix, probs)
    if n_fallback:
        print(f"    [posterior] {n_fallback} fallbacks total")
    return out


def _parse_binary_vector(text: str, domain: int, encoding: str) -> np.ndarray | None:
    """Parse one full-domain judgment into a 0/1 vector, or None when malformed."""
    if encoding == "yes_index":
        for candidate in _json_candidates(text, "{"):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or set(payload) != {"yes"}:
                continue
            yes = payload["yes"]
            if (
                not isinstance(yes, list)
                or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= domain for value in yes)
                or len(set(yes)) != len(yes)
            ):
                continue
            vector = np.zeros(domain, dtype=float)
            vector[np.asarray(yes, dtype=int) - 1] = 1.0
            return vector
        return None

    for candidate in _json_candidates(text, "["):
        try:
            values = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(values, list) or len(values) != domain:
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value not in (0, 1)
            for value in values
        ):
            continue
        return np.asarray(values, dtype=float)
    return None


async def _score_posterior_text_trials(
    backend,
    condition: ExperimentCondition,
    prefixes: list[list[int]],
    ctx: TaskContext,
    *,
    progress_path: str | None = None,
    progress_identity: dict | None = None,
) -> ScoredPrefixes:
    """Estimate target marginals with resumable full-domain vector judgments."""
    cfg = condition.mode_config
    encoding = backend.text_trial_encoding
    prefix_messages = [text_trial_messages(condition, prefix, ctx, encoding) for prefix in prefixes]
    # Starts with the eval-mode key so release_batch_sidecars' prefix cleanup covers it.
    batch_key = condition_batch_key(condition, f"{condition.eval_mode}-text-trials")
    progress = TextTrialProgress(
        progress_path,
        cache_identity=progress_identity,
        text_format=TEXT_TRIAL_FORMAT,
        prefixes=prefixes,
        n_trials=cfg.n_trials,
        domain=condition.domain,
    )
    durable_progress = progress_path is not None
    retry_sweeps = backend.text_trial_retry_sweeps if durable_progress else 0

    vectors = np.full((len(prefixes), cfg.n_trials, condition.domain), np.nan)
    requested_counts = np.zeros(len(prefixes), dtype=int)
    thinking: list[list[dict]] = [[] for _ in prefixes]
    invalid_vectors = 0

    async def complete_phase(phase: str, prefix_indices: list[int], recovery_number: int | None = None) -> None:
        structured = recovery_number is not None
        pending = [
            (prefix_index, trial_index)
            for prefix_index in prefix_indices
            for trial_index in range(cfg.n_trials)
            if not progress.has_result(phase, prefix_index, trial_index)
        ]
        total = len(prefix_indices) * cfg.n_trials
        completed = total - len(pending)
        if structured:
            print(
                f"    [posterior] structured retry {recovery_number}/{len(RECOVERY_PHASES)} for "
                f"{len(prefix_indices)} all-invalid prefixes: "
                f"{len(pending)}/{total} remaining requests ({completed} resumed)..."
            )
        else:
            print(
                f"    [posterior] dispatching {len(pending)}/{total} remaining "
                f"full-domain vector requests ({cfg.n_trials} trials per prefix, "
                f"{completed} resumed)..."
            )
        if not pending:
            return

        for retry_sweep in range(retry_sweeps + 1):
            sweep_pending = list(pending)

            def checkpoint_result(result_index, result) -> None:
                prefix_index, trial_index = sweep_pending[result_index]
                progress.record_result(phase, prefix_index, trial_index, result.text, result.thinking_record())

            results = await _generate_all(
                backend,
                [prefix_messages[prefix_index] for prefix_index, _trial in sweep_pending],
                max_tokens=posterior_vector_max_tokens(condition.domain),
                generation_config=GenerationSamplingConfig(temperature=cfg.temperature),
                thinking_budget=(
                    max(cfg.thinking_budget, backend.structured_recovery_thinking_headroom_tokens)
                    if structured else cfg.thinking_budget
                ),
                batch_key=f"{batch_key}-missing-prefix-retry-{recovery_number}" if structured else batch_key,
                response_json_schema=_posterior_vector_schema(condition.domain) if structured else None,
                completion_callback=checkpoint_result,
                isolate_retryable_errors=durable_progress,
            )
            failures = [
                (sweep_pending[index], result)
                for index, result in enumerate(results)
                if isinstance(result, Exception)
            ]
            if not failures:
                return
            pending = [key for key, _exc in failures]
            if retry_sweep < retry_sweeps:
                print(
                    f"    [posterior] retrying {len(pending)} transient "
                    f"transport failures ({retry_sweep + 1}/{retry_sweeps})..."
                )
                continue
            raise TextTrialTransportFailure(
                f"{len(failures)} text-trial requests still failed after "
                f"{retry_sweeps + 1} transport sweep(s); completed peers were checkpointed"
            ) from failures[0][1]

    def apply_phase(phase: str, prefix_indices: list[int]) -> None:
        nonlocal invalid_vectors
        for prefix_index in prefix_indices:
            for trial_index in range(cfg.n_trials):
                requested_counts[prefix_index] += 1
                thinking[prefix_index].append(progress.result_thinking(phase, prefix_index, trial_index))
                vector = _parse_binary_vector(
                    progress.result_text(phase, prefix_index, trial_index), condition.domain, encoding,
                )
                if vector is None:
                    invalid_vectors += 1
                else:
                    vectors[prefix_index, trial_index] = vector

    initial_prefix_indices = list(range(len(prefixes)))
    await complete_phase(INITIAL_PHASE, initial_prefix_indices)
    apply_phase(INITIAL_PHASE, initial_prefix_indices)

    valid_counts = np.isfinite(vectors).all(axis=2).sum(axis=1)
    missing = np.flatnonzero(valid_counts == 0)
    for recovery_number, phase in enumerate(RECOVERY_PHASES, start=1):
        if not len(missing):
            break
        plan = progress.ensure_recovery_plan(phase, missing.tolist())
        await complete_phase(phase, plan, recovery_number)
        apply_phase(phase, plan)
        valid_counts = np.isfinite(vectors).all(axis=2).sum(axis=1)
        missing = np.flatnonzero(valid_counts == 0)
    if len(missing):
        first = [prefixes[prefix_index] for prefix_index in missing[:10]]
        progress.reset_prefixes(missing.tolist())
        raise ValueError(
            f"No valid full-domain binary vector for {len(missing)} posterior prefixes; "
            f"first missing: {first}"
        )

    probabilities = np.nanmean(vectors, axis=1)
    if invalid_vectors:
        print(f"    [posterior] ignored {invalid_vectors} malformed full-domain vectors")
    for index, prefix in enumerate(prefixes):
        _log_prefix_range(
            prefix, probabilities[index],
            f", valid vectors {int(valid_counts[index])}/{int(requested_counts[index])}",
        )
    return ScoredPrefixes(
        values=list(probabilities),
        prefix_diagnostics=[
            {"requested_vectors": int(requested), "valid_vectors": int(valid)}
            for requested, valid in zip(requested_counts, valid_counts)
        ],
        thinking=thinking,
    )


async def _score_json_mode(
    backend,
    condition: ExperimentCondition,
    prefixes: list[list[int]],
    ctx: TaskContext,
    *,
    max_retries: int,
    failure_log_path: str | None,
    compose_messages,
    parse,
    empty,
    retry_instruction: str,
    generation_config: GenerationSamplingConfig,
) -> ScoredPrefixes:
    """Generate one JSON answer per prefix and re-ask only prefixes whose answer fails `parse`.

    `parse(text, spec)` returns the stored value, or None / raises ValueError for a
    malformed answer; prefixes still malformed after `max_retries` store `empty()`.
    """
    cfg = condition.mode_config
    specs, pending = [], {}
    for index, prefix in enumerate(prefixes):
        preamble, spec = _preamble(condition, prefix, ctx)
        specs.append(spec)
        pending[index] = compose_messages(preamble, spec)
    results = [None] * len(prefixes)
    thinking: list[list[dict]] = [[] for _ in prefixes]
    last_raw: dict[int, str] = {}
    failure_events: list[dict] = []
    batch_key = condition_batch_key(condition, condition.eval_mode)

    for attempt in range(1 + max_retries):
        if not pending:
            break
        indices = sorted(pending)
        gens = await _generate_all(
            backend, [pending[i] for i in indices],
            max_tokens=cfg.max_tokens,
            thinking_budget=cfg.thinking_budget,
            generation_config=generation_config,
            batch_key=f"{batch_key}-attempt-{attempt}",
        )
        still_pending = {}
        for idx, gen in zip(indices, gens):
            thinking[idx].append(gen.thinking_record())
            text = _strip_thinking(gen.text)
            parse_error = None
            try:
                parsed = parse(text, specs[idx])
            except (TypeError, ValueError) as exc:
                parsed = None
                parse_error = exc
            if parsed is not None:
                results[idx] = parsed
                continue
            last_raw[idx] = text
            still_pending[idx] = _retry_json_messages(pending[idx], retry_instruction)
            if attempt < max_retries:
                failure_events.append(_make_parse_failure_event(
                    prefixes[idx], text, "retry", attempt + 1, max_retries, parse_error=parse_error,
                ))
        pending = still_pending

    for idx in pending:
        results[idx] = empty()
        failure_events.append(_make_parse_failure_event(
            prefixes[idx], last_raw[idx], "exhausted", max_retries + 1, max_retries,
        ))

    _publish_parse_failures(failure_log_path, failure_events)
    for prefix, result in zip(prefixes, results):
        print(f"    prefix {prefix}: {len(result)} hypotheses")
    return ScoredPrefixes(results, thinking=thinking)


async def score_hypothesis_eval(
    backend,
    condition: ExperimentCondition,
    prefixes: list[list[int]],
    ctx: TaskContext,
    *,
    max_retries: int,
    failure_log_path: str | None = None,
) -> ScoredPrefixes:
    """Rate given hypotheses. Values are normalized {label: probability} per prefix."""
    def parse(text, spec):
        parsed = _parse_json_obj(text)
        if parsed is None:
            return None
        return _normalized_label_mass(_validate_hypothesis_eval(parsed, spec.labels), spec.labels)

    return await _score_json_mode(
        backend, condition, prefixes, ctx, max_retries=max_retries, failure_log_path=failure_log_path,
        compose_messages=lambda preamble, spec: compose_hypothesis_eval_messages(preamble, spec.labels, spec.transform_note),
        parse=parse, empty=dict,
        retry_instruction='Return a JSON object like {"1": 0.35, "2": 0.65}.',
        generation_config=GREEDY,
    )


async def score_hypothesis_generation(
    backend,
    condition: ExperimentCondition,
    prefixes: list[list[int]],
    ctx: TaskContext,
    *,
    max_retries: int,
    failure_log_path: str | None = None,
) -> ScoredPrefixes:
    """Generate hypotheses freely. Values are [{hypothesis, confidence}] per prefix."""
    cfg = condition.mode_config

    def parse(text, _spec):
        parsed = _parse_json_array(text)
        return None if parsed is None else _validate_hypothesis_generation(parsed, cfg.n_hypotheses)

    return await _score_json_mode(
        backend, condition, prefixes, ctx, max_retries=max_retries, failure_log_path=failure_log_path,
        compose_messages=lambda preamble, _spec: compose_hypothesis_generation_messages(
            preamble, cfg.n_hypotheses, condition.domain,
        ),
        parse=parse, empty=list,
        retry_instruction='Return a JSON array like [{"hypothesis": "description", "confidence": 0.8}].',
        generation_config=cfg.sampling,
    )


SCORERS = {
    "posterior": score_posterior,
    "hypothesis_eval": score_hypothesis_eval,
    "hypothesis_generation": score_hypothesis_generation,
}
