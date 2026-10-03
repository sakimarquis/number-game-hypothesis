"""Run a short local robustness matrix for HF thinking parsing and enclosure."""

import argparse
import asyncio
import gc

import torch

from backends.hf import HfBackend
from backends.thinking_codec import ThinkingSplit, find_subsequence


DEFAULT_MODELS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E2B-it")
SINGLE_PROMPT = [{"role": "user", "content": "Is 8 even? Answer Yes or No."}]
BATCH_PROMPTS = (
    [{"role": "user", "content": "Is 2 even? Yes or No."}],
    [{"role": "user", "content": "Is 7 even? Yes or No."}],
)


def _validate_generation(backend, generated_ids, thinking_budget: int) -> tuple[ThinkingSplit, int]:
    """Check the token boundary before trusting the codec's text-level split."""
    protocol = backend._thinking_protocol
    boundary_positions = find_subsequence(generated_ids.tolist(), protocol.answer_boundary_ids)
    if not boundary_positions:
        raise RuntimeError(f"{backend.model_id} produced no complete thinking-to-answer boundary")
    close_position = boundary_positions[0]
    if close_position > thinking_budget:
        raise RuntimeError(
            f"{backend.model_id} closed thinking after {close_position} tokens, beyond budget {thinking_budget}"
        )

    split = backend._thinking_codec.split_generated(generated_ids)
    if not split.thinking_text:
        raise RuntimeError(f"{backend.model_id} parsed an empty thinking trace")
    if not split.final_text:
        raise RuntimeError(f"{backend.model_id} parsed an empty final answer")
    for marker in (protocol.start_text, protocol.end_text):
        if marker and marker in split.final_text:
            raise RuntimeError(f"{backend.model_id} left thinking marker {marker!r} in the final answer")
    return split, close_position


def _generate_checked(backend, messages, thinking_budget: int, answer_tokens: int,
                      max_prompt_tokens: int) -> tuple[ThinkingSplit, int, int]:
    """Generate one raw continuation and verify budget, parse, and prompt length."""
    rendered = backend._render_chat(messages)
    inputs = backend._encode_text(rendered)
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens > max_prompt_tokens:
        raise RuntimeError(f"Prompt has {prompt_tokens} tokens; limit is {max_prompt_tokens}")

    # Use the backend's production processor while retaining raw IDs for assertions.
    kwargs = backend._gen_kwargs(answer_tokens, 0.0, thinking_budget)
    backend._add_thinking_budget_processor(kwargs.setdefault("logits_processor", []), prompt_tokens, thinking_budget)
    with torch.inference_mode():
        output_ids = backend.model.generate(**inputs, **kwargs)
    split, close_position = _validate_generation(
        backend, output_ids[0, prompt_tokens:], thinking_budget,
    )
    return split, close_position, prompt_tokens


def _generate_batch_checked(backend, thinking_budget: int, answer_tokens: int,
                            max_prompt_tokens: int) -> list[ThinkingSplit]:
    """Exercise left-padded, per-row processor state with two short prompts."""
    texts = [backend._render_chat(messages) for messages in BATCH_PROMPTS]
    backend.tokenizer.padding_side = "left"
    inputs = backend._encode_text(texts, padding=True)
    prompt_width = inputs["input_ids"].shape[1]
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    if max(prompt_lengths) > max_prompt_tokens:
        raise RuntimeError(f"Batch prompt has {max(prompt_lengths)} tokens; limit is {max_prompt_tokens}")

    # One processor instance must independently close and parse every batch row.
    kwargs = backend._gen_kwargs(answer_tokens, 0.0, thinking_budget)
    backend._add_thinking_budget_processor(kwargs.setdefault("logits_processor", []), prompt_width, thinking_budget)
    with torch.inference_mode():
        output_ids = backend.model.generate(**inputs, **kwargs)
    return [
        _validate_generation(backend, row[prompt_width:], thinking_budget)[0]
        for row in output_ids
    ]


def check_model(model_id: str, budgets: list[int], answer_tokens: int,
                max_prompt_tokens: int) -> None:
    """Run single-turn, two-turn, batch, public-result, and binary-logit checks."""
    print(f"\n=== {model_id} ===")
    backend = HfBackend(model_id, enable_thinking=True, batch_size=2)
    print(f"protocol: {backend._thinking_protocol.start_text!r} ... {backend._thinking_protocol.end_text!r}")

    for budget in budgets:
        split, close, prompt_len = _generate_checked(
            backend, SINGLE_PROMPT, budget, answer_tokens, max_prompt_tokens,
        )
        print(f"single budget={budget}: close={close}, prompt={prompt_len}, answer={split.final_text!r}")

    # The second response is generated from real assistant history, not a flattened prompt.
    first, _, _ = _generate_checked(
        backend, [{"role": "user", "content": "Is 2 even? Yes or No."}],
        budgets[-1], answer_tokens, max_prompt_tokens,
    )
    history = [
        {"role": "user", "content": "Is 2 even? Yes or No."},
        {"role": "assistant", "content": first.final_text, "reasoning_content": first.thinking_text},
        {"role": "user", "content": "Is 3 odd? Yes or No."},
    ]
    second, _, second_prompt_len = _generate_checked(
        backend, history, budgets[-1], answer_tokens, max_prompt_tokens,
    )
    print(f"multi-turn: thoughts=2/2, prompt={second_prompt_len}, answer={second.final_text!r}")

    batch = _generate_batch_checked(backend, budgets[-1], answer_tokens, max_prompt_tokens)
    print(f"batch: parsed={len(batch)}/{len(BATCH_PROMPTS)}")

    # Public single and batch results must expose traces, not only internal codec state.
    public = asyncio.run(backend.generate(history, max_tokens=answer_tokens, thinking_budget=budgets[-1]))
    if not public.thinking_text or not public.text:
        raise RuntimeError(f"{model_id} generate() did not return both thinking and final text")
    public_batch = asyncio.run(backend.generate_batch(
        list(BATCH_PROMPTS), max_tokens=answer_tokens, thinking_budget=budgets[-1],
    ))
    if any(not result.thinking_text or not result.text for result in public_batch):
        raise RuntimeError(f"{model_id} generate_batch() did not return both thinking and final text")
    p_yes, fallback_count = asyncio.run(backend.score_binary([SINGLE_PROMPT], thinking_budget=budgets[-1]))
    print(
        f"public results: single=yes, batch={len(public_batch)}/{len(BATCH_PROMPTS)}; "
        f"P(Yes)={p_yes.item():.4f}, fallbacks={fallback_count}"
    )

    del backend
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 32])
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    args = parser.parse_args()

    for model_id in args.models:
        check_model(model_id, args.budgets, args.answer_tokens, args.max_prompt_tokens)


if __name__ == "__main__":
    main()
