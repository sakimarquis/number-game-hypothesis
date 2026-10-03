"""HuggingFace Transformers backend for local open-source models."""

import inspect

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    LogitsProcessor,
    StoppingCriteria,
)

from backends.base import Backend, GenerateResult
from backends.thinking_codec import (
    HfThinkingCodec,
    ThinkingProtocol,
    find_subsequence,
    suffix_matches,
)
from configs.settings import GenerationSamplingConfig


def _uses_multimodal_loader(model_id: str) -> bool:
    normalized = model_id.lower()
    return any(family in normalized for family in ("gemma-4", "gemma4", "qwen3.5", "qwen3_5"))


class ThinkingBudgetProcessor(LogitsProcessor):
    """Force the tokenizer-owned thinking close sequence after the budget.

    Tracks state per sequence so it works with both single and batched generation.
    """

    def __init__(self, protocol: ThinkingProtocol, prompt_width: int, budget: int):
        self.protocol = protocol
        self.prompt_width = prompt_width
        self.budget = budget
        self.think_tokens: dict[int, int] = {}
        self.in_thinking: dict[int, bool] = {}
        self.force_pos: dict[int, int] = {}
        self.closed: set[int] = set()
        self.reentry_ids = list(dict.fromkeys(protocol.start_ids + protocol.end_ids))

    def __call__(self, input_ids, scores):
        boundary_ids = self.protocol.answer_boundary_ids
        for i in range(input_ids.shape[0]):
            row = input_ids[i].tolist()
            prompt = row[:self.prompt_width]
            generated = row[self.prompt_width:]
            if i not in self.in_thinking:
                last_start = max(find_subsequence(prompt, self.protocol.start_ids), default=-1)
                last_end = max(find_subsequence(prompt, self.protocol.end_ids), default=-1)
                self.in_thinking[i] = self.protocol.starts_in_thinking or last_start > last_end
                self.think_tokens[i] = len(generated)

            if suffix_matches(generated, boundary_ids):
                self.in_thinking[i] = False
                self.force_pos.pop(i, None)
                self.closed.add(i)
            elif suffix_matches(generated, self.protocol.end_ids) and i not in self.force_pos:
                self.in_thinking[i] = False
                if self.protocol.answer_prefix_ids:
                    self.force_pos[i] = len(self.protocol.end_ids)

            if self.in_thinking.get(i, False) or i in self.force_pos:
                self.think_tokens[i] = max(self.think_tokens.get(i, 0), len(generated))
                if self.think_tokens[i] >= self.budget or i in self.force_pos:
                    pos = self.force_pos.get(i, 0)
                    scores[i, :] = float("-inf")
                    scores[i, boundary_ids[pos]] = 0.0
                    if pos + 1 < len(boundary_ids):
                        self.force_pos[i] = pos + 1
                    else:
                        self.force_pos.pop(i, None)

            if i in self.closed and self.protocol.forbid_reentry:
                scores[i, self.reentry_ids] = float("-inf")
        return scores


class PresencePenaltyProcessor(LogitsProcessor):
    """Subtract a fixed penalty from tokens already emitted in this response."""

    def __init__(self, prompt_width: int, penalty: float):
        self.prompt_width = prompt_width
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_width:]
        for i in range(generated.shape[0]):
            if generated.shape[1] > 0:
                scores[i, torch.unique(generated[i])] -= self.penalty
        return scores


class _CapturePostThinkLogits(LogitsProcessor):
    """Capture logits at the step right after the thinking close boundary.

    When the generated prefix ends with the close sequence, scores predict the
    first visible answer token (Yes/No). Captures once per sequence.
    """

    def __init__(self, end_ids: tuple[int, ...], batch_size: int, prompt_width: int = 0):
        self.end_ids = end_ids
        self.batch_size = batch_size
        self.prompt_width = prompt_width
        self.captured: dict[int, torch.Tensor] = {}

    @property
    def all_captured(self) -> bool:
        return len(self.captured) >= self.batch_size

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            generated = input_ids[i, self.prompt_width:].tolist()
            if i not in self.captured and suffix_matches(generated, self.end_ids):
                self.captured[i] = scores[i].clone()
        return scores


class _AllCaptured(StoppingCriteria):
    """Stop generation once all sequences have post-boundary logits captured."""

    def __init__(self, capture: _CapturePostThinkLogits):
        self._capture = capture

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._capture.all_captured


class HfBackend(Backend):
    """Direct Transformers inference; methods are async but run synchronously on the GPU."""

    caller_batched = True

    def __init__(self, model_id: str, device: str = "auto",
                 enable_thinking: bool = False, batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_id = model_id
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking
        self._thinking_codec = HfThinkingCodec(self.tokenizer, self.model_id) if enable_thinking else None
        self._thinking_protocol = self._thinking_codec.protocol if enable_thinking else None
        model_cls = AutoModelForMultimodalLM if _uses_multimodal_loader(model_id) else AutoModelForCausalLM
        self.model = model_cls.from_pretrained(
            model_id, device_map=device, torch_dtype="auto",
        )
        self.model.eval()
        self._logits_to_keep_kwarg = self._resolve_logits_to_keep_kwarg()
        self._yes_ids, self._no_ids = self._resolve_binary_ids()

    def _gen_kwargs(self, max_tokens: int, temperature: float, thinking_budget: int,
                    generation_config: GenerationSamplingConfig | None = None) -> dict:
        temperature = generation_config.temperature if generation_config is not None else temperature
        total = (
            thinking_budget + len(self._thinking_protocol.answer_boundary_ids) + max_tokens
            if self.enable_thinking else max_tokens
        )
        kwargs = dict(max_new_tokens=total, do_sample=temperature > 0,
                      pad_token_id=self.tokenizer.pad_token_id)
        if temperature > 0:
            kwargs["temperature"] = temperature
        if generation_config is not None:
            for name in ("top_p", "top_k", "min_p", "repetition_penalty"):
                value = getattr(generation_config, name)
                if value is not None:
                    kwargs[name] = value
        return kwargs

    def _add_thinking_budget_processor(self, processors: list, prompt_width: int, thinking_budget: int) -> None:
        processors.append(ThinkingBudgetProcessor(
            self._thinking_protocol,
            prompt_width=prompt_width,
            budget=thinking_budget,
        ))

    def _add_presence_penalty_processor(
        self, kwargs: dict, prompt_width: int,
        generation_config: GenerationSamplingConfig | None,
    ) -> None:
        if generation_config is not None and generation_config.presence_penalty:
            kwargs.setdefault("logits_processor", []).append(PresencePenaltyProcessor(
                prompt_width, generation_config.presence_penalty,
            ))

    def _decode_result(self, token_ids: torch.Tensor) -> GenerateResult:
        """Decode final text and retain parsed thinking when present."""
        if self.enable_thinking:
            split = self._thinking_codec.split_generated(token_ids)
            return GenerateResult(text=split.final_text, thinking_text=split.thinking_text)
        return GenerateResult(text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip())

    async def generate(self, messages, max_tokens=1, temperature=0.0, *,
                       thinking_budget: int | None = None, generation_config=None,
                       response_json_schema=None):
        results = await self.generate_batch(
            [messages],
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
            generation_config=generation_config,
        )
        return results[0]

    def _resolve_binary_ids(self) -> tuple[list[int], list[int]]:
        """Return token ID lists for Yes and No variants via encode (tokenizer-agnostic)."""
        yes_ids, no_ids = set(), set()
        for v in ["Yes", "yes", "YES"]:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if ids:
                yes_ids.add(ids[0])
        for v in ["No", "no", "NO"]:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if ids:
                no_ids.add(ids[0])
        return list(yes_ids), list(no_ids)

    async def score_binary(self, messages_list: list[list[dict]],
                           *, thinking_budget: int | None = None) -> tuple[np.ndarray, int]:
        """Batched scoring returning (p_yes, n_fallback).

        Without thinking: single forward pass, logits at last prompt position.
        With thinking: generate thinking (budget-limited), capture logits right
        after the thinking close boundary via LogitsProcessor, falling back to a per-sequence
        recovery forward pass only when capture misses.
        """
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        scores, n_fallback = self._score_binary_rendered(
            messages_list,
            enable_thinking=self.enable_thinking,
            thinking_budget=thinking_budget,
        )
        return scores.float().cpu().numpy(), n_fallback

    @torch.inference_mode()
    def _score_binary_rendered(
        self,
        messages_list: list[list[dict]],
        *,
        enable_thinking: bool,
        thinking_budget: int = 0,
    ):
        texts = [self._render_chat(m, enable_thinking=enable_thinking) for m in messages_list]
        self.tokenizer.padding_side = "left"
        inputs = self._encode_text(texts, padding=True)

        if enable_thinking:
            logits = self._think_then_score(inputs, thinking_budget)
        else:
            logits = self._last_token_logits(inputs)

        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, self._yes_ids].sum(dim=-1)
        p_no = probs[:, self._no_ids].sum(dim=-1)
        denom = p_yes + p_no
        fallback = denom == 0
        n_fallback = fallback.sum().item()
        result = torch.where(fallback, torch.full_like(p_yes, 0.5), p_yes / denom)
        return result, n_fallback

    async def score_binary_with_shared_thinking(
        self,
        thinking_messages: list[dict],
        messages_list: list[list[dict]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate one prefix-level thought, then score target questions from it."""
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        thinking_message = self._generate_thinking_message(thinking_messages, thinking_budget)
        continued = self._continued_messages(thinking_messages, thinking_message, messages_list)

        chunks = []
        n_fallback = 0
        for i in range(0, len(continued), self.batch_size):
            scores, fallback = self._score_binary_rendered(
                continued[i:i + self.batch_size], enable_thinking=False,
            )
            chunks.append(scores)
            n_fallback += fallback
        result = torch.cat(chunks)
        return result.float().cpu().numpy(), n_fallback

    def _last_token_logits(self, inputs: dict) -> torch.Tensor:
        """Return next-token logits without projecting every prompt position when supported."""
        kwargs = dict(inputs)
        if self._logits_to_keep_kwarg is not None:
            kwargs[self._logits_to_keep_kwarg] = 1
        return self.model(**kwargs).logits[:, -1, :]

    def _resolve_logits_to_keep_kwarg(self) -> str | None:
        """Detect the Transformers knob that restricts logits to final positions."""
        forward = getattr(self.model, "forward", None)
        if forward is None:
            return None
        params = inspect.signature(forward).parameters
        if "logits_to_keep" in params:
            return "logits_to_keep"
        if "num_logits_to_keep" in params:
            return "num_logits_to_keep"
        return None

    @torch.inference_mode()
    def _generate_thinking_message(self, messages: list[dict], thinking_budget: int) -> dict:
        """Return an assistant message containing only the shared thinking block."""
        text = self._render_chat(messages, enable_thinking=True)
        inputs = self._encode_text(text)
        kwargs = self._gen_kwargs(max_tokens=0, temperature=0.0, thinking_budget=thinking_budget)
        self._add_thinking_budget_processor(
            kwargs.setdefault("logits_processor", []), inputs["input_ids"].shape[1], thinking_budget,
        )
        output_ids = self.model.generate(**inputs, **kwargs)
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        split = self._thinking_codec.split_generated(new_ids)
        return {
            "role": "assistant",
            "content": self._thinking_protocol.format_block(split.thinking_text),
        }

    def _think_then_score(self, inputs: dict, thinking_budget: int = 0) -> torch.Tensor:
        """Generate thinking, capture logits right after the close boundary.

        Uses _CapturePostThinkLogits to grab the model's next-token prediction at
        the close boundary during generate(), then recovers only the misses
        with a truncation-based forward pass. Returns (batch, vocab)
        logits predicting the first answer token (Yes/No).
        """
        batch = inputs["input_ids"].shape[0]
        prompt_width = inputs["input_ids"].shape[1]
        boundary_ids = self._thinking_protocol.answer_boundary_ids
        capture = _CapturePostThinkLogits(boundary_ids, batch, prompt_width=prompt_width)
        logits_processor = []
        self._add_thinking_budget_processor(logits_processor, prompt_width, thinking_budget)
        logits_processor.append(capture)

        output_ids = self.model.generate(
            **inputs,
            # Full answer-boundary width + one step whose scores predict the answer.
            max_new_tokens=thinking_budget + len(boundary_ids) + 1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,
            logits_processor=logits_processor,
            stopping_criteria=[_AllCaptured(capture)],
        )
        logits_by_row = [capture.captured.get(i) for i in range(batch)]
        missing = [i for i, logits in enumerate(logits_by_row) if logits is None]
        if not missing:
            return torch.stack(logits_by_row)

        recovered = self._recover_uncaptured_logits(inputs, output_ids, missing)
        for row, logits in zip(missing, recovered):
            logits_by_row[row] = logits
        return torch.stack(logits_by_row)

    def _recover_uncaptured_logits(
        self,
        inputs: dict,
        output_ids: torch.Tensor,
        missing: list[int],
    ) -> torch.Tensor:
        """Recover post-boundary logits only for sequences that missed the fast path.

        This is only for rows whose close boundary exists in the generated text
        but was not captured by the fast-path logits processor.
        """
        prompt_len = inputs["input_ids"].shape[1]
        device = inputs["input_ids"].device
        fallback_ids = output_ids[missing].clone()
        attn_mask = torch.zeros((len(missing), output_ids.shape[1]), dtype=torch.long, device=device)
        attn_mask[:, :prompt_len] = inputs["attention_mask"][missing]
        last_positions = []

        for i, seq in enumerate(fallback_ids):
            gen_part = seq[prompt_len:].tolist()
            end_pos = find_subsequence(gen_part, self._thinking_protocol.answer_boundary_ids)
            if not end_pos:
                raise RuntimeError(
                    "HF thinking scoring could not find a generated thinking close boundary; "
                    "answer-logit scoring is unsupported for this tokenizer/template output."
                )
            cut = prompt_len + end_pos[0] + len(self._thinking_protocol.answer_boundary_ids)
            attn_mask[i, prompt_len:cut] = 1
            fallback_ids[i, cut:] = self.tokenizer.pad_token_id
            last_positions.append(cut - 1)

        all_logits = self.model(input_ids=fallback_ids, attention_mask=attn_mask).logits
        pos = torch.tensor(last_positions, device=device)
        return all_logits[torch.arange(len(missing), device=device), pos, :]

    async def generate_batch(self, messages_list: list[list[dict]],
                             max_tokens: int = 1, temperature: float = 0.0,
                             *, thinking_budget: int | None = None,
                             generation_config=None, response_json_schema=None,
                             batch_key=None) -> list[GenerateResult]:
        """Generate responses for multiple prompts in a single batched forward pass."""
        return self._generate_batch_sync(messages_list, max_tokens, temperature, thinking_budget, generation_config)

    @torch.inference_mode()
    def _generate_batch_sync(self, messages_list, max_tokens, temperature, thinking_budget, generation_config):
        if not messages_list:
            return []
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        texts = [self._render_chat(m) for m in messages_list]
        # Left-pad so generated tokens align across sequences.
        if len(messages_list) > 1:
            self.tokenizer.padding_side = "left"
        inputs = self._encode_text(texts, padding=True)
        kwargs = self._gen_kwargs(max_tokens, temperature, thinking_budget, generation_config)
        self._add_presence_penalty_processor(
            kwargs, inputs["input_ids"].shape[1], generation_config,
        )
        if self.enable_thinking:
            self._add_thinking_budget_processor(
                kwargs.setdefault("logits_processor", []),
                inputs["input_ids"].shape[1],
                thinking_budget,
            )
        output_ids = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        results = []
        for j in range(len(messages_list)):
            new_ids = output_ids[j, prompt_len:]
            results.append(self._decode_result(new_ids))
        return results

    def _render_chat(self, messages, *, enable_thinking: bool | None = None) -> str:
        enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if enable_thinking:
            return self._thinking_codec.render_chat(messages, enable_thinking=True)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def _encode_text(self, text, **kwargs):
        return self.tokenizer(text, return_tensors="pt", **kwargs).to(self.model.device)
