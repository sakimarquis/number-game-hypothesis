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

from backends.base import BINARY_VARIANTS, Backend, GenerateResult
from backends.thinking_codec import (
    HfThinkingCodec,
    ThinkingProtocol,
    find_subsequence,
    suffix_matches,
)
from configs.settings import GREEDY, GenerationSamplingConfig


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
        self.forced: set[int] = set()  # rows whose thinking hit the budget
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
                    if i not in self.force_pos:
                        self.forced.add(i)
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
    """Direct Transformers inference; methods are async but run synchronously on the GPU in `batch_size` chunks."""

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
        # Some multimodal forwards do not accept the knob that restricts logits to final positions.
        self._keeps_last_logits = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self._yes_ids, self._no_ids = self._resolve_binary_ids()
        eos = self.model.generation_config.eos_token_id
        self._eos_ids = set(eos if isinstance(eos, list) else [eos])

    def _gen_kwargs(self, max_tokens: int, thinking_budget: int, generation_config: GenerationSamplingConfig) -> dict:
        temperature = generation_config.temperature
        total = (
            thinking_budget + len(self._thinking_protocol.answer_boundary_ids) + max_tokens
            if self.enable_thinking else max_tokens
        )
        kwargs = dict(max_new_tokens=total, do_sample=temperature > 0,
                      pad_token_id=self.tokenizer.pad_token_id)
        if temperature > 0:
            kwargs["temperature"] = temperature
        for name in ("top_p", "top_k", "min_p", "repetition_penalty"):
            value = getattr(generation_config, name)
            if value is not None:
                kwargs[name] = value
        return kwargs

    def _add_thinking_budget_processor(self, processors: list, prompt_width: int, thinking_budget: int) -> ThinkingBudgetProcessor:
        processor = ThinkingBudgetProcessor(self._thinking_protocol, prompt_width=prompt_width, budget=thinking_budget)
        processors.append(processor)
        return processor

    def _add_presence_penalty_processor(
        self, kwargs: dict, prompt_width: int, generation_config: GenerationSamplingConfig,
    ) -> None:
        if generation_config.presence_penalty:
            kwargs.setdefault("logits_processor", []).append(PresencePenaltyProcessor(
                prompt_width, generation_config.presence_penalty,
            ))

    def _decode_result(self, token_ids: torch.Tensor, *, forced_close: bool) -> GenerateResult:
        """Decode final text and retain parsed thinking when present."""
        # Rows that finish early end with EOS (then padding); a row without EOS hit max_new_tokens.
        truncated = not any(int(token) in self._eos_ids for token in token_ids)
        if self.enable_thinking:
            split = self._thinking_codec.split_generated(token_ids)
            return GenerateResult(
                text=split.final_text, thinking_text=split.thinking_text,
                thinking_tokens=split.thinking_tokens, forced_close=forced_close, truncated=truncated,
            )
        return GenerateResult(text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(), truncated=truncated)

    def _resolve_binary_ids(self) -> tuple[list[int], list[int]]:
        """Return token ID lists for Yes and No variants via encode (tokenizer-agnostic)."""
        def first_ids(label):
            encoded = (self.tokenizer.encode(text, add_special_tokens=False) for text in BINARY_VARIANTS[label])
            return list({ids[0] for ids in encoded if ids})

        return first_ids("yes"), first_ids("no")

    async def score_binary(self, messages_list: list[list[dict]],
                           *, thinking_budget: int | None = None) -> tuple[np.ndarray, int]:
        """Batched scoring returning (p_yes, n_fallback).

        Without thinking: single forward pass, logits at last prompt position.
        With thinking: generate thinking (budget-limited) and capture logits right
        after the thinking close boundary via LogitsProcessor.
        """
        return self._score_binary_chunks(
            messages_list, enable_thinking=self.enable_thinking,
            thinking_budget=self._effective_thinking_budget(thinking_budget),
        )

    def _score_binary_chunks(self, messages_list, *, enable_thinking: bool, thinking_budget: int = 0):
        chunks = []
        n_fallback = 0
        for start in range(0, len(messages_list), self.batch_size):
            scores, fallback = self._score_binary_rendered(
                messages_list[start:start + self.batch_size],
                enable_thinking=enable_thinking, thinking_budget=thinking_budget,
            )
            chunks.append(scores)
            n_fallback += fallback
        return torch.cat(chunks).float().cpu().numpy(), n_fallback

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

    def _last_token_logits(self, inputs: dict) -> torch.Tensor:
        """Return next-token logits without projecting every prompt position when supported."""
        kwargs = dict(inputs)
        if self._keeps_last_logits:
            kwargs["logits_to_keep"] = 1
        return self.model(**kwargs).logits[:, -1, :]

    async def _shared_thinking_message(self, messages: list[dict], thinking_budget: int) -> tuple[dict, dict]:
        with torch.inference_mode():
            inputs = self._encode_text(self._render_chat(messages, enable_thinking=True))
            kwargs = self._gen_kwargs(max_tokens=0, thinking_budget=thinking_budget, generation_config=GREEDY)
            budget = self._add_thinking_budget_processor(
                kwargs.setdefault("logits_processor", []), inputs["input_ids"].shape[1], thinking_budget,
            )
            output_ids = self.model.generate(**inputs, **kwargs)
        split = self._thinking_codec.split_generated(output_ids[0, inputs["input_ids"].shape[1]:])
        message = {"role": "assistant", "content": self._thinking_protocol.format_block(split.thinking_text)}
        record = GenerateResult("", thinking_tokens=split.thinking_tokens, forced_close=0 in budget.forced).thinking_record()
        return message, record

    async def _score_binary_after_shared_thinking(self, continued_messages: list[list[dict]]) -> tuple[np.ndarray, int]:
        return self._score_binary_chunks(continued_messages, enable_thinking=False)

    def _think_then_score(self, inputs: dict, thinking_budget: int = 0) -> torch.Tensor:
        """Generate thinking, capture logits right after the close boundary.

        The budget processor forces the boundary within the generation cap, so every
        row is captured. Returns [batch, vocab] logits predicting the first answer token.
        """
        batch = inputs["input_ids"].shape[0]
        prompt_width = inputs["input_ids"].shape[1]
        boundary_ids = self._thinking_protocol.answer_boundary_ids
        capture = _CapturePostThinkLogits(boundary_ids, batch, prompt_width=prompt_width)
        logits_processor = []
        self._add_thinking_budget_processor(logits_processor, prompt_width, thinking_budget)
        logits_processor.append(capture)

        self.model.generate(
            **inputs,
            # Full answer-boundary width + one step whose scores predict the answer.
            max_new_tokens=thinking_budget + len(boundary_ids) + 1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,
            logits_processor=logits_processor,
            stopping_criteria=[_AllCaptured(capture)],
        )
        return torch.stack([capture.captured[i] for i in range(batch)])

    async def generate(self, messages_list: list[list[dict]],
                       max_tokens: int = 1,
                       *, thinking_budget: int | None = None,
                       generation_config=GREEDY, response_json_schema=None,
                       batch_key=None) -> list[GenerateResult]:
        results = []
        for start in range(0, len(messages_list), self.batch_size):
            results.extend(self._generate_batch_sync(
                messages_list[start:start + self.batch_size], max_tokens, thinking_budget, generation_config,
            ))
        return results

    @torch.inference_mode()
    def _generate_batch_sync(self, messages_list, max_tokens, thinking_budget, generation_config):
        if not messages_list:
            return []
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        texts = [self._render_chat(m) for m in messages_list]
        # Left-pad so generated tokens align across sequences.
        if len(messages_list) > 1:
            self.tokenizer.padding_side = "left"
        inputs = self._encode_text(texts, padding=True)
        kwargs = self._gen_kwargs(max_tokens, thinking_budget, generation_config)
        self._add_presence_penalty_processor(
            kwargs, inputs["input_ids"].shape[1], generation_config,
        )
        budget = self._add_thinking_budget_processor(
            kwargs.setdefault("logits_processor", []), inputs["input_ids"].shape[1], thinking_budget,
        ) if self.enable_thinking else None
        output_ids = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return [
            self._decode_result(output_ids[j, prompt_len:], forced_close=budget is not None and j in budget.forced)
            for j in range(len(messages_list))
        ]

    def _render_chat(self, messages, *, enable_thinking: bool | None = None) -> str:
        enable_thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if enable_thinking:
            return self._thinking_codec.render_chat(messages, enable_thinking=True)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def _encode_text(self, text, **kwargs):
        return self.tokenizer(text, return_tensors="pt", **kwargs).to(self.model.device)
