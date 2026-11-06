"""HF chat-template thinking protocol resolution and output splitting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThinkingProtocol:
    start_ids: tuple[int, ...]
    end_ids: tuple[int, ...]
    start_text: str = ""
    end_text: str = ""
    prefill_text: str = ""
    starts_in_thinking: bool = False
    answer_prefix_ids: tuple[int, ...] = ()
    forbid_reentry: bool = False

    @property
    def answer_boundary_ids(self) -> tuple[int, ...]:
        return self.end_ids + self.answer_prefix_ids

    def format_block(self, thinking: str) -> str:
        """Normalize one thinking trace as a resumable assistant prefix."""
        if not self.start_text or not self.end_text:
            raise ValueError("Thinking block formatting requires text boundaries.")
        body = thinking.strip()
        if body.startswith(self.start_text):
            body = body[len(self.start_text):].lstrip()
        if body.endswith(self.end_text):
            body = body[:-len(self.end_text)].rstrip()
        separator = "" if self.start_text.endswith(("\n", "\r")) else "\n"
        return f"{self.start_text}{separator}{body}{self.end_text}\n"


@dataclass(frozen=True)
class ThinkingSplit:
    final_text: str
    thinking_text: str
    thinking_tokens: int


def text_thinking_protocol() -> ThinkingProtocol:
    """Return the `<think>...</think>` text protocol used by renderers without tokenizer access."""
    return ThinkingProtocol(start_ids=(), end_ids=(), start_text="<think>", end_text="</think>")


def find_subsequence(seq, pattern: tuple[int, ...]) -> list[int]:
    if not pattern or len(pattern) > len(seq):
        return []
    width = len(pattern)
    return [
        i for i in range(len(seq) - width + 1)
        if tuple(int(x) for x in seq[i:i + width]) == pattern
    ]


def suffix_matches(row, pattern: tuple[int, ...]) -> bool:
    return len(row) >= len(pattern) and tuple(int(x) for x in row[-len(pattern):]) == pattern


def _apply_chat_template(renderer, messages, enable_thinking: bool) -> str:
    return renderer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )


def _special_token_strings(tokenizer) -> list[str]:
    tokens = []
    for attr in ("all_special_tokens", "additional_special_tokens"):
        for token in getattr(tokenizer, attr, []) or []:
            if token not in tokens:
                tokens.append(token)
    if hasattr(tokenizer, "get_added_vocab"):
        for token in tokenizer.get_added_vocab():
            if token not in tokens:
                tokens.append(token)
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if decoder:
        for token in decoder.values():
            text = getattr(token, "content", str(token))
            if text not in tokens:
                tokens.append(text)
    return tokens


def _thinking_marker_candidates(tokenizer, rendered: str) -> list[str]:
    candidates = [
        token for token in _special_token_strings(tokenizer)
        if token and any(word in token.lower() for word in ("think", "thought"))
    ]
    for marker in _rendered_marker_candidates(rendered):
        if marker not in candidates:
            candidates.append(marker)
    return candidates


def _rendered_marker_candidates(rendered: str) -> list[str]:
    markers = []
    start = 0
    while True:
        left = rendered.find("<", start)
        if left < 0:
            return markers
        right = rendered.find(">", left + 1)
        if right < 0:
            return markers
        marker = rendered[left:right + 1]
        if any(word in marker.lower() for word in ("think", "thought")) and marker not in markers:
            markers.append(marker)
        start = right + 1


def _close_variants(open_marker: str) -> list[str]:
    variants = []
    if open_marker.startswith("<") and not open_marker.startswith("</"):
        variants.append("</" + open_marker[1:])
    if "start" in open_marker.lower():
        variants.append(open_marker.replace("start", "end").replace("START", "END"))
    return variants


def _token_ids(tokenizer, marker: str) -> tuple[int, ...]:
    ids = tuple(tokenizer.encode(marker, add_special_tokens=False))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if not ids or (unk_id is not None and marker != unk_token and all(int(x) == unk_id for x in ids)):
        return ()
    return ids


def _last_marker_in_render(markers: list[str], rendered: str) -> str:
    positions = [(rendered.rfind(marker), marker) for marker in markers if marker in rendered]
    if not positions:
        return ""
    return max(positions)[1]


def _is_gemma4(model_id: str, renderer) -> bool:
    normalized = model_id.lower()
    template = str(getattr(renderer, "chat_template", "") or "")
    return "gemma-4" in normalized or "gemma4" in normalized or "<|channel>thought" in template


class Gemma4ChannelProtocol:
    """Gemma 4's explicit thought-channel protocol and tokenizer-backed HF validation."""

    _START_TEXT = "<|channel>thought\n"
    _END_TEXT = "<channel|>"
    _TEMPLATE_TRIGGER = "<|think|>"

    @classmethod
    def api_protocol(cls) -> ThinkingProtocol:
        """Return text boundaries for a server configured with the Gemma/Harmony parser."""
        return ThinkingProtocol(
            start_ids=(),
            end_ids=(),
            start_text=cls._START_TEXT,
            end_text=cls._END_TEXT,
            starts_in_thinking=True,
        )

    @classmethod
    def resolve_hf(
        cls,
        renderer,
        tokenizer,
        model_id: str,
        enabled: str,
        disabled: str,
    ) -> ThinkingProtocol:
        template = str(getattr(renderer, "chat_template", "") or getattr(tokenizer, "chat_template", "") or "")
        source = "\n".join([enabled, disabled, template])
        start_ids = _token_ids(tokenizer, cls._START_TEXT)
        end_ids = _token_ids(tokenizer, cls._END_TEXT)
        if not start_ids:
            raise ValueError("Gemma 4 thought-channel open marker has no reliable tokenizer encoding.")
        if not end_ids:
            raise ValueError("Gemma 4 thought-channel close marker has no reliable tokenizer encoding.")

        open_stem = cls._START_TEXT.rstrip("\n")
        last_open = enabled.rfind(open_stem)
        if last_open >= 0:
            if cls._END_TEXT in enabled[last_open:]:
                raise ValueError(
                    f"{model_id} Gemma 4 chat template closes thinking inside the generation prompt; "
                    "there is no generated close boundary for answer-logit scoring."
                )
            prefill_text = ""
        elif cls._TEMPLATE_TRIGGER in enabled:
            prefill_text = cls._START_TEXT
        else:
            close_pos = source.find(cls._END_TEXT)
            open_source = source[:close_pos] if close_pos >= 0 else source
            if open_stem not in open_source:
                raise ValueError(f"{model_id} Gemma 4 chat template exposes no thought-channel open boundary.")
            prefill_text = cls._START_TEXT

        return ThinkingProtocol(
            start_ids=start_ids,
            end_ids=end_ids,
            start_text=cls._START_TEXT,
            end_text=cls._END_TEXT,
            prefill_text=prefill_text,
            starts_in_thinking=True,
        )


def resolve_api_thinking_protocol(name: str | None) -> ThinkingProtocol | None:
    """Resolve a configured server-side protocol; API clients cannot inspect its tokenizer.

    `text` servers are forced closed client-side in two stages; `channel` servers
    (Gemma 4 reasoning parser) emit thought and answer in one stream.
    """
    if name is None:
        return None
    if name == "text":
        return text_thinking_protocol()
    if name == "channel":
        return Gemma4ChannelProtocol.api_protocol()
    raise ValueError(f"Unknown API thinking protocol {name!r}. Expected 'text' or 'channel'.")


class HfThinkingCodec:
    def __init__(self, renderer, model_id: str):
        self.renderer = renderer
        self.tokenizer = getattr(renderer, "tokenizer", renderer)
        self.model_id = model_id
        self.protocol = self._resolve_protocol()

    def render_chat(self, messages, enable_thinking: bool) -> str:
        rendered = _apply_chat_template(self.renderer, messages, enable_thinking)
        if enable_thinking:
            rendered += self.protocol.prefill_text
        return rendered

    def split_generated(self, generated_ids) -> ThinkingSplit:
        ids = [int(x) for x in generated_ids.tolist()]
        positions = find_subsequence(ids, self.protocol.end_ids)
        if not positions:
            raise RuntimeError(
                "HF thinking generation did not produce the tokenizer-owned close boundary; "
                "answer parsing is unsupported for this output."
            )
        end = positions[0]
        return ThinkingSplit(
            final_text=self.tokenizer.decode(ids[end + len(self.protocol.end_ids):], skip_special_tokens=True).strip(),
            thinking_text=self.tokenizer.decode(ids[:end], skip_special_tokens=True).strip(),
            thinking_tokens=end,
        )

    def _resolve_protocol(self) -> ThinkingProtocol:
        probe = [{"role": "user", "content": "Return Yes or No."}]
        try:
            enabled = _apply_chat_template(self.renderer, probe, True)
            disabled = _apply_chat_template(self.renderer, probe, False)
        except TypeError as exc:
            raise ValueError(f"{self.model_id} chat template does not accept enable_thinking") from exc

        tokenizer = self.tokenizer
        if _is_gemma4(self.model_id, self.renderer):
            return Gemma4ChannelProtocol.resolve_hf(
                self.renderer, tokenizer, self.model_id, enabled, disabled,
            )

        # The rendered thinking prompt identifies the opening channel; tokenizer
        # metadata supplies the matching close marker when answer logits are safe.
        markers = _thinking_marker_candidates(tokenizer, enabled)
        template = str(getattr(self.renderer, "chat_template", "") or getattr(tokenizer, "chat_template", "") or "")
        close_evidence = "\n".join([disabled, template])
        special_tokens = set(_special_token_strings(tokenizer))
        open_markers = [
            marker for marker in markers
            if not marker.startswith("</") and "end" not in marker.lower()
        ]
        open_marker = _last_marker_in_render(open_markers, enabled)
        close_candidates = [
            marker for marker in markers
            if marker != open_marker and (marker.startswith("</") or "end" in marker.lower())
        ]
        for marker in _close_variants(open_marker):
            if marker not in close_candidates and (marker in special_tokens or marker in close_evidence):
                close_candidates.append(marker)

        start_ids = _token_ids(tokenizer, open_marker) if open_marker else ()
        close_with_ids = [
            (marker, ids) for marker in close_candidates
            if (ids := _token_ids(tokenizer, marker))
        ]
        if close_with_ids:
            close_marker, end_ids = close_with_ids[0]
            if open_marker:
                last_open = enabled.rfind(open_marker)
                if last_open >= 0 and close_marker in enabled[last_open:]:
                    raise ValueError(
                        f"{self.model_id} chat template closes thinking inside the generation prompt; "
                        "there is no generated close boundary for answer-logit scoring."
                    )
            answer_prefix_text = disabled.rsplit(close_marker, 1)[1] if close_marker in disabled else ""
            answer_prefix_ids = _token_ids(tokenizer, answer_prefix_text) if answer_prefix_text else ()
            return ThinkingProtocol(
                start_ids, end_ids, open_marker, close_marker,
                answer_prefix_ids=answer_prefix_ids,
                forbid_reentry=len(start_ids) == 1 and len(end_ids) == 1,
            )

        thinking_tokens = ", ".join(markers) or "none"
        raise ValueError(
            f"{self.model_id} chat template exposes no tokenizer-owned thinking close boundary. "
            f"Found thinking markers: {thinking_tokens}. HF thinking is unsupported for this model."
        )
