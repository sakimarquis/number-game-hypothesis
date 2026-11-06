"""Model registry, experiment grid, and project constants."""

import os
from dataclasses import dataclass
from itertools import product

from configs.backend_registry import NO_LOGPROB_BACKENDS, TOGGLE_THINKING_BACKENDS, reasoning_enabled, top_logprobs
from utils import load_yaml

RESULTS_ROOT = os.path.join(os.environ["DATA_DIR"], "number_game", "results") if "DATA_DIR" in os.environ else "results"


@dataclass(frozen=True)
class BackendBinding:
    """Per-(model, backend) kwargs from model_aliases.yml."""
    batch_size: int = 8
    thinking_batch_size: int | None = None
    api_key_env: str | None = None
    execution: str = "online"
    batch_priority: int = 0
    renderer: str | None = None
    thinking_renderer: str | None = None
    tokenizer_trust_remote_code: bool = False
    thinking_protocol: str | None = None
    reasoning_effort: str | None = None
    supports_json_schema: bool = False

    def effective_batch_size(self, *, enable_thinking: bool) -> int:
        if enable_thinking and self.thinking_batch_size is not None:
            return self.thinking_batch_size
        return self.batch_size


@dataclass
class ModelInfo:
    id: str
    display: str
    default_backend: str
    backends: dict[str, BackendBinding]
    summary_color: str | None = None
    # HYPOTHESIS_GENERATION token fields overridden for this model's thinking runs.
    thinking_generation_tokens: dict[str, int] | None = None

    def binding(self, backend: str) -> BackendBinding:
        if backend not in self.backends:
            raise KeyError(
                f"Model {self.id!r} has no binding for backend {backend!r}. "
                f"Available: {sorted(self.backends)}"
            )
        return self.backends[backend]


@dataclass(frozen=True)
class TaskGrid:
    """Per-task factor levels for the factorial grid."""
    eval_modes: list[str]
    h_spaces: list[str]
    samplings: list[str]
    domains: list[int]
    featured_stimuli: list[str] | None = None


@dataclass(frozen=True)
class ScoringPolicy:
    """Scoring prefix limit and parse-retry budget."""
    max_examples: int
    max_retries: int


@dataclass(frozen=True)
class BayesianConfig:
    """Bayesian baseline hyperparameters."""
    lam: float
    sigma: float


@dataclass(frozen=True)
class PosteriorConfig:
    method: str
    n_trials: int
    temperature: float
    thinking_mode: str
    thinking_budget: int

    def __post_init__(self) -> None:
        if self.method not in {"logits", "text_trials"}:
            raise ValueError(
                f"Unknown posterior method {self.method!r}; expected 'logits' or 'text_trials'"
            )
        if self.n_trials < 1:
            raise ValueError("POSTERIOR.n_trials must be positive")
        if self.thinking_mode not in {"shared_prefix", "per_target"}:
            raise ValueError(f"Unknown POSTERIOR.thinking_mode: {self.thinking_mode!r}")


@dataclass(frozen=True)
class HypothesisEvalConfig:
    max_tokens: int
    thinking_budget: int


@dataclass(frozen=True)
class GenerationSamplingConfig:
    """Decoder sampling parameters; backends send only the fields that are set."""
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None


GREEDY = GenerationSamplingConfig(temperature=0.0)


@dataclass(frozen=True)
class HypothesisGenerationConfig:
    max_tokens: int
    n_hypotheses: int
    thinking_budget: int
    sampling: GenerationSamplingConfig


@dataclass(frozen=True)
class BackendConfig:
    """Resolved backend kwargs for the chosen (model, backend) pair.

    `enable_thinking` is the chat-template/provider thinking toggle; `thinking`
    is whether the run's results are thinking results (toggle or reasoning effort).
    """
    backend: str
    binding: BackendBinding
    enable_thinking: bool
    reasoning_effort: str | None
    top_logprobs: int
    result_key: str
    model_info: ModelInfo

    @property
    def reasoning(self) -> bool:
        return reasoning_enabled(self.reasoning_effort)

    @property
    def thinking(self) -> bool:
        return self.enable_thinking or self.reasoning

    @property
    def supports_logprobs(self) -> bool:
        # Hosted reasoning endpoints reject logprobs.
        return self.backend not in NO_LOGPROB_BACKENDS and not self.reasoning


ModeConfig = PosteriorConfig | HypothesisEvalConfig | HypothesisGenerationConfig
RUN_PROFILES = ("full", "mini")
EVAL_MODES = ("posterior", "hypothesis_eval", "hypothesis_generation")
# Result-key suffix of runs whose results are thinking results.
THINKING_SUFFIX = "_thinking"

# hypothesis_eval rates a supplied candidate list (explicit); hypothesis_generation
# measures free proposal (implicit). Task-level h_spaces sweep only posterior.
HYPOTHESIS_H_SPACE = {"hypothesis_eval": "explicit", "hypothesis_generation": "implicit"}

# Cross-model summaries compare these posterior prompts (h_space, sampling) and
# these per-mode conditions (eval_mode, h_space, sampling); the mini run profile
# scores exactly their union.
SUMMARY_PROMPTS = (("implicit", "default"), ("explicit", "default"), ("implicit", "strong"), ("implicit", "weak"))
SUMMARY_MODE_CONDITIONS = (
    ("posterior", "implicit", "default"),
    *((eval_mode, h_space, "default") for eval_mode, h_space in HYPOTHESIS_H_SPACE.items()),
)
MINI_CONDITIONS = frozenset({("posterior", *prompt) for prompt in SUMMARY_PROMPTS} | set(SUMMARY_MODE_CONDITIONS))


def _load_models() -> dict[str, ModelInfo]:
    """Load model registry from model_aliases.yml."""
    raw = load_yaml(os.path.join(os.path.dirname(__file__), "model_aliases.yml"))
    return {
        alias: ModelInfo(
            id=entry["id"],
            display=entry["display"],
            default_backend=entry["default_backend"],
            backends={name: BackendBinding(**(kwargs or {})) for name, kwargs in entry["backends"].items()},
            summary_color=entry.get("summary_color"),
            thinking_generation_tokens=entry.get("thinking_generation_tokens"),
        )
        for alias, entry in raw.items()
    }


# Alias → ModelInfo (loaded once at import time)
MODELS: dict[str, ModelInfo] = _load_models()

# Case-insensitive alias → canonical alias key
_ALIAS_MAP: dict[str, str] = {}
for _alias, _info in MODELS.items():
    for _key in [_alias, _info.id, _info.display]:
        _ALIAS_MAP[_key.lower()] = _alias


# Result keys of the model-agnostic reference producers; `--model bayesian` builds both.
BAYESIAN = "bayesian"
HUMAN = "human"

# Draw figures under per-LLM result trees (results/{model}/...). They are rebuildable from that model's caches; summary/, bayesian/, and human/ figures are always drawn.
PLOT_MODEL_FIGURES = False


def is_baseline_run(model_name: str | None) -> bool:
    """True when the selected model is the Bayesian/human baseline sentinel, not an LLM."""
    return model_name is not None and model_name.lower() == BAYESIAN


def resolve_model(name: str) -> ModelInfo:
    """Resolve any alias (short name, model ID, display name) to ModelInfo, case-insensitively."""
    key = _ALIAS_MAP.get(name.lower())
    if key is None:
        raise KeyError(
            f"Unknown model '{name}'. Available aliases: {', '.join(MODELS)}"
        )
    return MODELS[key]


def resolve_backend_config(
    model_info: ModelInfo,
    backend: str | None,
    thinking_settings: dict,
    thinking: bool = False,
) -> BackendConfig:
    """Resolve runtime knobs for one selected (model, backend) pair; `thinking` is the CLI flag."""
    backend = backend or model_info.default_backend
    binding = model_info.binding(backend)

    enable_thinking = backend in TOGGLE_THINKING_BACKENDS and thinking
    # OpenAI reasoning is selected at runtime. OpenRouter bindings may instead pin
    # a provider-normalized effort for models whose reasoning mode is mandatory.
    if backend == "openai":
        reasoning_effort = thinking_settings.get("reasoning_effort", "none") if thinking else "none"
    elif backend == "openrouter":
        reasoning_effort = binding.reasoning_effort
    else:
        reasoning_effort = None
    # Only runtime-selected thinking changes the result namespace; pinned
    # OpenRouter efforts are part of the model binding itself.
    runtime_thinking = enable_thinking or (backend == "openai" and reasoning_enabled(reasoning_effort))

    return BackendConfig(
        backend=backend,
        binding=binding,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        top_logprobs=top_logprobs(backend),
        result_key=model_info.display + (THINKING_SUFFIX if runtime_thinking else ""),
        model_info=model_info,
    )


def load_generation_sampling(model_id: str, enable_thinking: bool) -> GenerationSamplingConfig:
    """Resolve hypothesis-generation sampling by model family and thinking mode."""
    raw = load_yaml(os.path.join(os.path.dirname(__file__), "generate.yml"))
    normalized_id = model_id.casefold()
    parameters = dict(raw["default"])
    for family in raw["families"].values():
        if any(fragment.casefold() in normalized_id for fragment in family["model_id_contains"]):
            mode = "thinking" if enable_thinking else "non_thinking"
            parameters.update(family.get("all", family.get(mode)))
            break
    return GenerationSamplingConfig(**parameters)


@dataclass(frozen=True)
class ExperimentCondition:
    """One task/domain/eval-mode/prompt combination; the model selects its result root.

    Conditions with the same prompt_group share one result directory.
    """
    task: str       # "Tenenbaum99" | "Bigelow16"
    eval_mode: str  # "posterior" | "hypothesis_eval" | "hypothesis_generation"
    h_space: str    # "implicit" | "explicit"
    sampling: str   # "default" | "strong" | "weak" | "pedagogical" | "adversarial"
    domain: int     # 100, 200
    mode_config: ModeConfig | None = None

    @property
    def context_key(self) -> tuple[str, int]:
        """Groups conditions sharing one task/domain stimulus set and Bayesian baseline."""
        return self.task, self.domain

    @property
    def prompt_group(self) -> tuple[str, int, str, str]:
        """Groups conditions sharing one task/domain/prompt directory."""
        return self.task, self.domain, self.h_space, self.sampling

    @property
    def prompt(self) -> str:
        """Prompt condition name, `{h_space}_{sampling}`."""
        return f"{self.h_space}_{self.sampling}"

    def metadata(self) -> dict:
        """Condition fields recorded in raw caches and identities."""
        return {
            "task": self.task, "eval_mode": self.eval_mode, "h_space": self.h_space,
            "sampling": self.sampling, "domain": self.domain,
        }


def h_space_levels_for_eval(eval_mode: str, configured_h_spaces: list[str]) -> list[str]:
    """Return the h-space levels owned by one eval mode."""
    return list(configured_h_spaces) if eval_mode == "posterior" else [HYPOTHESIS_H_SPACE[eval_mode]]


@dataclass(frozen=True)
class Experiment:
    """Typed bundle parsed from configs/number_game.yml."""
    tasks: dict[str, TaskGrid]
    scoring: ScoringPolicy
    bayesian: BayesianConfig
    posterior: PosteriorConfig
    hypothesis_eval: HypothesisEvalConfig
    hypothesis_generation: HypothesisGenerationConfig
    backend: BackendConfig
    model: str
    seed: int
    overwrite: bool
    concurrency: int
    run_profile: str = "full"  # one of RUN_PROFILES


def make_grid(experiment: Experiment) -> list[ExperimentCondition]:
    """Build the experiment grid from the per-task factors in `experiment`.

    To grid a modified task set, pass a replace()'d experiment.
    """
    conditions = []
    for task_name, factors in experiment.tasks.items():
        for eval_mode in factors.eval_modes:
            mode_config = getattr(experiment, eval_mode)
            conditions.extend(
                ExperimentCondition(
                    task=task_name, eval_mode=eval_mode, h_space=h, sampling=s, domain=d,
                    mode_config=mode_config,
                )
                for h, s, d in product(
                    h_space_levels_for_eval(eval_mode, factors.h_spaces), factors.samplings, factors.domains
                )
            )
    return conditions
