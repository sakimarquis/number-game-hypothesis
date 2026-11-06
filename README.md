# Hypothesis generation and updating in large language models

Official code repository for the [Hypothesis generation and updating in large language models](https://arxiv.org/abs/2605.05851).

How do LLMs generate hypotheses from a few examples, update them with new evidence, and generalize beyond what they have seen? We study these questions through the **number game**: given accepted numbers such as `{16, 8, 2, 64}`, infer the hidden rule and predict which other numbers it accepts.

We compare LLMs with Bayesian inference and human judgments using three complementary probes:

| Probe | What the model does |
| --- | --- |
| **Predict** (`posterior`) | Predicts whether each number belongs to the hidden concept. |
| **Eval** (`hypothesis_eval`) | Assigns probabilities to supplied candidate hypotheses. |
| **Generate** (`hypothesis_generation`) | Proposes hypotheses freely, with confidence values. |

Examples are revealed one at a time to track how predictions and hypotheses change. Experiments use `Tenenbaum99` (8 stimulus sets; domains 1–100 and 1–200) and `Bigelow16` (255 sets; domain 1–100), with human judgments in domain 100. See [tasks/README.md](tasks/README.md) for task sources, prompts, and evaluation semantics.

## Quick start

Requires **Python 3.11+** and **uv**. Run commands from the repository root:

```bash
uv sync --extra cpu
# For a local GPU with CUDA 12.8, use instead:
# uv sync --extra cu128
```

CPU is sufficient for hosted models and analysis. Choose one Torch extra: `cpu`, `mps`, `cu124`, `cu126`, `cu128`, or `cu130`. The commands below use `uv run`; an activated `.venv` can use `python` directly.

The example below uses `gpt54-mini` through OpenAI. Set `OPENAI_API_KEY` in your environment before scoring; the Bayesian baseline requires no credentials. See [backends/README.md](backends/README.md) for other providers and local models.

```bash
# 1. Run all three probes on the smaller task and prompt subset.
uv run run_number_game.py --model gpt54-mini --run-profile mini --task Tenenbaum99 --domain 100

# 2. Convert generated hypotheses to predicates and analyze them.
uv run run_hypothesis_analysis.py --model gpt54-mini
```

Successful scoring runs automatically start prediction analysis, which builds any missing Bayesian baselines on first use. Hypothesis analysis also updates shared predicate libraries; run it serially. This small example does not cover every aggregate comparison.

## Run an experiment

Remove the task/domain selectors to cover both tasks and all configured domains:

```bash
uv run run_number_game.py --model gpt54-mini --run-profile mini
uv run run_hypothesis_analysis.py --model gpt54-mini
uv run -m analysis.summary
```

`mini` selects only the cross-model summary conditions (`SUMMARY_PROMPTS` and `SUMMARY_MODE_CONDITIONS` in [configs/settings.py](configs/settings.py)); it keeps all stimulus sets and configured prefix lengths. Use `--run-profile full` for the entire configured grid (the default); both profiles share raw caches. Repeat with other [model aliases](configs/model_aliases.yml) to compare models; omitting `--model` uses the configured model. Add `--thinking` to scoring and per-model analysis commands for thinking-mode results.

To build or refresh the shared Bayesian baselines, human reference analyses, and their figures explicitly:

```bash
uv run run_number_game.py --model bayesian
```

To refresh prediction analysis from existing caches:

```bash
uv run run_analysis.py --model gpt54-mini --include-per-condition
```

Omit `--include-per-condition` to rebuild only aggregate results. Rerun `run_hypothesis_analysis.py` for hypothesis results. Cross-model summaries read the resulting analysis CSVs. The [analysis guide](analysis/README.md) explains what each command refreshes, its inputs and outputs, and figure meanings.

### Select conditions and reuse caches

The grid varies task, domain, evaluation mode, `h_space`, and `sampling`; their meanings are defined in [tasks/README.md](tasks/README.md).

- **Selectors schedule scoring; they are not inspection commands.** Missing raw caches are computed. Existing caches are reused only when semantic identity, condition metadata, prefixes, and result shapes match. Incompatible or incomplete caches, including files without `cache_identity`, stop the run without replacement.
- Add `--refresh-raw-cache` to deliberately replace selected raw caches, or set `OVERWRITE: true`. Profiles, selectors, concurrency, and retry settings do not change semantic cache identity; `mini` and `full` share caches.
- LLM scoring never reads Bayesian caches. Analysis loads compatible Bayesian caches and builds missing or stale ones (after changing tasks, domains, or Bayesian settings) before its stages run. Aggregate analysis reads compatible caches across the configured grid, including conditions outside the latest scoring selection.

<details>
<summary>Advanced scoring options</summary>

Select a specific task, domain, prompt, and probe:

```bash
uv run run_number_game.py --model gpt54-mini --task Bigelow16 --domain 100 --prompt explicit_strong --eval-mode hypothesis_eval
```

Predict uses Yes/No token probabilities (`logits`) or repeated full-domain binary judgments (`text_trials`). Backends without logprobs default to `text_trials`. When new text-trial scoring is needed, run without `--allow-text-posterior` first: preflight prints a request/token estimate and stops before model requests. Repeat with the flag to proceed; complete matching caches need no opt-in.

`--posterior-method logits|text_trials` overrides the protocol; unsupported logits fail before scoring. See the [backend guide](backends/README.md) for thinking, batching, and recovery behavior.

</details>

## Configuration and outputs

| File | Controls |
| --- | --- |
| [configs/number_game.yml](configs/number_game.yml) | Experiment grid, profiles, scoring, concurrency, and thinking budgets. |
| [configs/model_aliases.yml](configs/model_aliases.yml) | Model IDs, result names, summary order and colors, default backends, model-specific bindings, and per-model thinking generation budgets. |
| [configs/backend_registry.py](configs/backend_registry.py) | Backend names, credentials, and shared provider policy. |
| [configs/generate.yml](configs/generate.yml) | Model-family sampling settings for hypothesis generation. |

Dependencies live in [pyproject.toml](pyproject.toml); run `uv lock` after changing them.

Outputs go to `results/`, or `$DATA_DIR/number_game/results` when `DATA_DIR` is set:

```text
results/
  bayesian/{task}/d{domain}/               # shared baselines
  human/                                 # human reference analyses
  {model}/{task}/d{domain}/{prompt}/data/  # raw condition caches
  {model}/{task}/d{domain}/{prompt}/       # condition figures (off by default)
  {model}/analysis/                       # analysis tables and caches
  {model}/                               # aggregate figures (off by default)
  summary/                               # cross-model figures and tables
  summary/all/                           # same summaries over every registered summary model
```

`{model}` is the configured result name: `gpt54-mini` writes to `GPT54_Mini/` (or `GPT54_Mini_thinking/`). `{prompt}` is `{h_space}_{sampling}`, such as `implicit_default`. Domain 100 aggregate outputs use the model's root analysis/figure directories; other domains use `d{domain}/` underneath them. Cross-domain tables stay in the model's root `analysis/`.

Per-model figures are skipped unless `PLOT_MODEL_FIGURES` in [configs/settings.py](configs/settings.py) is enabled; enable it and rerun analysis to rebuild them from the model's caches. With it enabled, start with `llm_panels.png` and `bayesian_panels.png` in a condition directory to compare generalization curves, including human judgments where available. The [figure guide](analysis/README.md#find-and-interpret-figures) covers evidence trajectories, hypothesis comparisons, and model summaries. Available outputs depend on the conditions run.

## Code layout

| Area | Responsibility |
| --- | --- |
| [tasks/](tasks/README.md) | Task data, prompts, and hypothesis spaces. |
| [backends/](backends/README.md) | Model execution, batching, thinking, and recovery. |
| `scoring/` | Evaluation protocols and raw prefix-cache production. |
| [analysis/](analysis/README.md) | Cache reconstruction, metrics, fitting, plots, and stage dispatch. |
| `configs/` | Typed configuration, model resolution, grid construction, and paths. |

## Citation

If you use this code, please cite:

```bibtex
@misc{xiong2026hypothesis,
  title         = {Hypothesis generation and updating in large language models},
  author        = {Hua-Dong Xiong},
  year          = {2026},
  eprint        = {2605.05851},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.05851}
}
```

## Data and license

The number-game tasks build on [Tenenbaum (1999)](https://dspace.mit.edu/entities/publication/318bda02-f396-45c5-ab06-9f16238269c5) and [Bigelow & Piantadosi (2016)](https://doi.org/10.5334/jopd.19). See [dataset/README.txt](dataset/README.txt) for the bundled human-data fields. Source code is licensed under [GPL-3.0](LICENSE).
