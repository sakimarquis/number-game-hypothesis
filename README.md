# Hypothesis generation and updating in large language models

Official code repository for the [paper](https://arxiv.org/abs/2605.05851).

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

CPU is sufficient for hosted models and analysis. Choose one Torch extra: `cpu`, `mps`, `cu124`, `cu126`, `cu128`, or `cu130`.

The example below uses `gpt54-mini` through OpenAI. Set `OPENAI_API_KEY` in your environment before scoring; the Bayesian baseline requires no credentials. See [backends/README.md](backends/README.md) for other providers and local models.

```bash
# 1. Build shared Bayesian baselines and human reference analyses.
uv run run_number_game.py --model bayesian

# 2. Run all three probes on the smaller task and prompt subset.
uv run run_number_game.py --model gpt54-mini --run-profile mini --task Tenenbaum99 --domain 100

# 3. Convert generated hypotheses to predicates and analyze them.
uv run run_hypothesis_analysis.py --model gpt54-mini
```

Scoring makes model requests and automatically runs prediction analysis. Hypothesis analysis also updates shared predicate libraries; run it serially. This small example does not cover every aggregate comparison.

## Run and analyze experiments

Remove the task/domain selectors to cover both tasks and all configured domains:

```bash
uv run run_number_game.py --model gpt54-mini --run-profile mini
uv run run_hypothesis_analysis.py --model gpt54-mini
uv run -m analysis.summary
```

`mini` selects the prompt subset used by cross-model summaries; it keeps all stimulus sets and configured prefix lengths. Use `--run-profile full` for the entire prompt grid (the default). Repeat with other [model aliases](configs/model_aliases.yml) to compare models. Add `--thinking` to scoring and per-model analysis commands for thinking-mode results.

To refresh prediction analysis from existing caches:

```bash
uv run run_analysis.py --model gpt54-mini --include-per-condition
```

Omit `--include-per-condition` to rebuild only aggregate results. Use `--model bayesian` to refresh reference analyses, and rerun `run_hypothesis_analysis.py` for hypothesis results. Cross-model summaries read the resulting analysis CSVs. The [analysis guide](analysis/README.md) explains stages and figures.

### Select conditions and reuse caches

- **Selectors schedule scoring; they are not inspection commands.** Missing raw caches are computed. Existing caches are reused only when semantic identity, condition metadata, prefixes, and result shapes match. Incompatible or incomplete caches, including files without `cache_identity`, stop the run without replacement.
- Add `--refresh-raw-cache` to deliberately replace selected raw caches, or set `OVERWRITE: true`. Profiles, selectors, concurrency, and retry settings do not change semantic cache identity; `mini` and `full` share caches.
- LLM runs require compatible Bayesian caches and never rebuild them implicitly. Rebuild baselines after changing tasks, domains, or Bayesian settings. Aggregate analysis reads compatible caches across the configured grid, including conditions outside the latest scoring selection.

<details>
<summary>Advanced scoring options</summary>

Select a specific task, domain, prompt, and probe:

```bash
uv run run_number_game.py --model gpt54-mini --task Bigelow16 --domain 100 --prompt explicit_strong --eval-mode hypothesis_eval
```

Predict uses Yes/No token probabilities (`logits`) or repeated full-domain binary judgments (`text_trials`). Backends without logprobs default to `text_trials`. When new text-trial scoring is needed, run without `--allow-text-posterior` first: preflight prints a request/token estimate and stops before model requests. Repeat with the flag to proceed; complete matching caches need no opt-in.

`--posterior-method logits|text_trials` overrides the protocol; unsupported logits fail before scoring. See the [backend guide](backends/README.md) for thinking, batching, and recovery behavior.

</details>

## Configuration

| File | Controls |
| --- | --- |
| [configs/number_game.yml](configs/number_game.yml) | Tasks, domains, prompt grid, profiles, scoring, and thinking budgets. |
| [configs/model_aliases.yml](configs/model_aliases.yml) | Model IDs, result names, and backend bindings. |
| [configs/backend_registry.py](configs/backend_registry.py) | Provider credentials and request policies. |
| [configs/generate.yml](configs/generate.yml) | Model-family sampling settings. |

Dependencies live in [pyproject.toml](pyproject.toml); run `uv lock` after changing them. Contribution guidance is in [AGENTS.md](AGENTS.md).

## Results

Outputs go to `results/`, or `$DATA_DIR/number_game/results` when `DATA_DIR` is set:

```text
results/
  bayesian/{task}/d{domain}/               # shared baselines
  human/                                 # human reference analyses
  {model}/{task}/d{domain}/{prompt}/data/  # raw caches
  {model}/{task}/d{domain}/{prompt}/       # condition figures
  {model}/analysis/                       # analysis tables and caches
  {model}/                               # aggregate figures
  summary/                               # cross-model figures and tables
```

`{model}` is the configured result name: `gpt54-mini` writes to `GPT54_Mini/` (or `GPT54_Mini_thinking/`). `{prompt}` is `{h_space}_{sampling}`, such as `implicit_default`. Domain 100 aggregate outputs use the model's root figure/analysis directories; other domains add `d{domain}/`. Cross-domain tables stay in the root `analysis/` directory.

Start with `llm_panels.png` and `bayesian_panels.png` in a condition directory to compare generalization curves, including human judgments where available. The [figure guide](analysis/README.md#find-and-interpret-figures) covers evidence trajectories, hypothesis comparisons, and model summaries. Available outputs depend on the conditions run.

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
