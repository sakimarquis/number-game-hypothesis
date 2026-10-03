# Analysis

Analysis compares model predictions and stated hypotheses with Bayesian and human references. It reconstructs stimulus trajectories from raw prefix caches, computes condition and aggregate metrics, and writes tables used by cross-model summaries. Experiment terminology is defined in [tasks/README.md](../tasks/README.md).

## Run analysis

Run these from the repository root after generating compatible Bayesian baselines:

```powershell
uv run run_analysis.py --model gpt54-mini
uv run run_analysis.py --model gpt54-mini --include-per-condition
uv run run_hypothesis_analysis.py --model gpt54-mini
uv run -m analysis.summary
```

`run_analysis.py` refreshes default aggregate stages; `--include-per-condition` also rebuilds posterior condition figures and derived caches. Successful LLM scoring hands off to this command in a fresh process, keeping analysis imports out of scoring. `--model bayesian` refreshes baseline analyses. Add `--thinking` to model analysis commands for thinking results.

`run_hypothesis_analysis.py` structures free generations, runs hypothesis condition and aggregate stages, and refreshes hypothesis summaries. `python -m analysis.summary` builds default cross-model summaries from analysis CSVs; it does not infer missing metrics from raw model outputs. If a summary is missing data, refresh the producing model analyses first.

## Hypothesis structuring

[hypothesis_structuring.py](hypothesis_structuring.py) parses precise descriptions into support sets over `1..domain`. It reuses a predicate with identical finite-domain support or adds one to `library/library_d{domain}.py`. Predicate libraries and `hypothesis_generation_structured.json` are published together transactionally. Unresolved descriptions remain distinguishable from executable matches.

Rerunning structuring reparses the raw descriptions. Existing indices retain their executable support because other caches and human annotations can reference them. A corrected description can therefore move to another existing index or append a new predicate. Library labels alone are not semantic ground truth: audit the original description against its complete support over `1..domain`.

The parser retains numeric-list operands and supported negation, digit properties, bounded forms, and explicit Boolean grouping. Unsupported branches and conflicting explicit equivalences remain unresolved. Examples following a complete quoted rule do not supply extra rule conditions. Bounds or parity stated for an attribute, such as a root, a digit position, or a count of digits or divisors, apply to that attribute or leave the description unresolved; they never become bounds or parity of the number itself. Run structuring writers serially; the transaction does not isolate concurrent processes.

An understood branch with empty support can participate in a Boolean expression; an unsupported branch cannot be treated as empty. A complete hypothesis with empty support remains `other`. Explicit first-N sequence lists and stated totals must agree with the sequence before acceptance. Prime-factor counts distinguish explicit multiplicity from distinct factors; descriptions asserting both conventions remain unresolved.

Hypothesis-generation analysis requires this structured artifact and never reads raw free text as its analysis input. The projection logic in [hypothesis_space.py](hypothesis_space.py) puts predictions, supplied-hypothesis ratings, and structured generations into a shared predictive space while tracking matched and unmatched mass.

## Stage ownership and tables

[pipeline.py](pipeline.py) owns dispatch. Default stages run per domain through `DEFAULT_STAGES`; hypothesis stages use `HYPOTHESIS_STAGES`. Cross-domain extension stages run once per model through `DEFAULT_MODEL_STAGES` and `HYPOTHESIS_MODEL_STAGES`. Condition plotting is a separate pass, not an aggregate stage.

| Default stage | Purpose | Main CSVs |
| --- | --- | --- |
| `divergence` | Compare final posterior predictions with references. | `divergence_metrics.csv`, `divergence_human_metrics.csv`. |
| `alpha_beta` | Fit prior and size-principle weights. | `alpha_beta_fits.csv`. |
| `prompt` | Compare prompt sensitivity and prediction width. | `prompt_sensitivity.csv`, `prompt_effective_support.csv`. |
| `trajectory` | Track predictions and fits as evidence grows. | `trajectory_metrics.csv`, `trajectory_prefix_metrics.csv`. |
| `rt_correlation` | Relate available model metrics to human response times. | `rt_correlation_*.csv`. |
| `extension` (cross-domain) | Measure d200 extension and d100-to-d200 change. | `extension_mass.csv`, `extension_rule_discrimination.csv`, `extension_domain_shift.csv`. |

| Hypothesis stage | Purpose | Main CSVs |
| --- | --- | --- |
| `cross_mode` | Compare modes in shared predictive space. | `cross_mode_projection_*.csv`. |
| `nested_pair` | Compare nested rule-pair behavior. | `nested_pair_*.csv`. |
| `alpha_beta_mode` | Fit projected predictions for each evaluation mode. | `alpha_beta_mode_fits.csv`. |
| `generation_narrowing` | Count unique structured hypotheses over evidence. | `generation_narrowing_counts.csv`. |
| `hypothesis_divergence` | Compare projected hypothesis predictions with references. | `hypothesis_divergence_metrics.csv`. |
| `hypothesis_generation` | Measure hypothesis types and confidence. | `hypothesis_generation_*.csv`. |
| `hypothesis_compare` | Measure top-1 properties, entropy, and cross-mode distance. | `hypothesis_compare_*_metrics.csv`, `hypothesis_compare_*_summary.csv`. |
| `hypothesis_fit` | Fit projected hypothesis-interface predictions. | `hypothesis_fit_metrics.csv`. |
| `hypothesis_extension` (cross-domain) | Extension diagnostics for eval and generation. | `hypothesis_extension_mass.csv`, `hypothesis_extension_rule_discrimination.csv`, `hypothesis_extension_domain_shift.csv`. |

Outputs depend on available compatible inputs; a stage need not emit every table for every model.

## Data and metric contracts

Raw caches use a flat `prefix_cache`; [cache.py](cache.py)'s `reconstruct_trajectory()` rebuilds per-stimulus sequences. Condition identity and raw-cache replacement rules are documented in the [root README](../README.md#select-conditions-and-reuse-caches). Analysis CSVs use `stimulus` and `stimulus_kind` for stimulus identifiers; pooled stimulus-kind rows use `all`.

`SCORING.trajectory_prefix_horizon` selects stimulus sets with at least that many examples and truncates trajectories to that horizon (capped by the longest stimulus). It controls analysis, independently of the scoring limit `SCORING.max_examples`.

Aggregate alpha-beta fits pool available task stimuli at fixed model, domain, hypothesis framing, sampling, and fit scope. Scopes are `n=1..4` and `full` when available; `task` records the pooled task group. `alpha_beta_mode_fits.csv` adds evaluation mode after projection. Valid, exhausted, and usable counts distinguish source availability from finite fit targets; only complete final-prefix projections contribute to `full`.

| Metric | Interpretation |
| --- | --- |
| `alpha`, `beta` | Fitted prior and size-principle weights; Bayesian reference is `(1, 1)`. |
| `alpha_beta_ratio` | Ratio after flooring alpha and beta at 0.05; figures display its natural logarithm. |
| KL | Direction is Bayesian to model where labeled `kl_bayes`. |
| `jsd` / `*_jsd` | Base-2 Jensen-Shannon **distance**, the square root of divergence. |
| Effective support | Number of targets with `P(yes) > 0.5`. |
| Top-1 support fraction | Fraction of the domain accepted by the highest-confidence hypothesis. |
| Top-1 evidence accuracy | Whether the hypothesis accepts all shown examples (0 or 1); summaries average these indicators. |

Summary tables keep model identity in `model`, data provenance in `source`, and evaluation-mode display names in `mode_label`. Cross-model mode trajectories count observations with `n_models`; model trajectories use `n_stimuli`. Summary labels `Predict`, `Eval`, and `Generate` mean `posterior`, `hypothesis_eval`, and `hypothesis_generation`. These modes are distinct from producers such as `llm`, `human`, and `bayesian`. Entropy units depend on the metric: posterior summary entropy uses nats; hypothesis comparison entropy uses bits.

## Find and interpret figures

Paths below are relative to `results/` (see the root README for `DATA_DIR`). `{prompt}` means `{h_space}_{sampling}`. Per-domain model CSVs live in `{model}/analysis/`, figures in `{model}/`; domain 100 uses these directories directly, while other domains use a `d{domain}/` child. Cross-domain tables stay in root `analysis/`.

### Baselines and individual conditions

Bayesian decompositions live under `bayesian/{task}/d{domain}/decompositions/`; human fit comparisons live under `human/{task}/d{domain}/alpha_beta/`. Model condition figures live under `{model}/{task}/d{domain}/{prompt}/`. Large tasks restrict per-stimulus plots using `featured_stimuli`; this does not restrict scoring.

| Figure family | What it shows / inputs |
| --- | --- |
| `bayesian_panels.png`, `llm_panels.png`, `histogram/` | Final membership curves, with human judgments where available. |
| `alpha_beta/ablation_*.png` | Final curves against prior-only, likelihood-only, and joint fits. |
| `trajectory_*.png` | Prefix-wise KL, entropy, support, and alpha-beta fits. |
| `hypothesis_comparison.png` | Supplied-hypothesis probabilities against the Bayesian reference. |
| `hypothesis_generation/`, `hypothesis_generation_types.png`, `hypothesis_generation_exemplars.png` | Structured generation confidence, type composition, and selected examples. |

### Model aggregates

| Figure family | Question / inputs |
| --- | --- |
| `divergence_by_type.png`, `prompt_sensitivity.png`, `generalization_width.png` | How do final predictions vary by stimulus kind and prompt? Uses posterior divergence and support rows. |
| `hspace_alpha.png`, `domain_beta.png`, `loss.png` | How do pooled fit weights and error vary across prompts and fit scopes? Uses `alpha_beta_fits.csv`. |
| `trajectory_*.png` | How do posterior metrics and fitted weights change with evidence? Uses prefix trajectory tables. |
| `hypothesis_entropy.png`, `hypothesis_matched_mass.png`, `hypothesis_unmatched_mass.png` | How concentrated and interpretable are projected hypothesis outputs? Uses hypothesis comparison rows. |
| `narrowing.png`, `confidence_weighted_types_*.png`, `trajectory_rule_by_kind.png` | How do hypothesis count and rule/interval/other mass change with evidence? Uses structured generation and eval rows. |
| `hypothesis_top1_summary.png`, `hypothesis_top1_agreement.png` | What does the highest-confidence hypothesis accept, and how do modes agree? Uses top-1 comparison rows. |

### Cross-model summaries

[summary.py](summary.py) reads per-model analysis CSVs and writes figures under `summary/`, with derived tables in `summary/cache/`. Domain 200 figures use `summary/d200/`; paired thinking comparisons use `summary/d{domain}_thinking/`. Non-root domain cache names carry a `d{domain}_` prefix.

| Figure family | Comparison / inputs |
| --- | --- |
| `kl.png`, `alpha.png`, `beta.png`, `alpha_beta_ratio.png` | Default implicit posterior trajectories over complete four-prefix stimuli; trajectory CSVs. |
| `condition*.png`, `fitting_performance.png` | Prompt, mode, domain, and fit-performance comparisons; divergence, entropy, and pooled fit CSVs. |
| `hypothesis_support.png`, `hypothesis_accuracy.png`, `hypothesis_rule_proportion.png`, `hypothesis_confidence.png` | Top-1 eval/generation trajectories; hypothesis comparison summaries. Human d100 references use cached description metrics where defined. |
| `hypothesis_*accuracy_support_scatter*.png`, `hypothesis_eval_gen_gap*.png` | Per-model support versus evidence accuracy and paired eval-minus-generation gaps; top-1 metrics. |
| `extension_mass_*.png`, `rule_discrimination*.png`, `domain_shift_*.png` | Mass beyond 100, rule versus non-rule acceptance in 101..200, and d100 shape change after extending the domain; extension CSVs. |

Summary selection, pairing, and error bars are figure-specific. [summary.py](summary.py) owns these definitions; the tables above group related plots rather than duplicate every filename and plotting option.
