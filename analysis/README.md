# Analysis

Analysis compares model predictions and stated hypotheses with Bayesian and human references. It reconstructs stimulus trajectories from raw prefix caches, computes condition and aggregate metrics, and writes tables used by cross-model summaries. Experiment terminology is defined in [tasks/README.md](../tasks/README.md).

## Run analysis

Run the analysis commands in the [root README](../README.md#run-an-experiment) from the repository root; stages build missing or stale Bayesian baselines on first use.

`run_analysis.py` refreshes default aggregate stages for one LLM; `--include-per-condition` also rebuilds posterior condition figures. Add `--thinking` to model analysis commands for thinking results.

`run_hypothesis_analysis.py` structures free generations and runs hypothesis condition and aggregate stages. `python -m analysis.summary` builds default cross-model summaries from analysis CSVs; it does not infer missing metrics from raw model outputs. If a summary is missing data, refresh the producing model analyses first.

## Pooled prior and size principle

Every α-β fit fixes the prior to a pooled prior and fits only β, displayed as the size principle: w(h | X) ∝ p(h) |h|^(−βn) with log p(h) = α log p₀(h) + a[base(h)] + c[transform(h)]. [prior_pooling.py](prior_pooling.py) owns it:

- `pooled_prior` fits one α and one shape v = a[base] + c[transform] per prompt (`h_space`, `sampling`), jointly over the d100 Predict data of every `SUMMARY_MODEL_ORDER` model and its `_thinking` variant, plus humans for the default implicit prompt.
  - Rows are each source's unique (task, prefix) inputs with tasks pooled. Humans contribute their full sets at n = |set|.
  - β is one per (source, n) during this fit, anchored at 1 with λ = `ALPHA_BETA_REG_LAMBDA`; the shape ridge is 1e-3.
  - Rules take their base and transform from the task's rule families; all intervals share one base.
  - Each prompt's prior is cached as `results/prior_pooling/pooled_prior_{h_space}_{sampling}.json`. It is refit when its source list, Bayesian settings, or regularization change; delete the file to refit after a source's raw caches change, then rerun the affected analyses.
- Which prior each fit uses:
  - Predict fits use their own prompt's prior at every domain, so `condition_beta.png` compares prompts each under its own prior.
  - Eval and Generate fits, and human fits, use the default implicit prior.
- The Bayesian reference stays the unfitted model, (α, β) = (1, 1).

`python -m analysis.prior_pooling` reruns the comparison that chose this prior and writes `results/prior_pooling/prior_pooling_cv.csv`. The comparison uses the default implicit sources above, with β one per (source, n) in every configuration:

- Scoring is 4-fold CV keyed by input, so an input is held out for every source at once.
- `r2` is the pooled squared Pearson r of held-out predictions over rated targets.
- The CSV has one row per config, source, and `n_examples` (`all` or 1–4), with `n_inputs` and the configuration's `n_params`.

| Config | Prior parameter | `n_params` | LLM mean `r2` (16 sources) | Human `r2` |
| --- | --- | --- | --- | --- |
| `1_alpha_per_source_n` | α per (source, n); the former per-scope α fit with tasks pooled | 136 | 0.229 | 0.240 |
| `2_alpha_per_source_n_reg0.1` | As 1, λ = 0.1 on α | 136 | 0.233 | 0.240 |
| `3_alpha_per_n` | One α per n, shared by all sources | 72 | 0.232 | 0.240 |
| `4_alpha_per_source` | One α per source, shared across n | 85 | 0.229 | 0.240 |
| `5_alpha_plus_pooled_shape` | α per (source, n) on log p₀, plus the shared shape v | 182 | 0.278 | 0.306 |
| `6_rank1` | α = 1; log p₀ + v₀ + w·v with shared v₀, v and one w per (source, n) | 228 | 0.285 | 0.316 |
| `7_alpha_per_n_plus_pooled_shape` | As 5, α per n | 118 | 0.280 | 0.309 |
| `8_alpha_per_source_plus_pooled_shape` | As 5, α per source | 131 | 0.280 | 0.308 |
| `9_alpha_global_plus_pooled_shape` | As 5, one α for every source and n; the pooled prior | 115 | 0.280 | 0.309 |

The shared shape carries the gain: without it, pooling α differently changes little, and with it a single global α matches α per n, per source, or per (source, n). `9` is the pooled prior because it matches 5, 7, and 8 with the fewest parameters, leaving the size principle as the only per-(source, n) quantity.

## Hypothesis structuring

[hypothesis_structuring.py](hypothesis_structuring.py) parses precise descriptions into support sets over `1..domain`. It builds on [hypothesis_terms.py](hypothesis_terms.py), the term grammar that turns rule text into executable terms, and [hypothesis_text.py](hypothesis_text.py), the generic text utilities (number words, normalization, number extraction, set literals). Its module docstring states the parsing rules.

A structured description reuses the library predicate with identical finite-domain support or appends one to `library/library_d{domain}.py`. Predicate libraries and `hypothesis_generation_structured.json` are published together transactionally; run structuring writers serially, since the transaction does not isolate concurrent processes. Rerunning structuring reparses the raw descriptions. Existing indices retain their executable support because other caches and human annotations can reference them, so a corrected description can move to another existing index or append a new predicate. Library labels alone are not semantic ground truth: audit the original description against its complete support over `1..domain`.

Each unresolved item is stored as `other` with a `reason` derived from how the parser fails: `vague` (no precise rule, or text that retracts itself), `contradictory` (a rule plus a remark or clause that disagrees with it), `empty` (a precise rule with no support in `1..domain`), or `unparsed` (anything else the grammar cannot read). Per-domain stats count each as `other_<reason>`. `hypothesis_types` requires `reason` on every `other` item. `python -m analysis.hypothesis_structuring --dry-run` writes nothing and reports how supports would change against the existing structured files; every run also prints the most frequent `unparsed` and `contradictory` templates as the grammar work queue.

Hypothesis-generation analysis requires this structured artifact and never reads raw free text as its analysis input. The projection logic in [hypothesis_space.py](hypothesis_space.py) puts predictions, supplied-hypothesis ratings, and structured generations into a shared predictive space while tracking rule, interval, and `other` (non-executable) mass.

## Stage ownership and tables

[pipeline.py](pipeline.py) owns dispatch. Default stages run per domain through `DEFAULT_STAGES`; hypothesis stages use `HYPOTHESIS_STAGES`. Cross-domain extension stages run once per model through `DEFAULT_MODEL_STAGES` and `HYPOTHESIS_MODEL_STAGES`. Condition plotting is a separate pass, not an aggregate stage.

| Default stage | Purpose | Main CSVs |
| --- | --- | --- |
| `divergence` | Compare final posterior predictions with references. | `divergence_metrics.csv`, `divergence_human_metrics.csv`. |
| `alpha_beta` | Fit the size principle under each prompt's pooled prior. | `alpha_beta_fits.csv`. |
| `prompt` | Compare prompt sensitivity and prediction width. | `prompt_sensitivity.csv`, `prompt_effective_support.csv`. |
| `trajectory` | Track posterior KL, entropy, and support as evidence grows. | `trajectory_prefix_metrics.csv`. |
| `rt_correlation` | Relate available model metrics to human response times. | `rt_correlation_*.csv`. |
| `extension` (cross-domain) | Measure d200 extension. | `extension_mass.csv`, `extension_rule_discrimination.csv`. |

| Hypothesis stage | Purpose | Main CSVs |
| --- | --- | --- |
| `nested_pair` | Compare nested rule-pair behavior. | `nested_pair_*.csv`. |
| `alpha_beta_mode` | Fit the size principle of projected predictions for each evaluation mode. | `alpha_beta_mode_fits.csv`. |
| `generation_narrowing` | Count unique structured hypotheses over evidence. | `generation_narrowing_counts.csv`. |
| `hypothesis_divergence` | Compare projected hypothesis predictions with references. | `hypothesis_divergence_metrics.csv`. |
| `hypothesis_types` | Measure rule/interval/other composition and confidence of eval and generation outputs. | `hypothesis_types_*.csv`; the per-stimulus `hypothesis_types_metrics.csv` also counts `other` items by reason (`other_vague`, `other_empty`, `other_contradictory`, `other_unparsed`). |
| `hypothesis_compare` | Project all modes into shared predictive space per prefix; measure top-1 properties, entropy, rule/interval/other mass, top-k rule share, and cross-mode distance. | `hypothesis_compare_*_metrics.csv`, `hypothesis_compare_*_summary.csv`. |
| `hypothesis_fit` | Fit explicit posterior predictions to the Eval label space and compare with Eval self-reports. | `hypothesis_fit_metrics.csv`. |
| `hypothesis_extension` (cross-domain) | Extension diagnostics for eval and generation. | `hypothesis_extension_mass.csv`, `hypothesis_extension_rule_discrimination.csv`. |

Outputs depend on available compatible inputs; a stage need not emit every table for every model.

## Data and metric contracts

Raw caches use a flat `prefix_cache`; [cache.py](cache.py)'s `reconstruct_trajectory()` rebuilds per-stimulus sequences. Condition identity and raw-cache replacement rules are documented in the [root README](../README.md#select-conditions-and-reuse-caches). Analysis CSVs use `stimulus` and `stimulus_kind` for stimulus identifiers; pooled stimulus-kind rows use `all`.

Analysis covers prefix lengths `PREFIX_LENGTHS = (1, 2, 3, 4)` in [alpha_beta_scope.py](alpha_beta_scope.py). Trajectory stages keep stimulus sets with at least `max(PREFIX_LENGTHS)` examples (capped by the longest stimulus) and truncate them to that length, so every prefix bin averages the same stimuli. This is independent of the scoring limit `SCORING.max_examples`.

Aggregate size-principle fits are per model, domain, `h_space`, `sampling`, and fit scope, under the pooled prior chosen in [Pooled prior and size principle](#pooled-prior-and-size-principle). Every scope pools tasks, and `task` records the pooled task group. Prefix scopes `n=1..4` fit the trajectory-horizon stimuli truncated to n examples; `full` fits every task's full stimulus sets. `alpha_beta_mode_fits.csv` adds evaluation mode after projection. Valid, exhausted, and usable counts distinguish source availability from finite fit targets; only complete final-prefix projections contribute to `full`.

| Metric | Interpretation |
| --- | --- |
| `beta` | Fitted size-principle weight under the pooled prior, displayed as "Size principle (β)"; the Bayesian reference is 1. |
| `alpha` | The pooled prior's α, fixed per prompt and copied onto each row; not fitted per scope. The Bayesian reference is 1. |
| KL | Direction is Bayesian to model where labeled `kl_bayes`. Both sides are normalized, then the model side mixes in `KL_LAPSE` = 0.001 uniform mass ([metrics.py](metrics.py)), so a target the model rules out costs a bounded penalty. |
| Text-trial log space | Text-trial posteriors (`mode_config.method == "text_trials"`) are frequencies over each prefix's `valid_vectors`. Log-space metrics (posterior KL to Bayesian or human, posterior entropy, RT `H_llm`, and posterior entropy and JS distance in `hypothesis_compare`) use Jeffreys-smoothed marginals `(k + 0.5) / (n + 1)`; linear metrics (correlation, support, size-principle fits, extension mass and means) use the raw frequencies. Logits posteriors are unchanged. |
| `kl_bayes_ceiling` | In `trajectory_prefix_metrics.csv`, the expected smoothed `kl_bayes` of the Bayesian prediction itself remeasured with the same number of text trials (200 seeded draws): the score of a perfect Bayesian under text-trial measurement. NaN for logits rows. |
| `jsd` / `*_jsd` | Base-2 Jensen-Shannon **distance**, the square root of divergence. |
| Effective support | Number of targets with `P(yes) > 0.5`. |
| Grouped Eval families | Bigelow16 Eval labels name a base family; its executable proxy is the uniform mean over the family's transformed rules that accept every shown example, or over all of them when none does. Projection, the hypothesis-fit basis, and top-1 metrics share this proxy. |
| Top-1 normalized confidence | Normalized confidence of the highest-confidence executable hypothesis (Eval skips `other` labels above it). NaN when the top-1 type is `other`, so an all-`other` Eval answer does not score 1.0; like size and accuracy, it averages executable top-1 rows only. |
| Top-1 hypothesis size | Fraction of the domain accepted by the highest-confidence hypothesis; for a grouped family, its members' mean. |
| Top-1 type | `top1_type` is `rule` or `interval` for an executable top-1, `other` for an answer without one, and empty for an exhausted answer. Summary `top1_{rule,interval,other}_preference_mean` average type indicators over answered prefixes, so `other` counts against rule preference and exhausted answers are skipped. |
| Top-k rule share | `rule_share_top{1,3,5}` in `hypothesis_compare_entropy_metrics.csv`: rule weight / (rule + interval weight) among a mode's k most confident executable hypotheses; `other` items are excluded, items tied at the cutoff split its remaining slots equally, and the value is NaN without executable items. |
| Top-1 evidence accuracy | Whether the hypothesis accepts all shown examples (0 or 1); a grouped family needs one member that accepts them all. Summaries average these indicators. |
| Extension mass | Probability mass on 101..200 after normalizing d200 per-target `P(yes)` into a distribution, so a model's overall yes-rate cancels. Its Bayesian value depends on how much of each rule lies beyond 100. |

Summary tables keep model identity in `model`, data provenance in `source`, and evaluation-mode display names in `mode_label`. Cross-model mode trajectories count observations with `n_models`; model trajectories use `n_stimuli`. Summary labels `Predict`, `Eval`, and `Generate` mean `posterior`, `hypothesis_eval`, and `hypothesis_generation`. These modes are distinct from producers such as `llm`, `human`, and `bayesian`. Entropy units depend on the metric: posterior summary entropy uses nats; hypothesis comparison entropy uses bits.

## Find and interpret figures

Paths below are relative to `results/`. The [root README](../README.md#configuration-and-outputs) defines `DATA_DIR`, `{prompt}`, and the per-domain and cross-domain analysis and figure directories.

### Baselines and individual conditions

Bayesian decompositions live under `bayesian/{task}/d{domain}/decompositions/`; human fit comparisons live under `human/{task}/d{domain}/alpha_beta/`. Model condition figures live under `{model}/{task}/d{domain}/{prompt}/`; they and the model aggregate figures below are drawn only when `PLOT_MODEL_FIGURES` is enabled (see the [root README](../README.md#configuration-and-outputs)). Large tasks restrict per-stimulus plots using `featured_stimuli`; this does not restrict scoring.

| Figure family | What it shows / inputs |
| --- | --- |
| `bayesian_panels.png`, `llm_panels.png`, `histogram/` | Final membership curves, with human judgments where available. |
| `alpha_beta/ablation_*.png` | Final curves against the Bayesian model, the pooled prior alone, the likelihood alone (uniform prior), and the size-principle fit. |
| `trajectory_*.png` | Prefix-wise KL, entropy, support, and size principle (`trajectory_beta.png`). |
| `hypothesis_comparison.png` | Supplied-hypothesis probabilities against the Bayesian reference. |
| `hypothesis_generation/`, `hypothesis_generation_types.png`, `hypothesis_generation_exemplars.png` | Structured generation confidence, type composition, and selected examples. |

### Model aggregates

| Figure family | Question / inputs |
| --- | --- |
| `divergence_by_type.png`, `prompt_sensitivity.png`, `generalization_width.png` | How do final predictions vary by stimulus kind and prompt? Uses posterior divergence and support rows. |
| `domain_beta.png`, `loss.png` | How do the size principle and fit error vary across prompts and fit scopes? Uses `alpha_beta_fits.csv`. |
| `trajectory_*.png` | How do posterior metrics change with evidence? Uses prefix trajectory tables; `trajectory_beta.png` draws the prefix-scope size principle from `alpha_beta_fits.csv`. |
| `hypothesis_entropy.png`, `hypothesis_other_mass.png` | How concentrated and interpretable are projected hypothesis outputs? Uses hypothesis comparison rows. |
| `narrowing.png`, `confidence_weighted_types_*.png`, `trajectory_rule_mass.png` | How do hypothesis count and rule/interval/other mass change with evidence? Uses structured generation and eval rows. |
| `hypothesis_top1_summary.png`, `hypothesis_top1_agreement.png` | What does the highest-confidence hypothesis accept, and how do modes agree? Uses top-1 comparison rows. |

### Cross-model summaries

[summary.py](summary.py) reads per-model analysis CSVs and draws each `SUMMARY_COHORTS` cohort: `BEST_SUMMARY_MODELS` under `summary/` and every registered summary model under `summary/all/`. Only `all/` draws `fitting_human_kl.png`, `human_correlation_by_kind.png`, `hypothesis_{eval,generation}_accuracy_size_scatter.png`, and `rule_discrimination.png`. Paths below are relative to the cohort root; derived tables go in its `cache/`. Domain 200 figures use `d200/`; paired thinking comparisons use `d{domain}_thinking/`. Non-root domain cache names carry a `d{domain}_` prefix.

| Figure family | Comparison / inputs |
| --- | --- |
| `kl.png`, `beta.png` | Default implicit posterior trajectories over complete four-prefix units. `kl.png` uses trajectory CSVs; `beta.png` draws the size principle of the task-pooled prefix scopes in `alpha_beta_fits.csv`. The d100 human endpoint (the human full-scope size principle for `beta.png`) draws as a dotted black reference line. `kl.png` and `kl_thinking.png` also draw thinking variants and a `Text-trial ceiling` pooled from text-trial models' `kl_bayes_ceiling`. |
| `beta_size.png` | Does model size shift the size principle? One mean line per `SIZE_FAMILY_MODELS` size and thinking mode, drawn only when every family has that model, with the d100 human endpoint. Drawn identically in both cohort roots. |
| `beta_thinking_mean.png` | Does thinking shift the size principle? One line per thinking mode, the mean over the cohort's paired base/thinking models with SEM across models, with the d100 human endpoint. |
| `condition*.png` | Prompt and mode comparisons; divergence and entropy CSVs. Prompt trajectories (`condition_{kl,beta}.png`) draw the d100 human endpoint as a dotted black reference line; in `condition_beta.png` each prompt's size principle is fit under that prompt's own pooled prior. `condition.png`, `condition_task.png`, and `condition_eval_gen.png` show base/thinking pairs. Entropy panels in `condition.png` and `condition_task.png` draw the mean human entropy over the same endpoint stimuli as a reference line (dense Bigelow16 ratings only); their size-principle and KL panels draw the d100 human full-scope fit and KL means. `condition_task.png` compares Predict, Eval, and Generate on the size principle, KL, predictive entropy, and d200 prob. mass on 101–200 (averaged over rule and interval stimuli and prefixes 1–4), over pairs complete in every panel. `condition_eval_gen.png` compares Eval and Generate on top-1 accuracy, confidence, rule AUC, hypothesis size, and rule preference, with human references. Rule AUC is the criterion-free probability that a `rule` stimulus has a higher top-3 rule share than a `similarity` stimulus of the same task, scored at each stimulus's full example set; ties count one half, so 0.5 is chance. `condition_rule_auc_top_k.png` repeats it for top-1, top-3, and top-5 (`summary_rule_auc_top_k_by_model.csv`). The human AUC rates each Bigelow16 set by rule / (rule + interval) descriptions and classifies sets with `classify_bigelow_kind`. |
| `fitting_human_kl.png`, `human_correlation_by_kind.png` | How close is each LLM to humans? `fitting_human_kl.png` draws per-model full-scope size-principle fit r² under the pooled prior (`alpha_beta_fits.csv`) beside per-model mean KL(human ‖ LLM) over every human-rated default implicit stimulus (`divergence_human_metrics.csv`), with one shared model legend; `human_correlation_by_kind.png` draws the LLM–human correlation on `rule` and `similarity` stimuli (both tasks pooled) as the mean ± SEM across per-model means. |
| `hypothesis_size.png`, `hypothesis_accuracy.png`, `hypothesis_rule_preference.png`, `hypothesis_confidence.png` | Top-1 Eval/Generate trajectories over base/thinking pairs with 95% CI across models. Human d100 references use cached description metrics where defined. |
| `hypothesis_beta.png` | Predict/Eval/Generate size-principle trajectories (task-pooled prefix scopes of each mode's default condition in `alpha_beta_mode_fits.csv`; Predict is the default implicit fit behind `beta.png`) over base/thinking pairs with 95% CI and the d100 human endpoint from `beta.png`; not drawn under `d{domain}_thinking/`. `beta_prompts.png` draws only the non-thinking lines of the same pairs. |
| `hypothesis_entropy_trajectory.png` | Predict/Eval/Generate predictive-entropy trajectories in nats (`hypothesis_compare_entropy_summary.csv`, tasks averaged within model) over base/thinking pairs with 95% CI and the d100 human endpoint entropy from `condition.png`; the endpoint feeds the `condition_task.png` entropy bars; not drawn under `d{domain}_thinking/`. |
| `hypothesis_*accuracy_size_scatter*.png` | Per-model hypothesis size versus evidence accuracy; top-1 metrics. |
| `extension_mass.png` | d200 extension mass by prefix for Predict/Eval/Generate over base/thinking pairs where both variants are complete for every mode; `summary_extension_mass.csv`. |
| `extension_decline_{drop,slope}.png` | Similarity minus rule decline of d200 extension mass per model, one panel per Predict/Eval/Generate, thinking bars lighter, Bayesian dashed line. Per stimulus, `drop` is m(1) − mean m(2..4) and `slope` is the least-squares slope over prefixes; stimuli with an undefined prefix (all-zero d200 P(yes)) are skipped before averaging within kind; `summary_extension_decline.csv`. |
| `rule_discrimination*.png` | Rule versus non-rule acceptance in 101..200; extension CSVs. |

Summary selection, pairing, and error bars are figure-specific. [summary.py](summary.py) owns these definitions; the tables above group related plots rather than duplicate every filename and plotting option.
