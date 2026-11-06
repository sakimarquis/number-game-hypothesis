# Tasks and evaluation

The number game asks which numbers belong to an unknown concept after seeing accepted examples. A task supplies stimulus sets and human reference data; an evaluation mode determines what the LLM is asked to return. Installation and commands are in the [root README](../README.md).

## Tasks, stimuli, and conditions

| Task | Stimulus sets | Source |
| --- | --- | --- |
| `Tenenbaum99` | 8 hand-written sets | Tenenbaum (1999). |
| `Bigelow16` | 255 sets | Bigelow & Piantadosi (2016) human data. |

A **stimulus set** (`StimulusSet`; the `stimulus` column in analysis tables) is a full example sequence, such as `D={16,8,2,64}`. A **prefix** is one shown evidence sequence: `[16]`, `[16,8]`, `[16,8,2]`, or `[16,8,2,64]`. Scoring stops at `SCORING.max_examples`; identical prefixes within a condition are scored once and reused. `dataset` refers to the raw files under `dataset/`, not to an evaluation mode.

An `ExperimentCondition` fixes task, domain, evaluation mode, `h_space`, and `sampling`; the selected model determines its result root. The domain is `1..domain`. Conditions sharing task, domain, `h_space`, and sampling form a prompt group (`ExperimentCondition.prompt_group`) and share one result directory named by the prompt `{h_space}_{sampling}` (`ExperimentCondition.prompt`), which figures also use as their prompt label. Enabled combinations and the scoring prefix horizon live in [configs/number_game.yml](../configs/number_game.yml). Human reference judgments are native to domain 100.

## `h_space` and `sampling`

`implicit` gives the domain, examples, and sampling text. `explicit` also describes the task/domain rule labels, four prefix-dependent intervals, and an `other` option. This changes the stated hypothesis space, not the observed examples.

| Sampling | Text given to the model |
| --- | --- |
| `default` | No sampling text. |
| `strong` | Positives were sampled uniformly from accepted numbers. |
| `weak` | The examples are positives, without uniform sampling. |
| `pedagogical` | A teacher chose informative examples. |
| `adversarial` | The examples may be misleading. |

The task's `h_spaces` sweep applies only to `posterior`. `hypothesis_eval` is always explicit; `hypothesis_generation` is always implicit (`HYPOTHESIS_H_SPACE` in [configs/settings.py](../configs/settings.py)). The task also fixes its rule space (`TASK_RULE_SPACES` in [stimuli.py](stimuli.py)): Tenenbaum uses all bases untransformed; Bigelow composes its primordial bases with every registered transform.

## Evaluation modes

### Posterior: predict membership

The output is a probability curve over every target in the domain. With `logits`, each target gets a Yes/No question, such as “The program says yes to 16, 8. Would it say yes to 32?” The scorer normalizes the available Yes/No probability mass.

With `text_trials`, each request supplies binary judgments for the whole domain. The scorer averages `POSTERIOR.n_trials` vectors, excluding malformed trials as whole rows. A prefix with no valid vector gets up to two structured-recovery rounds; if none succeeds, scoring fails before publishing the cache. A prefix with a valid vector does not retry its individual malformed trials. `posterior_diagnostics` records requested and valid counts. Backend selection and wire formats are explained in [backends/README.md](../backends/README.md).

### Hypothesis eval: rate supplied candidates

The model assigns probabilities to a numbered candidate list. Answers map back to that prefix's labels; stored results contain only candidate labels.

Candidates consist of task/domain rules, four intervals derived from the shown prefix, and `other`. The intervals round the example range outward to multiples of 10 or 5, use exact min/max, or span the full domain. For `[16]` in domain 100, these are `10..20`, `15..20`, `16..16`, and `1..100`. Interval labels may coincide; numbered entries need not have unique labels.

Tenenbaum uses individual rule labels; Bigelow uses grouped rule families. [prompt_space.py](prompt_space.py) builds the shared prompt-local candidate specification for scoring and analysis.

### Hypothesis generation: propose rules freely

The model proposes up to `HYPOTHESIS_GENERATION.n_hypotheses` rules with confidence values:

```json
[{"hypothesis": "multiples of 8", "confidence": 0.6}, {"hypothesis": "powers of 2", "confidence": 0.3}]
```

A hypothesis-mode prefix whose answer is still unparseable after `SCORING.max_retries` re-asks stores `{}` (eval) or `[]` (generation). Such caches stay reusable; analysis counts those prefixes as exhausted.

There is no supplied candidate list. Later analysis structures precise descriptions into executable predicates over the domain; it does not match them against the hypothesis-eval candidate list. See the [analysis workflow](../analysis/README.md#hypothesis-structuring).

## Implementation map

[stimuli.py](stimuli.py) loads task data and human judgments, [hypotheses.py](hypotheses.py) defines hypothesis registries, [prompt_space.py](prompt_space.py) derives prefix-local candidates, and [prompts.py](prompts.py) renders prompts. Evaluation and parsing live in [scoring/eval_modes.py](../scoring/eval_modes.py).
