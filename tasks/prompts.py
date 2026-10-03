"""Composable prompt system for the number game experiment.

Prompts are built from orthogonal blocks:
  1. Domain context       -- fixed per domain
  2. Sampling frame       -- varies by sampling factor
  3. H-space frame        -- varies by h_space factor when the eval mode needs it
  4. Examples             -- the observed numbers
  5. Eval tail            -- varies by eval_mode
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.settings import ExperimentCondition
    from tasks.prompt_space import PromptSpaceSpec

DOMAIN_CONTEXT = (
    'A computer program takes a number from 1 to {domain} and says "yes" or "no" '
    "depending on whether the number satisfies a single easy-to-state concept."
)

SAMPLING_BLOCKS: dict[str, str] = {
    "default": "",
    "strong": (
        "These numbers were drawn uniformly at random from all the numbers "
        "(1 to {domain}) that the program accepts."
    ),
    "weak": (
        "These are known positive examples, but make no assumption about how "
        "they were selected."
    ),
    "pedagogical": (
        "A teacher who knows the rule deliberately chose these numbers to help "
        "you identify it. They picked the most informative examples."
    ),
    "adversarial": (
        "All shown numbers are accepted by the program, but their selection may "
        "be biased or chosen to mislead."
    ),
}

HSPACE_EXPLICIT = (
    "The concept is one of the following types:\n{rule_list}\n\n"
    "{transform_note}"
    "It could also be numbers in an interval:\n{interval_list}\n\n"
    "Or it could be something else entirely."
)

TRANSFORM_NOTE = (
    "These concepts may also appear with a linear transformation "
    "(e.g., 2n, 3n+1, n-2) or exponential transformation "
    "(e.g., 2^n, 2^n+1) applied to the base set.\n\n"
)

POSTERIOR_TAIL = (
    'Would the program say "yes" to the number {target}? Answer only Yes or No.'
)

POSTERIOR_VECTOR_TAIL = (
    "Decide whether the program would say Yes to every number from 1 to {domain}. "
    "Output ONLY a JSON array of exactly {domain} binary integers. "
    "The first entry answers number 1 and the last entry answers number {domain}: "
    "use 1 for Yes and 0 for No. "
    "Do not include explanations or markdown."
)

POSTERIOR_YES_INDEX_TAIL = (
    "Decide whether the program would say Yes to every number from 1 to {domain}. "
    'OUTPUT FORMAT: Return exactly one JSON object of the form {{"yes": [...]}} '
    "where the list contains every integer from 1 through {domain} that the "
    "program would answer Yes to, in ascending order and without duplicates. "
    'Use {{"yes": []}} if none qualify. Do not include any other keys, reasoning, '
    "prose, or markdown."
)

POSTERIOR_REASONING_TAIL = (
    "Infer the concept from the accepted numbers; do not answer any target question yet."
)

HYPOTHESIS_EVAL_TAIL = (
    "Below is a numbered list of candidate concepts for the accepted numbers.\n"
    "{candidate_note}"
    "Assign each candidate a probability from 0.0 to 1.0 that it is the "
    "program's concept. Include every key from \"1\" to \"{n_hypotheses}\" "
    "exactly once; use 0.0 for impossible or negligible candidates. "
    "Probabilities should sum to 1.0.\n\n"
    "{hypothesis_list}\n\n"
    "Output ONLY a JSON object whose keys are candidate numbers as strings and "
    "whose values are finite numbers. No explanations or markdown fences. "
    "Format:\n"
    '{{"1": 0.35, "2": 0.40, "3": 0.25}}'
)

HYPOTHESIS_GENERATION_TAIL = (
    "Generate {n_hypotheses} candidate concepts for the accepted numbers. "
    "Each must define a set within 1..{domain}. Describe each concept clearly "
    "as a canonical label, such as \"multiples of 3\", "
    "\"perfect squares\", or \"numbers from 10 to 20\".\n\n"
    "Output ONLY a JSON array ordered by confidence: "
    '[{{"hypothesis": "description", "confidence": 0.8}}]'
)

def interval_label(lo: int, hi: int) -> str:
    return f"numbers from {lo} to {hi}"


def compose_preamble(condition: ExperimentCondition, examples: list[int], spec: PromptSpaceSpec) -> str:
    """Build blocks 1-4: domain + sampling + h_space + examples.

    hypothesis_eval lists its candidates in the eval tail, so its preamble omits
    the explicit h-space frame.
    """
    parts = [DOMAIN_CONTEXT.format(domain=condition.domain)]

    sampling_text = SAMPLING_BLOCKS[condition.sampling]
    if sampling_text:
        parts.append(sampling_text.format(domain=condition.domain))

    if condition.h_space == "explicit" and condition.eval_mode != "hypothesis_eval":
        parts.append(
            HSPACE_EXPLICIT.format(
                rule_list="\n".join(f"- {r}" for r in spec.rule_labels),
                interval_list="\n".join(f"- {label}" for label in spec.interval_labels),
                transform_note=spec.transform_note,
            )
        )

    example_str = ", ".join(str(x) for x in examples)
    parts.append(f'The program says "yes" to: {example_str}')
    return "\n\n".join(parts)


def _compose_messages(preamble: str, user_content: str) -> list[dict]:
    """System preamble + user content → chat message pair.

    Splitting enables prompt caching: all targets in a stimulus share the
    same system prefix, so the provider caches the preamble after the first call.
    The terse-reasoning hint is no-op for non-thinking models and keeps thinking
    traces short enough to fit within the HF thinking-budget cap.
    """
    return [
        {"role": "system", "content": preamble},
        {"role": "user", "content": f"{user_content}\n\nThink concisely."},
    ]


def compose_binary_messages(preamble: str, target: int) -> list[dict]:
    return _compose_messages(preamble, POSTERIOR_TAIL.format(target=target))


def compose_posterior_vector_messages(preamble: str, domain: int) -> list[dict]:
    return _compose_messages(preamble, POSTERIOR_VECTOR_TAIL.format(domain=domain))


def compose_posterior_yes_index_messages(preamble: str, domain: int) -> list[dict]:
    return _compose_messages(preamble, POSTERIOR_YES_INDEX_TAIL.format(domain=domain))


def compose_posterior_reasoning_messages(preamble: str) -> list[dict]:
    return _compose_messages(preamble, POSTERIOR_REASONING_TAIL)


def compose_hypothesis_eval_messages(
    preamble: str, hypotheses: list[str], candidate_note: str = ""
) -> list[dict]:
    hypothesis_list = "\n".join(f"{i}. {h}" for i, h in enumerate(hypotheses, 1))
    note = candidate_note.strip()
    if note:
        note += "\n\n"
    return _compose_messages(
        preamble,
        HYPOTHESIS_EVAL_TAIL.format(
            hypothesis_list=hypothesis_list,
            candidate_note=note,
            n_hypotheses=len(hypotheses),
        ),
    )


def compose_hypothesis_generation_messages(preamble: str, n_hypotheses: int, domain: int) -> list[dict]:
    return _compose_messages(
        preamble,
        HYPOTHESIS_GENERATION_TAIL.format(n_hypotheses=n_hypotheses, domain=domain),
    )
