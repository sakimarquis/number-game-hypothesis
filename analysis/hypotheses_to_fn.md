---
name: hypotheses-to-fn
description: Cast on a Tenenbaum99 results root (e.g., `results/<model>/Tenenbaum99`). Converts every raw `hypothesis_generation.json` under each `d{domain}/<prompt_condition>/data/` into a structured JSON next to it, and updates one shared library per domain at `repo_root/library/library_d{domain}.py`.
---

# Hypotheses To Executable Predicates

Cast on `results/<model>/Tenenbaum99`. The skill enumerates every `d{domain}/<prompt_condition>/data/hypothesis_generation.json` under that root, classifies each generated hypothesis, registers precise ones as Python predicates in a per-domain library, and writes a structured JSON next to every raw file.

Goal: conservative extraction. Only precise hypotheses become executable; vague text becomes `"other"`.

## Workflow

Main-agent procedure when cast on `results/<model>/Tenenbaum99`:

1. Enumerate domain roots (`d100`, `d200`, …) under the cast target.
2. For each domain, enumerate implicit prompt conditions (e.g., `implicit_default`, `implicit_adversarial`) and locate every `data/hypothesis_generation.json`.
3. Read the current `library/library_d{domain}.py`. Create it from the skeleton in **Library Contract** if it does not exist.
4. Dispatch **one subagent per `d{domain}/<prompt_condition>` shard** in parallel (across domains too). Brief each subagent with: domain size, domain root, prompt condition name, the full current `library/library_d{domain}.py`, and the raw payload(s) for that shard. Point them at this skill and require the marker-separated output of the **Output Contract**.
5. Merge per domain: collect all proposals for `d{domain}`, deduplicate new predicates by behavior signature on `1..{domain}` against the existing library and across the batch, assign final indices using the smallest unused non-negative integers, and write the reconciled `library/library_d{domain}.py`.
6. Write each `hypothesis_generation_structured.json` next to its raw input using the **final** indices.
7. Report per domain: number of new functions added, number of structured JSONs written, and any borderline `"other"` classifications worth spot-checking.

Subagents only **propose**. The main agent owns final merge and index assignment — parallel writes to the same library file would collide. Domains are independent and can run fully in parallel.

## Library Location

Libraries live in `repo_root/library/`, one file per domain:

```text
repo_root/library/library_d100.py
repo_root/library/library_d200.py
```

Every prompt condition under `d{domain}` reads and extends the same file. Reuse existing functions whenever the finite-domain behavior signature matches; add a new function only when no existing function matches.

## Subagent Prompt Template

You are structuring generated number-game hypotheses for one domain/prompt-condition shard.

Inputs:

- Domain size: `1..{domain}`
- Domain root: `results/{model}/Tenenbaum99/d{domain}`
- Prompt condition: `{prompt_condition}`
- Current shared `library/library_d{domain}.py` from the repo root:

  ```python
  {existing_library_py}
  ```

- Raw `data/hypothesis_generation.json` payloads for this shard, keyed by their path relative to the domain root:

  ```json
  {raw_hypothesis_generation_payloads}
  ```

Return marker-separated text only — no markdown fences. Use your best-effort indices; the main agent will reconcile across all subagents for the domain.

## Output Contract

```text
=== FILE: library_d{domain}.py ===
<full proposed library content, including any new functions you would append>

=== FILE: <relative output path>/hypothesis_generation_structured.json ===
<structured JSON content>

=== FILE: <next relative output path>/hypothesis_generation_structured.json ===
<structured JSON content>
```

Each structured JSON is written next to its raw `hypothesis_generation.json`. It preserves the raw structure and all original fields, and adds:

- `type`: `"rule"`, `"interval"`, or `"other"` for every hypothesis.
- `index`: only for `"rule"` or `"interval"`. Never on `"other"`.

After the main agent merges all proposals for one domain, the final `repo_root/library/library_d{domain}.py` and structured JSONs are written using the reconciled indices.

## Library Contract

Keep the library minimal — an index-to-function table:

```python
import math
from collections.abc import Callable

LIBRARY: dict[int, Callable[[list[int]], bool]] = {}


def register(index: int):
    def deco(fn: Callable[[list[int]], bool]):
        LIBRARY[index] = fn
        return fn
    return deco
```

Each predicate is a pure function:

```python
@register(0)
def fn_0000(xs: list[int]) -> bool:
    """even numbers"""
    return all(x % 2 == 0 for x in xs)
```

`all(...)` over `xs` makes `fn([n])` equivalent to membership for `n`.

Use only: `all(...)`, interval bounds, set membership, integer arithmetic, boolean combinations, and `math.isqrt` when needed. No I/O, network, randomness, or extra imports.

Predicates encode the hypothesis itself; do not add domain clamping like `x <= 100`.

Do not store `type`, `note`, `source`, or `derived_from` in `LIBRARY`. The description belongs in the docstring; `type` belongs in structured JSON.

## Index Rules

Indexes are explicit source-level identifiers.

- Never modify, reorder, rename, or renumber existing functions.
- New functions use the smallest non-negative integer not already used.
- Function names are `fn_NNNN` with at least four digits (more if needed, e.g., `fn_0042` or `fn_12345`).
- Decorator index and function-name suffix must match.

```python
@register(42)
def fn_0042(xs: list[int]) -> bool:
    """multiples of 7"""
    return all(x % 7 == 0 for x in xs)
```

## Deduplication

Deduplicate by behavior on the active finite domain, not by text.

```python
def signature(fn, domain):
    return tuple(fn([x]) for x in range(1, domain + 1))
```

Before adding a function, compare its signature against all existing library functions and all functions newly added in this same merge. If signatures match, reuse the existing or earlier index.

Examples that share one index on the active domain:

- `"even numbers"` and `"multiples of 2"`
- `"numbers from 16 to 16"` and `"the number 16"`
- `"x > 50"` and `"numbers from 51 to DOMAIN"`

When multiple raw entries in the same shard have the same signature, keep each raw entry in the structured JSON but assign them the same `index`. Any later projection must collapse rows by `index` before turning confidence into support-vector mass; do not double-count duplicate raw phrasings. A conservative default is keeping the maximum confidence per `index` per shard.

If a candidate predicate has an empty extension on `1..DOMAIN`, mark it `"other"` and do not add a function.

## Type Rules

`"interval"` — pure contiguous numeric ranges:

- bounded: `"10 to 20"`, `"between 10 and 20"`, `"[10, 20]"`;
- singleton: `"only 16"`, `"the number 16"`;
- one-sided: `"greater than 50"`, `"at least 50"`, `"at most 20"`.

`"rule"` — precise, executable, non-pure-interval:

- odd, even, prime, composite, square, cube, multiple of `k`, power of `k`;
- `"numbers ending in d"`, `"5n+k"`;
- explicit finite sets like `{16, 17, 19}`;
- precise compounds: `"even numbers less than 50"`, `"odd or prime"`.

`"other"` — vague, subjective, contrastive, or requires guessing parameters:

- similarity/salience: `"numbers similar to the examples"`, `"roundish"`;
- vague distance: `"near the examples"` without an explicit bound;
- underspecified compounds: `"some combination of even and high numbers"`.

Low recall is better than false precision.

## Short Example

Existing `repo_root/library/library_d100.py` contains:

```python
@register(0)
def fn_0000(xs: list[int]) -> bool:
    """even numbers"""
    return all(x % 2 == 0 for x in xs)
```

Raw input for `d100/implicit_default/data/hypothesis_generation.json`:

```json
[
  {"hypothesis": "multiples of 2", "confidence": 0.40},
  {"hypothesis": "even numbers", "confidence": 0.20},
  {"hypothesis": "numbers from 16 to 23", "confidence": 0.35},
  {"hypothesis": "16, 17, or 19", "confidence": 0.15},
  {"hypothesis": "even prime numbers greater than 10", "confidence": 0.05},
  {"hypothesis": "numbers that feel similar to the examples", "confidence": 0.05}
]
```

After merge, `library_d100.py` appends:

```python
@register(1)
def fn_0001(xs: list[int]) -> bool:
    """numbers from 16 to 23"""
    return all(16 <= x <= 23 for x in xs)


@register(2)
def fn_0002(xs: list[int]) -> bool:
    """one of 16, 17, or 19"""
    return all(x in {16, 17, 19} for x in xs)
```

Structured JSON written to `results/<model>/Tenenbaum99/d100/implicit_default/data/hypothesis_generation_structured.json`:

```json
[
  {"hypothesis": "multiples of 2", "confidence": 0.4, "type": "rule", "index": 0},
  {"hypothesis": "even numbers", "confidence": 0.2, "type": "rule", "index": 0},
  {"hypothesis": "numbers from 16 to 23", "confidence": 0.35, "type": "interval", "index": 1},
  {"hypothesis": "16, 17, or 19", "confidence": 0.15, "type": "rule", "index": 2},
  {"hypothesis": "even prime numbers greater than 10", "confidence": 0.05, "type": "other"},
  {"hypothesis": "numbers that feel similar to the examples", "confidence": 0.05, "type": "other"}
]
```
