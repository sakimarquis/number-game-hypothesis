You are labeling human-written number-game set descriptions.

## Input

A chunk of rows from `dataset/set_descriptions.csv`. Columns:

- `id`: subject id
- `set`: shown numbers (underscore-separated)
- `descr`: the human description

## Output

Return the same rows with one new column `type` appended. Preserve `id`, `set`, and `descr` exactly. CSV only — no markdown fences, no commentary, no extra columns. Header row first.

## Allowed `type` values

- `rule`: a precise non-contiguous predicate. Includes odd/even, prime/composite, multiples or divisibility, last-digit, contains-digit, squares/cubes/powers, explicit finite sets, and exact boolean combinations such as `"odd numbers under 50"` or `"evens or multiples of 11"`. Misspellings still count: `"uneven"` = odd, `"enens"` = evens.
- `interval`: a pure contiguous numeric range. Includes bounded ranges (`"between 10 and 20"`, `"10 to 20"`), one-sided (`"under 30"`, `"over 50"`, `"at least 100"`, `"any number 100 and down"`), and singletons (`"the number 16"`, `"only 47"`).
- `dk`: the writer explicitly says they do not know — `"not sure"`, `"no idea"`, `"can't tell"`, `"I don't know"`, `"???"`, or equivalent.
- `other`: vague, subjective, contrastive, malformed, non-numeric, or underspecified. Examples: `"random numbers near 80"`, `"numbers similar to these"`, `"numbers decreasing in value"`, `"between these numbers"`, `"integers"`, `"forms of 8"`.

## Rules

- Use `interval` only for purely contiguous ranges. If the description combines an interval with another exact property (`"odd numbers between 20 and 40"`), use `rule`.
- Use `rule` for exact patterns even with informal wording or typos.
- Use `dk` only for explicit uncertainty, never for vague hypotheses.
- Prefer low false precision: when exact membership is unclear, use `other`.
- One label per row. Decide on `descr` alone; ignore `set` unless the text is itself referential (e.g. `"these numbers"` → `other`).

## Short examples

```
id,set,descr,type
62,47,Odd numbers,rule
62,42_ 62,numbers between 42 and 65,interval
389,47,prime numbers,rule
219,73_ 35_ 53_ 59,numbers with fives or threes in them,rule
298,55_ 15_ 75_ 45,numbers ending in 5,rule
1,1,not sure,dk
1,1,random numbers near 80,other
1,1,integers,other
1,1,any odd number or even numbers ending in 0,rule
1,1,any number 100 and down,interval
```
