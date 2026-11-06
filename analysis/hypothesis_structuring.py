"""Structure free-form generated hypotheses against executable libraries.

The structuring contract is finite-domain semantic matching: parse precise
natural-language hypotheses into support signatures on 1..domain, reuse any
existing library function with the same signature, and append a new predicate
only when no existing function matches.

Parsing rules (anything they do not cover leaves the description unresolved):
- Every digit or content word of the rule text must be consumed by a term parser;
  leftover words are never dropped.
- The rule text is the quoted rule or the text before a justification such as
  "because"; a failed rule is never re-read from its justification, and a quoted
  rule preceded by a rival claim ("the number is 46 because ..., so the rule is
  '...'") stays unresolved.
- A parenthetical or "i.e." restatement must be equal to or looser than the rule
  (a listed example set must be a subset). Narrower or conflicting remarks, and
  remarks that neither parse nor are known non-narrowing notes such as
  "(inclusive)" or a claim about one example, leave the description unresolved.
  Examples after a complete quoted rule add no conditions.
- When the whole text does not parse, it is read clause by clause (sentences,
  semicolons, "and", "but", "so", "while"): the first clause is the rule, and each
  later clause must be a true claim about listed numbers ("69 is a multiple of
  3", "16 = 4*4"), a framed restatement with the same support ("the program
  checks divisibility by 6"), closed-list commentary ("this fits the data"), or a
  further condition on the number, which is ANDed.
- Numeric-list operands, supported negation, digit properties, bounded forms, and
  explicit Boolean grouping are kept. An understood branch with empty support can
  take part in a Boolean expression; an unsupported branch cannot be treated as
  empty, and a complete hypothesis with empty support stays unresolved.
- Bounds or parity stated for an attribute (a root, a digit position or digit
  property, the variable of a form like `2^k` or `7k+1`, a count of digits or
  divisors) apply to that attribute, never to the number itself.
- A stated linear transformation ("multiples of 2 with a linear transformation
  2n+1") maps the base set as tasks.hypotheses compose does, so the support is
  the image of the base set within 1..domain; a transformation given only as an
  example ("(e.g., 3n)") or without a form stays unresolved.
- Shifted and scaled bases ("one more than a perfect square", "twice a perfect
  square", "2^n - 1") replace the base set instead of conjoining with it; 0 is not
  a perfect square or cube, and n starts at 0 unless a bound on n is stated (a
  bound the grammar cannot apply leaves the description unresolved). A difference
  between digits is defined only for two-digit numbers.
- Explicit first-N lists and stated totals must agree with the sequence.
  Prime-factor counts distinguish multiplicity from distinct factors; asserting
  both conventions stays unresolved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Container
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from analysis.hypothesis_text import (
    NUMBER_TOKEN_RE,
    number_list_pattern,
    REMARK_MARKER,
    IGNORABLE_REMARK,
    EXAMPLE_CLAIM,
    GLUE_WORDS,
    VAGUE_MARKERS,
    normalize_struct_text,
    self_refuting,
    strip_boilerplate,
    library_label,
    extract_numbers,
    parse_number,
    set_literal,
    strip_outer_groups,
    top_level_matches,
    restore_operand_lists,
)
from analysis.hypothesis_terms import (
    Term,
    ORDINAL_PRIME_RE,
    ORDINAL_INTEGER_CLAUSE,
    parse_conjunctive_terms,
    MULTIPLICITY,
    parse_modulo_interval_or_union,
    has_exclusion_marker,
    parse_explicit_set,
    parse_bounded_form,
    parse_multiple_or_singleton_union,
    parse_only_singleton,
    parse_first_n_pattern,
    combine_terms,
    build_term,
    dedupe_terms,
    compile_expr,
    protect_operand_lists,
    split_or,
)


@dataclass(frozen=True)
class ParsedPredicate:
    kind: str
    label: str
    support: frozenset[int]
    expr: str


@dataclass(frozen=True)
class LibraryAddition:
    index: int
    label: str
    expr: str


@dataclass
class DomainIndexer:
    domain: int
    library_path: Path
    signature_to_index: dict[tuple[bool, ...], int]
    index_to_signature: dict[int, tuple[bool, ...]]
    additions: list[LibraryAddition]

    def index_for(self, predicate: ParsedPredicate) -> int:
        signature = _signature(predicate.support, self.domain)
        existing = self.signature_to_index.get(signature)
        if existing is not None:
            return existing
        index = _smallest_unused_index(self.index_to_signature)
        self.signature_to_index[signature] = index
        self.index_to_signature[index] = signature
        self.additions.append(LibraryAddition(index=index, label=predicate.label, expr=predicate.expr))
        return index


def structure_results(
    results_root: Path, *, task: str | None = None, dry_run: bool = False
) -> tuple[dict[int, Counter], Counter, dict[tuple[int, str, str], tuple[int, str]]]:
    """Return per-domain stats, dry-run changes vs. the on-disk structured files, and unresolved templates.

    Changes are keyed by (domain, old type, new type, old support size, new support size, hypothesis);
    an unresolved type reads "other:<reason>". Templates map (domain, reason, text with digit runs as "#")
    to (count, first raw text).
    """
    raw_paths = sorted(results_root.rglob("hypothesis_generation.json"))
    if task is not None:
        raw_paths = [path for path in raw_paths if _path_has_part(path, task)]
    indexers: dict[int, DomainIndexer] = {}
    outputs: list[tuple[Path, dict]] = []
    stats: dict[int, Counter] = {}
    changes: Counter = Counter()
    templates: dict[tuple[int, str, str], list] = {}  # key -> [count, first raw text]

    for raw_path in raw_paths:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        domain = int(payload["experiment_condition"]["domain"])
        if domain not in indexers:
            indexers[domain] = build_domain_indexer(domain)
        indexer = indexers[domain]
        counter = stats.setdefault(domain, Counter())
        structured = _structure_payload(payload, domain, indexer, counter)
        output_path = raw_path.with_name("hypothesis_generation_structured.json")
        outputs.append((output_path, structured))
        for items in structured["prefix_cache"].values():
            for item in items:
                if item["type"] == "other":
                    text = item.get("hypothesis", "")
                    key = (domain, item["reason"], re.sub(r"\d+", "#", normalize_struct_text(text)))
                    templates.setdefault(key, [0, text])[0] += 1
        if dry_run:
            if output_path.exists():
                old = json.loads(output_path.read_text(encoding="utf-8"))
                _diff_payload(old, structured, indexer, counter, changes)
            else:
                counter["unstructured_files"] += 1

    if not dry_run:
        writes = [
            (indexer.library_path, _library_text_with_additions(indexer.library_path, indexer.additions))
            for indexer in indexers.values()
            if indexer.additions
        ]
        writes.extend(
            (output_path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
            for output_path, payload in outputs
        )
        _write_text_transaction(writes)

    for domain, indexer in indexers.items():
        stats.setdefault(domain, Counter())["new_functions"] = len(indexer.additions)
        stats[domain]["files"] = sum(1 for path, _ in outputs if f"{os.sep}d{domain}{os.sep}" in str(path))
    return stats, changes, {key: tuple(entry) for key, entry in templates.items()}


def library_path(domain: int) -> Path:
    return Path(__file__).resolve().parent.parent / "library" / f"library_d{domain}.py"


@dataclass(frozen=True)
class HypothesisLibrary:
    """Executable generated-hypothesis predicates, by library index, as supports on 1..domain."""

    domain: int
    vectors: dict[int, np.ndarray]  # {index: [domain] 0/1 support}
    labels: dict[int, str]


def load_hypothesis_library(domain: int) -> HypothesisLibrary:
    path = library_path(domain)
    return _load_hypothesis_library_cached(str(path), path.stat().st_mtime_ns, domain)


@lru_cache(maxsize=None)
def _load_hypothesis_library_cached(path: str, mtime_ns: int, domain: int) -> HypothesisLibrary:
    """Execute library_dN.py once per file version; it is the source of generation semantics."""
    module = load_library_module(path, f"_number_game_library_d{domain}_{mtime_ns}")
    return HypothesisLibrary(
        domain=domain,
        vectors={
            int(index): np.asarray([float(bool(fn(x))) for x in range(1, domain + 1)])
            for index, fn in module.LIBRARY.items()
        },
        labels={int(index): fn.__doc__.strip().splitlines()[0] for index, fn in module.LIBRARY.items()},
    )


def build_domain_indexer(domain: int) -> DomainIndexer:
    index_to_signature = {
        index: tuple(bool(value) for value in vector)
        for index, vector in load_hypothesis_library(domain).vectors.items()
    }
    signature_to_index: dict[tuple[bool, ...], int] = {}
    for index, signature in index_to_signature.items():
        signature_to_index.setdefault(signature, index)
    return DomainIndexer(
        domain=domain,
        library_path=library_path(domain),
        signature_to_index=signature_to_index,
        index_to_signature=index_to_signature,
        additions=[],
    )


UNRESOLVED_REASONS = ("vague", "empty", "contradictory", "unparsed")


def structure_hypothesis(text: object, domain: int) -> ParsedPredicate | str:
    """Parse one generated hypothesis into an executable predicate, or name why it stays unresolved.

    Reasons come from how the parser fails: "vague" (no precise rule, or text that retracts
    itself), "contradictory" (a rule plus a remark or clause that disagrees with it), "empty"
    (a precise rule with no support in 1..domain), or "unparsed" (anything else the grammar
    cannot read, including an unreadable remark).
    """
    text = str(text)
    if _vague(normalize_struct_text(text)):
        return "vague"
    outcome = _parse_outcome(text, int(domain))
    if isinstance(outcome, str):
        return outcome
    # A known empty branch can participate in OR; only the complete hypothesis
    # must have executable support to enter the library.
    return outcome if outcome.support else "empty"


@lru_cache(maxsize=None)
def _parse_outcome(text: str, domain: int) -> ParsedPredicate | str:
    """The parsed predicate (support may be empty), or "contradictory" / "unparsed"."""
    try:
        parsed = _parse_hypothesis_text(text, domain)
    except _Contradiction:
        return "contradictory"
    except _Abstain:
        return "unparsed"
    return "unparsed" if parsed is None else parsed


def _parse_hypothesis_text_cached(text: str, domain: int) -> ParsedPredicate | None:
    """Sub-clause parse for the recursive readers: a predicate, or None for any failure."""
    outcome = _parse_outcome(text, domain)
    return None if isinstance(outcome, str) else outcome


def _parse_hypothesis_text(text: str, domain: int) -> ParsedPredicate | None:
    """Parse `text`; raises `_Abstain` (or `_Contradiction`) when the text is recognizably unstructurable."""
    raw = str(text).strip()
    normalized = normalize_struct_text(raw)
    if _vague(normalized):
        return None

    # Validate the first remark and parse the text without it; the recursion
    # handles any later remarks.
    remarks = _remarks(normalized)
    if remarks:
        start, end, remark = remarks[0]
        reading = _remark_reading(remark, domain)
        if reading is None:
            return None
        if not _remark_agrees(normalized, start, reading, domain):
            raise _Contradiction
        main = _parse_hypothesis_text_cached(normalized[:start].rstrip() + normalized[end:], domain)
        # Example lists after several quoted rules enumerate the observed data,
        # so the combined rule must accept every listed number.
        examples = [set(listed) for s, _, r in remarks if (listed := _listed_numbers(r)) and normalized[:s].endswith(("' ", '" '))]
        if main is not None and len(examples) >= 2 and not set().union(*examples) & set(range(1, domain + 1)) <= main.support:
            raise _Contradiction
        return None if main is None else ParsedPredicate(main.kind, library_label(raw), main.support, main.expr)

    rule = _rule_text(normalized)
    if rule is None:
        return None
    # Guards and the leftover-word check see commentary as part of the rule;
    # the clause-by-clause reading verifies it instead, so a guard's abstention
    # must fall through to it rather than propagate.
    try:
        parsed = _parse_hypothesis_candidate(raw, rule, domain)
    except _Abstain:
        parsed = None
    return _parse_with_commentary(raw, rule, domain) if parsed is None else parsed


def _vague(normalized: str) -> bool:
    return (
        not normalized or normalized in {"other", "unknown", "n/a"}
        or self_refuting(normalized) or any(marker in normalized for marker in VAGUE_MARKERS)
    )


class _Abstain(Exception):
    """A guard recognized a phrase that no parser can structure; the whole text stays unresolved."""


class _Contradiction(_Abstain):
    """The text states a rule and then disagrees with it (a narrower or conflicting remark, a false claim about listed numbers, or a framed restatement with a different support)."""


def _remark_reading(remark: str, domain: int) -> tuple[list[int] | None, ParsedPredicate | None] | None:
    """Return the (listed numbers, restated rule) a remark asserts, or None when it is unreadable.

    A claim about one example and a known non-narrowing note ("(inclusive)") assert nothing and read as (None, None); any other remark that neither lists numbers nor restates a parseable rule may narrow the rule, so it is unreadable.
    """
    if re.search(EXAMPLE_CLAIM, remark) or re.fullmatch(IGNORABLE_REMARK, remark):
        return None, None
    listed = _listed_numbers(remark)
    if listed is not None:
        return listed, None
    statement, _, members = remark.rpartition(":")
    listed = _listed_numbers(members) if statement else None
    restated = _parse_hypothesis_text_cached(_remark_statement(statement if listed is not None else remark), domain)
    return None if restated is None else (listed, restated)


def _remark_agrees(normalized: str, start: int, reading: tuple[list[int] | None, ParsedPredicate | None], domain: int) -> bool:
    """Return whether the remark reading at `start` agrees with the text before it.

    A remark explains either the nearest top-level operand or the whole prefix: listed numbers must lie in that set and a restated rule must cover it ("(i.e., multiples of 36: 36, 72)" checks both). When neither reads as a rule there is nothing to disagree with, so the text abstains instead.
    """
    listed, restated = reading
    if listed is None and restated is None:
        return True
    before = normalized[:start]
    protected = protect_operand_lists(before)
    connectives = top_level_matches(protected, r"\b(?:and|or)\b")
    nearest = restore_operand_lists(protected[connectives[-1].end():]) if connectives else before
    explained = [p for clause in (nearest, before) if (p := _parse_hypothesis_text_cached(clause, domain))]
    if not explained:
        raise _Abstain
    members = set() if listed is None else {x for x in listed if 1 <= x <= domain}
    return any(members <= p.support and (restated is None or p.support <= restated.support) for p in explained)


def _remark_statement(remark: str) -> str:
    # Notation-only restatements ("(2^k)", "(7k+1)", "(1 to 100)", "(same as 84)") become parseable descriptions.
    remark = re.sub(r"\b(\d+)\s*\*\s*(\d+)\b", lambda m: str(int(m.group(1)) * int(m.group(2))), remark)
    remark = re.sub(r"\b(\d+)\s*\^\s*(\d)\b", lambda m: str(int(m.group(1)) ** int(m.group(2))), remark)
    remark = re.sub(r"^(?:which is |that is )?(?:the same as|same as|equivalent to|rephrased as)\s+(?:being\s+)?", "", remark)
    # "(2 mod 4 = 2)" names a residue class only when its arithmetic holds.
    if (mod := re.fullmatch(r"(\d+)\s+mod\s+(\d+)(?:\s*=\s*(\d+))?", remark)) and int(mod.group(2)):
        a, m = int(mod.group(1)), int(mod.group(2))
        return f"numbers congruent to {a} mod {m}" if mod.group(3) is None or a % m == int(mod.group(3)) else remark
    for pattern, replacement in (
        (r"(?:yielding |interval |range |in the range )?(\d+)\s*(?:to|-)\s*(\d+)", r"numbers from \1 to \2"),
        (r"(\d+)", r"multiples of \1"),
        (r"(\d+)\s*\^\s*[a-z]", r"powers of \1"),
        (r"[a-z]\s*\^\s*2|[a-z] squared", "perfect squares"),
        (r"[a-z]\s*\^\s*3|[a-z] cubed", "perfect cubes"),
        (r"(\d+)\s*\*?\s*[a-z]\s*([+-])\s*(\d+)", r"numbers of the form \1n\2\3"),
        (r"(\d+)\s*\*?\s*[a-z]", r"multiples of \1"),
    ):
        if re.fullmatch(pattern, remark):
            return re.sub(pattern, replacement, remark)
    return remark


def _parse_hypothesis_candidate(raw: str, normalized: str, domain: int) -> ParsedPredicate | None:
    # The rule text arrives pre-normalized from _rule_text after the
    # empty/self-refuting/vague guards.
    for pattern, build in _WHOLE_RULES:
        match = re.fullmatch(pattern, normalized)
        if match:
            return build(match, raw, domain)
    exact = _parse_exact(raw, normalized, domain)
    if exact is not None:
        return exact
    if _abstains(normalized):
        return None
    stripped = strip_boilerplate(normalized)
    parts = split_or(stripped)
    if parts is None:
        return _parse_grouped_and(raw, stripped, domain)
    if len(parts) > 1:
        predicates = [_parse_hypothesis_text_cached(part, domain) for part in parts]
        if all(predicates):
            terms = [Term(p.support, p.expr, interval=p.kind == "interval") for p in predicates if p is not None]
            return _predicate(raw, [combine_terms(terms, "or")], domain, forced_kind="rule")
        return None
    return _parse_conjunction(raw, stripped, normalized, domain)


def _parse_exact(raw: str, text: str, domain: int) -> ParsedPredicate | None:
    # Returns the first exact parse or None; a guard raises _Abstain. The union,
    # singleton, and first-N parsers search rather than fullmatch, so the guards
    # between parsers must keep their positions.
    if re.search(r"\bdistinct prime factors?\b", text) and "multiplicity" in text:
        raise _Abstain
    multiple_union = parse_multiple_or_singleton_union(text, domain)
    if multiple_union is not None:
        return _predicate(raw, [multiple_union], domain, forced_kind="rule")
    simple_exclusion = _parse_simple_exclusion(raw, text, domain)
    if simple_exclusion is not None:
        return simple_exclusion
    if has_exclusion_marker(text):
        raise _Abstain
    finite = parse_explicit_set(text, domain)
    if finite is not None:
        return _predicate(raw, [finite], domain, forced_kind="rule")
    only = parse_only_singleton(text, domain)
    if only is not None:
        return _predicate(raw, [only], domain, forced_kind="interval")
    # Procedural output conditions and numerical commentary cannot become
    # extra conjuncts inside just one side of an OR.
    if re.search(r"\bor\b", text) and re.search(r"\band (?:the program )?(?:only )?(?:outputs?|says?|returns?)\s+['\"]?yes\b", text):
        raise _Abstain
    if re.search(r"\bor\b", text) and re.search(r"[.,]\s+\d+\s+is\s+(?:not\s+)?(?:a\s+)?(?:divisible|multiple)\b", text):
        raise _Abstain
    first_n = parse_first_n_pattern(text, domain)
    if first_n is not None:
        return _predicate(raw, [first_n], domain, forced_kind="rule")
    bounded_form = parse_bounded_form(text, domain)
    if bounded_form is not None:
        return _predicate(raw, [bounded_form], domain, forced_kind="rule")
    return None


# Phrases the term parsers would misread or drop; matching one abstains. They
# run after the exact parsers because those own phrases like "first 5 odd
# numbers" or "squares of integers from 1 to 9".
_ABSTAIN_PATTERNS: tuple[str, ...] = (
    # A sequence prefix the first-N parser did not resolve is not a property of x.
    rf"\bfirst\s+({NUMBER_TOKEN_RE.pattern})\s+(?:(?:perfect\s+)?(?:squares?|cubes?)|primes?|multiples?|even|odd)\b",
    # Bounds on a root or on one digit position are not bounds on x.
    (
        r"\b(?:squares?|cubes?) of (?:the )?(?:positive )?(?:integers?|numbers?)\s+"
        r"(?:from|between|under|over|below|above|less|greater|more|up to|at most|at least|\d)"
    ),
    (
        rf"\b(?:(?:first|last|leading|tens|units?|ones|hundreds) digit\s+(?:is\s+)?|digits\s+(?:are\s+)?)"
        rf"(?:(?:from\s+)?(?:{NUMBER_TOKEN_RE.pattern})\s*(?:to|through|-)|between\s+(?:{NUMBER_TOKEN_RE.pattern})\s+and)\s*(?:{NUMBER_TOKEN_RE.pattern})"
    ),
    # A digit class joined to a value ("digit is even and equal to 4") has no term.
    r"\bdigit (?:is|must be) (?:even|odd|prime) and (?:is )?(?:equal to|equals?)\b",
    # Division of the variable has no term.
    r"\b[nx]\s*/\s*\d+",
)


def _abstains(text: str) -> bool:
    return (
        any(re.search(pattern, text) for pattern in _ABSTAIN_PATTERNS)
        or _unresolved_ordinal_prime(text)
        or _non_integer_root(text)
        or _ambiguous_parity_count(text)
    )


def _unresolved_ordinal_prime(text: str) -> bool:
    # An ordinal beside "prime" parses only as a whole "3rd and 5th primes" list or an ordinal-integer clause.
    return (
        "prime" in text and bool(re.search(r"\b\d+(?:st|nd|rd|th)\b", text))
        and not ORDINAL_PRIME_RE.fullmatch(text) and not re.search(ORDINAL_INTEGER_CLAUSE, text)
    )


def _non_integer_root(text: str) -> bool:
    # Root clauses other than "has an integer square root" constrain the root, not x.
    if not re.search(r"\b(?:square|cube) roots?\b", text):
        return False
    return not re.fullmatch(
        r"(?:numbers? )?(?:with (?:an? )?(?:(?:odd|even|positive) )?(?:integer|whole number) (?:square|cube) roots?"
        r"|whose (?:square|cube) roots? (?:is|are) (?:an? )?(?:(?:odd|even|positive) )?(?:integers?|whole numbers?))",
        text,
    )


def _ambiguous_parity_count(text: str) -> bool:
    # Only digit, tens, divisor, binary-one, and counted-multiplicity prime-factor
    # counts have parity terms; a bare prime-factor count is ambiguous.
    for count in re.finditer(r"\b(?:even|odd) number of (?!the form\b)(?:(?:positive|total) )?(\w+)", text):
        if count.group(1) in {"1s", "ones", "set"} and "binary" in text:
            continue
        if count.group(1) == "prime" and re.match(rf"\s*factors{MULTIPLICITY}", text[count.end():]):
            continue
        if count.group(1) not in {"divisors", "factors", "digits", "tens"} or "binary" in text:
            return True
        if count.group(1) == "digits" and re.search(r"\b(?:base|octal|hexadecimal|hex)\b", text):
            return True
        if count.group(1) in {"divisors", "factors"} and "multiplicity" in text:
            return True
    return False


def _parse_grouped_and(raw: str, stripped: str, domain: int) -> ParsedPredicate | None:
    # Explicit grouping fixes conjunction scope without splitting operand
    # lists such as "divisible by 3 and 7".
    grouped = strip_outer_groups(stripped)
    for match in top_level_matches(grouped, r"\band\b") or ():
        sides = [grouped[:match.start()].strip(), grouped[match.end():].strip()]
        if not any(strip_outer_groups(side) != side for side in sides):
            continue
        predicates = [_parse_hypothesis_text_cached(side, domain) for side in sides]
        if all(predicates):
            terms = [Term(p.support, p.expr) for p in predicates if p is not None]
            return _predicate(raw, terms, domain, forced_kind="rule")
        return None
    return None


def _parse_conjunction(raw: str, stripped: str, normalized: str, domain: int) -> ParsedPredicate | None:
    # Must run after split_or: an OR branch such as "prime plus 2" recurses
    # into _WHOLE_RULES, where shifted_prime owns it.
    if re.search(rf"\bprime(?: number)?\s+(?:plus|minus|\+|-)\s+(?:{NUMBER_TOKEN_RE.pattern})|(?:{NUMBER_TOKEN_RE.pattern})\s*\*?\s*n,? where n is (?:a )?prime\b", normalized):
        return None
    source = stripped
    terms = parse_conjunctive_terms(source, domain)
    if not terms and stripped != normalized:
        source = normalized
        terms = parse_conjunctive_terms(source, domain)
    or_union = parse_modulo_interval_or_union(stripped, domain)
    if or_union is not None:
        return _predicate(raw, [or_union], domain, forced_kind="rule")
    if not terms:
        return None
    if _unaccounted_words(source, terms):
        return None
    return _predicate(raw, dedupe_terms(terms), domain)


def _unaccounted_words(source: str, terms: list[Term]) -> list[str]:
    """Return the digits and content words of `source` that no term span or remark covers.

    Remarks that survived validation only explain the text before them; any leftover is a dropped condition or commentary.
    """
    residual = list(source)
    for start, end in [term.span for term in terms if term.span] + [remark[:2] for remark in _remarks(source)]:
        residual[start:end] = " " * (end - start)
    rest = re.sub(r"\b\d+(?:st|nd|rd|th)\b|\b\d0s\b", " ", "".join(residual))
    return [word for word in re.findall(r"[a-z]+|\d+", rest) if word not in GLUE_WORDS]


def _build_output_condition(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    before = _parse_hypothesis_text_cached(match.group(1), domain)
    constraint = re.sub(r"^it\b", "the number", match.group(2))
    constraint = re.sub(r"^those\b", "numbers", constraint)
    after = _parse_hypothesis_text_cached(constraint, domain)
    if before is None or after is None:
        return None
    terms = [Term(p.support, p.expr, p.kind == "interval") for p in (before, after)]
    kind = "interval" if any(p.kind == "interval" and len(p.support) == 1 for p in (before, after)) else None
    return _predicate(raw, terms, domain, forced_kind=kind)


def _build_shifted_prime(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    offset = parse_number(match.group(2))
    if offset is None:
        return None
    if match.group(1) in {"minus", "-"}:
        offset = -offset
    return _predicate(raw, [build_term(f"_is_prime(x - ({offset}))", domain)], domain, forced_kind="rule")


def _build_coefficient_prime(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    coefficient = parse_number(match.group(1))
    if coefficient is None or coefficient <= 0:
        return None
    return _predicate(raw, [build_term(f"x % {coefficient} == 0 and _is_prime(x // {coefficient})", domain)], domain, forced_kind="rule")


def _build_form_where(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    # A condition on the variable of a stated form ("2^k where k >= 5") constrains
    # that variable; x ranges over the image of the form.
    head, cond = match.group("head", "cond")
    # "perfect square (n^2 where n is an integer)" closes the form's parenthetical after the condition.
    if head.count("(") > head.count(")") and cond.endswith(")"):
        head, cond = head + ")", cond[:-1].rstrip(" .")
    # In "divisible by 2^k where ...", x is a multiple of the form, not its image.
    if re.search(r"\b(?:divisible by|multiples? of)\s+\S+$", head):
        return None
    base = _parse_hypothesis_text_cached(head, domain)
    image = _parse_form_image(match, cond, domain)
    if base is None or image is None:
        return None
    return _predicate(raw, [Term(base.support, base.expr), image], domain, forced_kind="rule")


def _build_linear_image(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    base = _parse_hypothesis_text_cached(strip_outer_groups(match.group("head")), domain)
    if base is None:
        return None
    coef = int(match.group("coef") or 1)
    offset = int(match.group("offset") or 0) * (-1 if match.group("sign") == "-" else 1)
    # The base expr is written over x, so the lambda binds x to the preimage, which must lie in the domain.
    expr = f"(x - ({offset})) % {coef} == 0 and (lambda x: 1 <= x <= {domain} and ({base.expr}))((x - ({offset})) // {coef})"
    return _predicate(raw, [build_term(expr, domain)], domain, forced_kind=base.kind if coef == 1 else "rule")


def _build_total(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    count = parse_number(match.group(2))
    sequence = parse_first_n_pattern(match.group(1), domain, total=count)
    if sequence is not None:
        return _predicate(raw, [sequence], domain, forced_kind="rule")
    if re.search(r"\bfirst\b", match.group(1)):
        return None
    base = _parse_hypothesis_text_cached(match.group(1), domain)
    if base is None or count != len(base.support):
        return None
    return ParsedPredicate(base.kind, library_label(raw), base.support, base.expr)


def _build_exclusion(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    base = _parse_hypothesis_text_cached(match.group(1), domain)
    if base is None:
        return None
    excluded = build_term(f"x not in {set_literal(extract_numbers(match.group('nums')))}", domain)
    return _predicate(raw, [Term(base.support, base.expr, base.kind == "interval"), excluded], domain)


def _build_added_numbers(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    base = _parse_hypothesis_text_cached(match.group(1), domain)
    if base is None:
        return None
    extra = build_term(f"x in {set_literal(extract_numbers(match.group('nums')))}", domain)
    return _predicate(raw, [combine_terms([Term(base.support, base.expr), extra], "or")], domain, forced_kind="rule")


def _build_named_number(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    # "The program accepts the number 30" reports one accepted example; only "accepts only ..." names a one-number rule.
    value = parse_number(match.group(1))
    if value is None or not 1 <= value <= domain or re.search(r"\baccepts?\b", normalize_struct_text(raw)):
        return None
    return _predicate(raw, [build_term(f"x == {value}", domain, interval=True)], domain)


def _build_integer_subset(match: re.Match[str], raw: str, domain: int) -> ParsedPredicate | None:
    count = parse_number(match.group("count"))
    condition = _parse_hypothesis_text_cached(match.group("condition"), domain)
    if count is None or condition is None:
        return None
    return _predicate(raw, [Term(condition.support, condition.expr), build_term(f"1 <= x <= {count}", domain, interval=True)], domain)


# Whole-text rules, tried in order: the first pattern that fullmatches the
# rule text owns it, and its builder's None abstains for the whole text.
_WHOLE_RULES: tuple[tuple[str, Callable[[re.Match[str], str, int], ParsedPredicate | None]], ...] = (
    # OR needs the whole left side grouped before applying the output condition;
    # decimal divisors are handled by an exact bounded-form parser below.
    (
        r"(?=.*(?:\bor\b|\d+\.\d+))(.+?)\s*,?\s+and (?:the program )?(?:only )?(?:outputs?|says?|returns?) ['\"]?yes['\"]? "
        r"(?:only )?(?:if|for) (.+?)(?:, otherwise ['\"]?no['\"]?)?",
        _build_output_condition,
    ),
    (
        rf"(?:(?:the )?number is |numbers? (?:equal to|of the form) )?(?:a )?prime(?: number)?\s+(plus|minus|\+|-)\s+({NUMBER_TOKEN_RE.pattern})",
        _build_shifted_prime,
    ),
    (
        rf"(?:numbers? (?:of the form )?)?({NUMBER_TOKEN_RE.pattern})\s*\*?\s*n,? where n is (?:a )?prime(?: number)?",
        _build_coefficient_prime,
    ),
    # "<base> with a linear transformation 2n+1" is the task's transform (tasks.hypotheses.compose):
    # x ranges over the image of the base set, not over base members of that form.
    (
        r"(?P<head>.+?)\s+with an? linear transformation(?::| of)?\s+"
        r"(?:(?P<coef>\d+)\s*\*?\s*)?[a-z]\b(?:\s*(?P<sign>[+-])\s*(?P<offset>\d+))?\.?",
        _build_linear_image,
    ),
    # A linear (7k+1), power (2^k), or root (n^2) form directly before "where <var> ...",
    # where <var> is the form's own variable.
    (
        r"(?P<head>.*?\b(?:(?P<coef>\d+)(?:\s*\*\s*)?(?P<var>[a-z])\b(?:\s*(?P<sign>[+-])\s*(?P<offset>\d+))?"
        r"|(?P<power_base>\d+)\s*\^\s*(?P<power_var>[a-z])\b|(?P<root_var>[a-z])\s*\^\s*(?P<exponent>\d+))\)?)"
        r"\s*,?\s+where\s+(?P<where_var>(?(var)(?P=var)|(?(power_var)(?P=power_var)|(?P=root_var))))\b\s*(?P<cond>.+?)\.?",
        _build_form_where,
    ),
    # A stated total is a constraint on the complete set, not a number to add.
    (rf"(.+),\s*totall?ing\s+({NUMBER_TOKEN_RE.pattern})\s+(?:numbers|integers)", _build_total),
    (rf"(.+?)\s+(?:exclusive of|excluding)\s+(?P<nums>{number_list_pattern()})", _build_exclusion),
    (rf"(.+?)\s+plus (?:the )?(?:specific )?numbers?\s+(?P<nums>{number_list_pattern()})", _build_added_numbers),
    # A whole rule naming one number ("the number 42").
    (rf"(?:the )?(?:specific )?number ({NUMBER_TOKEN_RE.pattern})", _build_named_number),
    (
        rf"(?:the )?(?:numbers? are )?(?:the )?first\s+(?P<count>{NUMBER_TOKEN_RE.pattern})\s+"
        rf"(?:positive )?integers?(?:\s*\(1 to (?P=count)\))?\s+"
        rf"(?:that are|that satisfy the condition)\s+(?P<condition>.+)",
        _build_integer_subset,
    ),
)


def _rule_text(text: str) -> str | None:
    """Return the part of `text` that states the rule, or None when that part is ambiguous.

    There is one rule text: once it fails, falling back to a longer text would let a
    parser read the rule out of the justification ("... because they are the first two even numbers").
    """
    quoted_rule = re.search(
        r"(?:rule is\b|rule:|checks? if\b|checks? whether\b)\s*(['\"])(?P<body>.+?)\1",
        text,
        flags=re.IGNORECASE,
    )
    if quoted_rule:
        # Text before the quoted rule must be framing, not a rival rule
        # ("the number is 46 because ..., so the rule is '...'").
        if set(re.findall(r"[a-z]+|\d+", strip_boilerplate(text[:quoted_rule.start()]))) - GLUE_WORDS:
            return None
        # Parentheticals still on the tail are unparseable notes on the quoted rule;
        # drop them unless they may narrow it ("(exclusive of 59)", "(and ...)").
        tail = strip_boilerplate(text[quoted_rule.end():])
        notes = re.match(r"(?:\s*\([^()]*\)\s*[.,;:]?)+", tail)
        if notes and not re.search(r"\b(?:not|except|excluding|exclusive|strictly)\b|\(\s*(?:and|or)\b", notes.group(0)):
            tail = tail[notes.end():]
        tail = tail.strip(" .;:,?!")
        if not tail:
            return _clean_rule(quoted_rule.group("body"))
        constraint = strip_outer_groups(tail)
        within = re.match(r"(?:(?:with)?in|restricted to) (?P<range>(?:the )?(?:range|interval)\b.*)", constraint)
        if re.match(r"(?:and|or|with an? linear transformation)\b", constraint):
            return _clean_rule(f"({quoted_rule.group('body')}) {constraint}")
        if within:
            return _clean_rule(f"({quoted_rule.group('body')}) and numbers in {within.group('range')}")
        # Any other tail is commentary that _parse_with_commentary must verify.
        return _clean_rule(f"({quoted_rule.group('body')}); {constraint}")
    # A trailing "and says 'yes' for these" restates the condition.
    text = re.sub(
        r"\s*,?\s+and (?:the program )?(?:outputs?|says?|returns?) ['\"]?yes['\"]? "
        r"(?:if it is|if true|if it (?:is|matches) either|for (?:these|those|(?:any|all) of them))"
        r"(?:, otherwise ['\"]?no['\"]?)?$", "", text,
    )
    justification = re.search(r" because | since | explanation:| examples:", text)
    return _clean_rule(text[:justification.start()] if justification else text)


def _clean_rule(text: str) -> str:
    return strip_boilerplate(normalize_struct_text(text))


# Sentence ends (not "e.g." or an ellipsis), semicolons, clause connectives, and a comma before a numeric claim.
_CLAUSE_BOUNDARY = (
    r"\s*;\s*|(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\.\.)\.\s+"
    r"|,?\s+(?:and|but|while|whereas|so|thus|hence|therefore)\s+|,\s+(?=-?\d)"
)
_NUMBER_CLAIM = rf"(?:(?:since|as|because|also|then)\s+)?(?:the (?:numbers?|inputs?|examples?) )?(?P<nums>{number_list_pattern()})\s+(?:is|are)\s+(?P<pred>.+)"
_ARITHMETIC_CLAIM = r"[\s+*()-]*\d[\d\s+*()-]*(?:=[\s+*()-]*\d[\d\s+*()-]*)+"
_MEMBER_PRED = r"true|yes|valid|accepted|included|consistent|in range|in (?:the|this|that) (?:set|list|range|sequence|interval)|(?:a )?valid (?:example|input|member|multiple|number)s?"
_NON_MEMBER_PRED = r"false|no|invalid|excluded|rejected|not (?:valid|included|accepted|in range|in (?:the|this|that) (?:set|list|range|sequence|interval))"
_CONDITION_START = (
    r"(?:the (?:number|input)|it|x|n|is|are|must|has|have|with|whose|where|that are|less|greater|more|fewer|at (?:least|most)"
    r"|between|under|over|above|below|up to|within|in (?:the )?(?:range|interval)|from|divisible|a multiple|multiple"
    r"|ending|ends?|starting|starts?|containing|contains?|even|odd|prime|composite|not|a perfect|perfect|a power)\b"
)
# A clause about the rule itself rather than about x; it may only restate the rule.
_FRAMING = r"(?:the program|program|this|that|which|the rule|the hypothesis|(?:only )?(?:accepts|outputs|says|returns))\b"
# Closed list of commentary that says nothing about which numbers are accepted.
_META_COMMENT = (
    r"(?:this|that|it|the (?:rule|hypothesis))\s+(?:fits|explains|matches|covers|works for|is consistent with|satisfies|accounts for)"
    r"(?: (?:all|both))?(?: (?:the|these|those))?(?: (?:data|examples?|numbers|inputs|cases|observations))?(?: (?:perfectly|well|exactly))?"
    r"|the (?:rule|hypothesis) is (?:true|correct|consistent|valid|confirmed)"
    r"|(?:this is )?(?:a )?(?:simple|strong|plausible|natural|common|good|reasonable) (?:rule|hypothesis|candidate|pattern|explanation)"
    r"|confidence:? [\d.]+%?"
    r"|(?:says|outputs|returns) '?yes'? (?:if|for|when) (?:it is )?(?:one of |any of )?(?:these|those|the listed|the above)(?: specific)? (?:values|numbers|cases)"
)


def _parse_with_commentary(raw: str, rule: str, domain: int) -> ParsedPredicate | None:
    """Parse `rule` as a rule clause followed by commentary clauses, verifying each comment.

    Runs only when `rule` does not parse as a whole. After the first clause, each clause must be a true claim about listed numbers ("65 is false", "69 is a multiple of 3", "16 = 4*4"), a framed restatement with the same support ("the program checks divisibility by 6"), closed-list meta commentary, or a further condition on x. A restatement implied by the rule leaves its support unchanged, so ANDing a condition and accepting a restatement agree; framing only marks clauses that must not change the rule.
    """
    protected = protect_operand_lists(rule)
    boundaries = top_level_matches(protected, _CLAUSE_BOUNDARY)
    if not boundaries:
        return None
    cuts = [0, *(index for match in boundaries for index in match.span()), len(protected)]
    clauses = []
    for start, end in zip(cuts[::2], cuts[1::2]):
        clause = restore_operand_lists(protected[start:end]).strip(" ,")
        # Rejoin a claim split inside its number list ("46 and 76 are even").
        if len(clauses) > 1 and re.fullmatch(number_list_pattern(), clauses[-1]) and re.match(_NUMBER_CLAIM, clause):
            clause = f"{clauses.pop()} and {clause}"
        clauses.append(clause)
    head = _parse_hypothesis_text_cached(strip_outer_groups(clauses[0]), domain)
    if head is None:
        return None
    conditions = [head]
    support = head.support
    for clause in clauses[1:]:
        if not clause or re.fullmatch(_META_COMMENT, clause):
            continue
        if claim := re.fullmatch(_NUMBER_CLAIM, clause):
            if not _claim_holds(claim.group("nums"), claim.group("pred"), support, domain):
                raise _Contradiction
            continue
        if re.fullmatch(_ARITHMETIC_CLAIM, clause):
            # Digits, operators, and parentheses only, so eval cannot reach names.
            if len({eval(side) for side in clause.split("=")}) != 1:
                raise _Contradiction
            continue
        framed = re.match(_FRAMING, clause)
        # A condition reads as a predicate on x; "and numbers ending in 7" may name a
        # second set (a union) and "only 23 and 62 are ..." is about the examples.
        if not framed and not re.match(_CONDITION_START, clause):
            return None
        parsed = _parse_hypothesis_text_cached(_restatement(clause) if framed else clause, domain)
        if parsed is None:
            return None
        if framed:
            # A framed clause restates the rule; a broader one ("the number is 46 and the
            # program accepts multiples of 2") is a second, rival rule.
            if parsed.support != support:
                raise _Contradiction
            continue
        # AND binds tighter than a top-level OR in either clause; the grouping is ambiguous.
        if any(top_level_matches(protect_operand_lists(text), r"\bor\b") for text in (clauses[0], clause)):
            return None
        conditions.append(parsed)
        support = support & parsed.support
    if len(conditions) == 1:
        return ParsedPredicate(head.kind, library_label(raw), head.support, head.expr)
    return _predicate(raw, [Term(p.support, p.expr, p.kind == "interval") for p in conditions], domain)


def _claim_holds(nums: str, pred: str, support: frozenset[int], domain: int) -> bool:
    values = extract_numbers(nums)
    if re.fullmatch(_MEMBER_PRED, pred):
        return all(v in support for v in values if 1 <= v <= domain)
    if re.fullmatch(_NON_MEMBER_PRED, pred):
        return not any(v in support for v in values if 1 <= v <= domain)
    claimed = _parse_hypothesis_text_cached(f"numbers that are {pred}", domain)
    # An unreadable claimed property is a grammar gap, not a disagreement.
    if claimed is None:
        raise _Abstain
    # The claim may concern a number outside the domain, so evaluate the expression.
    holds = compile_expr(claimed.expr)
    return all(holds(v) for v in values)


def _restatement(clause: str) -> str:
    clause = re.sub(r"^(?:the )?program\s+(?:checks|tests)\s+(?:for\s+)?divisibility by\b", "divisible by", clause)
    clause = re.sub(r"^(?:(?:the )?program\s+)?(?:only\s+)?(?:says|outputs|returns|accepts)\s+(?:'?yes'?\s+)?(?:for|to|if|when)?\s*", "", clause)
    clause = re.sub(r"^(?:which|this|that|the rule|the hypothesis)\s+(?:is|means)\s+(?:that\s+)?", "", clause)
    return _remark_statement(clause)


def _remarks(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, body) of explanatory remarks in `text`, in order.

    A top-level parenthetical is a condition, not a remark, when it starts the
    text, is joined by a Boolean connective, or starts with "and" (an "(or ...)"
    body is a restatement); "x^(2)"-style notation and a form
    whose variable a following "where" binds ("(31n) where n >= 2") are not remarks.
    """
    remarks = []
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                opened = index
            depth += 1
        elif char == ")" and depth:
            depth -= 1
            before = text[:opened].rstrip(" ,") if depth == 0 else ""
            body = text[opened + 1:index]
            if (
                before and text[opened - 1] == " " and not re.search(r"\b(?:and|or)$", before)
                and not re.match(r"\s*and\b", body) and not re.match(r"\s*,?\s+where\b", text[index + 1:])
                # "(counting multiplicity)" is notation the factor-count terms consume.
                and not re.fullmatch(r"\s*(?:counting|counted with|including|with) (?:multiplicity|duplicates)\s*", body)
            ):
                remarks.append((opened, index + 1, body))
    for marker in top_level_matches(text, r"[,;]?\s+i\.e\.|[,;]\s*(?:that is|equivalently)\b|[,;]?\s+which includes\b") or ():
        remarks.append((marker.start(), len(text), text[marker.end():]))
    return [(start, end, re.sub(REMARK_MARKER, "", body).strip()) for start, end, body in sorted(remarks)]


def _listed_numbers(remark: str) -> list[int] | None:
    # "(47, 2209 - outside range)" lists members, then notes which fall in the domain.
    remark = re.sub(r"\s*-?\s*none in (?:the )?range$|\s*-\s*(?:none|only|outside|out of)\b[^()]*$", "", remark)
    if re.fullmatch(r"(?:\d++|[\s,;]|\.\.\.|…|\b(?:and|or)\b|\betc\b\.?)+", remark) and re.search(r"\d", remark):
        return [int(n) for n in re.findall(r"\d+", remark)]
    return None


def _structure_payload(payload: dict, domain: int, indexer: DomainIndexer, stats: Counter) -> dict:
    structured = {key: value for key, value in payload.items() if key != "prefix_cache"}
    prefix_cache = {}
    for prefix, items in payload["prefix_cache"].items():
        structured_items = []
        for item in items:
            stats["items"] += 1
            predicate = structure_hypothesis(item.get("hypothesis", ""), domain)
            next_item = dict(item)
            if isinstance(predicate, str):
                reason = predicate
                next_item["type"] = "other"
                next_item["reason"] = reason
                next_item.pop("index", None)
                stats["other"] += 1
                stats[f"other_{reason}"] += 1
            else:
                next_item["type"] = predicate.kind
                next_item["index"] = indexer.index_for(predicate)
                stats[predicate.kind] += 1
            structured_items.append(next_item)
        prefix_cache[prefix] = structured_items
    structured["prefix_cache"] = prefix_cache
    return structured


def _diff_payload(old: dict, new: dict, indexer: DomainIndexer, stats: Counter, changes: Counter) -> None:
    # Compare supports, not indices: indices of queued additions depend on write order.
    for prefix, new_items in new["prefix_cache"].items():
        for old_item, new_item in zip(old["prefix_cache"][prefix], new_items, strict=True):
            old_signature, new_signature = (
                None if item["type"] == "other" else indexer.index_to_signature[item["index"]] for item in (old_item, new_item)
            )
            if old_signature == new_signature:
                stats["unchanged"] += 1
                continue
            if old_signature is None:
                stats["other_to_parsed"] += 1
            elif new_signature is None:
                stats["parsed_to_other"] += 1
            else:
                stats["support_changed"] += 1
            old_size = sum(old_signature or ())
            new_size = sum(new_signature or ())
            old_type = old_item["type"]
            new_type = f"other:{new_item['reason']}" if new_item["type"] == "other" else new_item["type"]
            changes[(indexer.domain, old_type, new_type, old_size, new_size, new_item["hypothesis"])] += 1


def _parse_simple_exclusion(raw: str, text: str, domain: int) -> ParsedPredicate | None:
    patterns = (
        rf"(?:the\s+)?numbers?\s+(?:is\s+|are\s+)?not\s+(?:equal\s+to\s+)?(?P<nums>{number_list_pattern()})",
        rf"(?:the\s+)?numbers?\s+(?:is\s+|are\s+)?not\s+(?:the\s+)?numbers?\s+(?P<nums>{number_list_pattern()})",
        rf"numbers?\s+not\s+equal\s+to\s+(?P<nums>{number_list_pattern()})",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            nums = [n for n in extract_numbers(match.group("nums")) if 1 <= n <= domain]
            if nums:
                return _predicate(raw, [build_term(f"x not in {set_literal(nums)}", domain)], domain, forced_kind="rule")
    return None


def _parse_form_image(match: re.Match[str], cond: str, domain: int) -> Term | None:
    clauses = []
    # "and the result is less than 200" bounds x.
    result = re.search(r",?\s+and the result is (.+)$", cond)
    if result:
        cond = cond[:result.start()]
        bound = _parse_hypothesis_text_cached(f"numbers {result.group(1)}", domain)
        if bound is None:
            return None
        clauses.append(bound.expr)

    if re.search(r"\.\.|…|\betc\b", cond):
        return None
    cond = re.sub(rf"\b{match.group('where_var')}\b", "n", cond)
    kind = re.match(r"is (?:an? |any )?(?:(?P<sign>non-?negative|positive) )?(?P<noun>integer|whole number|natural number)s?\b,?\s*", cond)
    variable = []
    if kind:
        variable.append("x >= 1" if kind.group("sign") == "positive" or kind.group("noun") == "natural number" else "x >= 0")
        cond = cond[kind.end():]
    cond = re.sub(r"^(?:is )?(?:(?:with)?in )?(\d+)\s*(?:-|to)\s*(\d+)\b", r"is from \1 to \2", cond)
    listed = re.fullmatch(rf"is (?P<nums>{number_list_pattern()})", cond)
    if listed:
        variable.append(f"x in {set_literal(extract_numbers(listed.group('nums')))}")
    elif cond:
        # A trailing clause ("is prime, with an unknown restriction") is not a variable condition.
        if re.search(r"[,;?](?![^\[]*\])", cond):
            return None
        parsed = _parse_hypothesis_text_cached(f"n {cond}", domain)
        if parsed is None:
            return None
        variable.append(parsed.expr)
    # The sub-parsed exprs are written over x, so the lambda binds x to the form's variable.
    on_variable = f"(lambda x: {' and '.join(f'({e})' for e in variable) or 'True'})"

    if match.group("power_var"):
        expr = f"any({match.group('power_base')} ** n == x and {on_variable}(n) for n in range(0, x.bit_length() + 1))"
    elif match.group("root_var"):
        expr = f"any(n ** {match.group('exponent')} == x and {on_variable}(n) for n in range(0, x + 1))"
    else:
        coef = int(match.group("coef"))
        offset = int(match.group("offset") or 0) * (-1 if match.group("sign") == "-" else 1)
        expr = f"x - ({offset}) >= 0 and (x - ({offset})) % {coef} == 0 and {on_variable}((x - ({offset})) // {coef})"
    return build_term(" and ".join([expr, *clauses]), domain)


def _predicate(raw_label: str, terms: list[Term], domain: int, forced_kind: str | None = None) -> ParsedPredicate | None:
    if not terms:
        return None
    combined = terms[0] if len(terms) == 1 else combine_terms(terms, "and")
    kind = forced_kind or ("interval" if all(term.interval for term in terms) else "rule")
    return ParsedPredicate(
        kind=kind,
        label=library_label(raw_label),
        support=frozenset(x for x in combined.support if 1 <= x <= domain),
        expr=combined.expr,
    )


def _path_has_part(path: Path, part: str) -> bool:
    return any(piece.lower() == part.lower() for piece in path.parts)


def load_library_module(path: str | Path, name: str):
    """Import a library_dN.py file as a fresh module, never registered in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _library_text_with_additions(path: Path, additions: list[LibraryAddition]) -> str:
    text = path.read_text(encoding="utf-8").rstrip() + "\n\n"
    blocks = []
    for addition in additions:
        fn_name = f"fn_{addition.index:04d}"
        doc = json.dumps(addition.label, ensure_ascii=True)
        blocks.append(
            f"@register({addition.index})\n"
            f"def {fn_name}(x: int) -> bool:\n"
            f"    {doc}\n"
            f"    return {addition.expr}\n"
        )
    return text + "\n".join(blocks) + "\n"


def _write_text_transaction(writes: list[tuple[Path, str]]) -> None:
    """Publish library and structured caches together or restore every prior file."""
    prepared: list[tuple[Path, Path, Path | None]] = []
    published: list[tuple[Path, Path | None]] = []
    try:
        for target, text in writes:
            staged = _stage_bytes(target, text.encode("utf-8"))
            try:
                backup = _stage_bytes(target, target.read_bytes()) if target.exists() else None
            except BaseException:
                staged.unlink(missing_ok=True)
                raise
            prepared.append((target, staged, backup))
        for target, staged, backup in prepared:
            os.replace(staged, target)
            published.append((target, backup))
    except BaseException:
        for target, backup in reversed(published):
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(backup, target)
        raise
    finally:
        for _, staged, backup in prepared:
            staged.unlink(missing_ok=True)
            if backup is not None:
                backup.unlink(missing_ok=True)


def _stage_bytes(target: Path, content: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            os.chmod(temp_path, target.stat().st_mode)
        else:
            os.chmod(temp_path, 0o644)
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _signature(support: frozenset[int], domain: int) -> tuple[bool, ...]:
    return tuple(x in support for x in range(1, domain + 1))


def _smallest_unused_index(used: Container[int]) -> int:
    index = 0
    while index in used:
        index += 1
    return index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stats, changes, templates = structure_results(args.results_root, task=args.task, dry_run=args.dry_run)
    for domain in sorted(stats):
        summary = ", ".join(f"{key}={value}" for key, value in sorted(stats[domain].items()))
        print(f"d{domain}: {summary}")
    shown = changes.most_common(200)
    for (domain, old_type, new_type, old_size, new_size, text), count in shown:
        print(f"{count} d{domain} {old_type}->{new_type} {old_size}->{new_size} | {text}")
    if len(changes) > len(shown):
        print(f"... {len(changes) - len(shown)} more changed hypotheses ({sum(changes.values()) - sum(n for _, n in shown)} items)")
    # Work queue for the grammar: the most frequent phrasings it cannot read or reads as self-contradictory.
    queue = sorted(
        ((count, key, example) for key, (count, example) in templates.items() if key[1] in {"unparsed", "contradictory"}),
        key=lambda entry: (-entry[0], entry[1]),
    )
    for count, (domain, reason, template), example in queue[:40]:
        print(f"{count} d{domain} {reason} | {template}\n    e.g. {example}")
    if len(queue) > 40:
        print(f"... {len(queue) - 40} more unresolved templates ({sum(count for count, _, _ in queue[40:])} items)")


if __name__ == "__main__":
    main()
