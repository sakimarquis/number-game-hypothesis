"""Structure free-form generated hypotheses against executable libraries.

The structuring contract is finite-domain semantic matching: parse precise
natural-language hypotheses into support signatures on 1..domain, reuse any
existing library function with the same signature, and append a new predicate
only when no existing function matches.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tasks.hypotheses import is_prime as _is_prime


@dataclass(frozen=True)
class Term:
    support: frozenset[int]
    expr: str
    interval: bool = False


_TermBuilder = Callable[[re.Match[str], str, int], Term | None]
_TermRule = tuple[str, _TermBuilder]


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
    used_indices: set[int]
    additions: list[LibraryAddition]

    def index_for(self, predicate: ParsedPredicate) -> int:
        signature = _signature(predicate.support, self.domain)
        existing = self.signature_to_index.get(signature)
        if existing is not None:
            return existing
        index = _smallest_unused_index(self.used_indices)
        self.used_indices.add(index)
        self.signature_to_index[signature] = index
        self.additions.append(LibraryAddition(index=index, label=predicate.label, expr=predicate.expr))
        return index


_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD = "|".join(sorted([*_ONES, *_TENS, "hundred"], key=len, reverse=True))
_NUMBER_TOKEN_RE = re.compile(rf"(?<![\w.])-?\d+(?!\.\d)|\b(?:{_NUMBER_WORD})(?:[-\s](?:{_NUMBER_WORD}))?\b")
_ORDINAL_PRIME_RE = re.compile(
    r"(?:(?:the )?numbers? are )?(?:the )?\d+(?:st|nd|rd|th)"
    r"(?:\s*(?:,|and|or)\s*\d+(?:st|nd|rd|th))*\s+(?:primes|prime numbers?)"
)
_ORDINAL_INTEGER_CLAUSE = r"\b\d+(?:st|nd|rd|th)(?:\s*(?:,?\s+or)\s*\d+(?:st|nd|rd|th))*\s+positive integers?\b"
_VAGUE_MARKERS = (
    "description",
    "unknown property",
    "mysterious property",
    "unexplained property",
    "random",
    "biased towards",
    "biased",
    "cherry-pick",
    "cherry pick",
    "only provided",
    "provided only",
    "arbitrary",
    "perhaps",
    "wait",
    "re-evaluate",
    "reconsider",
    "too specific",
    "unlikely",
    "distractor",
    "exception",
    "non-exhaustive",
    "not listed",
    "not queried",
    "low confidence",
    "confidence is low",
    "similar to",
    "resemble",
    "near the",
    "close to the",
    "misleading",
    "salient",
    "outlier",
    "hardcoded exception",
    "specific exception",
    "data error",
    "mislabeled",
    "counterexample",
    "excluded",
    "hidden rule",
    "hidden complex",
    "some combination",
    "specific property",
    "obscure property",
    "or similar",
    "so maybe",
    "if the rule is",
    "user provided",
    "user is testing",
    "either about",
    "interval defined by specific endpoints",
    "english name",
    "letters in its english",
)


def structure_results(results_root: Path, *, task: str | None = None, dry_run: bool = False) -> dict[int, Counter]:
    raw_paths = sorted(results_root.rglob("hypothesis_generation.json"))
    if task is not None:
        raw_paths = [path for path in raw_paths if _path_has_part(path, task)]
    indexers: dict[int, DomainIndexer] = {}
    outputs: list[tuple[Path, dict]] = []
    stats: dict[int, Counter] = {}

    for raw_path in raw_paths:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        domain = int(payload["experiment_condition"]["domain"])
        if domain not in indexers:
            indexers[domain] = build_domain_indexer(domain)
        indexer = indexers[domain]
        counter = stats.setdefault(domain, Counter())
        structured = _structure_payload(payload, domain, indexer, counter)
        outputs.append((raw_path.with_name("hypothesis_generation_structured.json"), structured))

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
    return stats


def library_path(domain: int) -> Path:
    return Path(__file__).resolve().parent.parent / "library" / f"library_d{domain}.py"


def build_domain_indexer(domain: int) -> DomainIndexer:
    path = library_path(domain)
    module = load_library_module(path, f"_number_game_structuring_library_d{domain}")
    signature_to_index: dict[tuple[bool, ...], int] = {}
    used_indices = {int(index) for index in module.LIBRARY}
    for index, fn in module.LIBRARY.items():
        signature = tuple(bool(fn([x])) for x in range(1, domain + 1))
        signature_to_index.setdefault(signature, int(index))
    return DomainIndexer(
        domain=domain,
        library_path=path,
        signature_to_index=signature_to_index,
        used_indices=used_indices,
        additions=[],
    )


def parse_hypothesis_text(text: object, domain: int) -> ParsedPredicate | None:
    parsed = _parse_hypothesis_text_cached(str(text), int(domain))
    # A known empty branch can participate in OR; only the complete hypothesis
    # must have executable support to enter the library.
    return parsed if parsed is not None and parsed.support else None


@lru_cache(maxsize=None)
def _parse_hypothesis_text_cached(text: str, domain: int) -> ParsedPredicate | None:
    raw = str(text).strip()
    normalized = _normalize_struct_text(raw)
    if not normalized or normalized in {"other", "unknown", "n/a"}:
        return None
    if _self_refuting(normalized) or any(marker in normalized for marker in _VAGUE_MARKERS):
        return None

    equivalence = re.search(r"\(\s*(?:(?:i\.e\.|that is|equivalently)[,\s]+|(?=interval\b))([^()]+)\)", normalized)
    if equivalence:
        main_text = normalized[:equivalence.start()] + normalized[equivalence.end():]
        main = _parse_hypothesis_text_cached(main_text, domain)
        explanation = _parse_hypothesis_text_cached(equivalence.group(1), domain)
        if main is None or explanation is None or main.support != explanation.support:
            return None
        return ParsedPredicate(main.kind, _library_label(raw), main.support, main.expr)

    for candidate in _candidate_texts(normalized, domain):
        parsed = _parse_hypothesis_candidate(raw, candidate, domain)
        if parsed is not None:
            return parsed
    return None


def _parse_hypothesis_candidate(raw: str, normalized: str, domain: int) -> ParsedPredicate | None:
    # Candidates arrive pre-normalized from _candidate_texts after the
    # empty/self-refuting/vague guards.
    output_condition = re.fullmatch(
        r"(.+?)\s*,?\s+and (?:the program )?(?:only )?(?:outputs?|says?|returns?) ['\"]?yes['\"]? "
        r"(?:only )?(?:if|for) (.+?)(?:, otherwise ['\"]?no['\"]?)?",
        normalized,
    )
    # OR needs the whole left side grouped before applying the output condition;
    # decimal divisors are handled by an exact bounded-form parser below.
    if output_condition and (re.search(r"\bor\b", normalized) or re.search(r"\d+\.\d+", normalized)):
        before = _parse_hypothesis_text_cached(output_condition.group(1), domain)
        constraint = re.sub(r"^it\b", "the number", output_condition.group(2))
        constraint = re.sub(r"^those\b", "numbers", constraint)
        after = _parse_hypothesis_text_cached(constraint, domain)
        if before is not None and after is not None:
            terms = [Term(p.support, p.expr, p.kind == "interval") for p in (before, after)]
            kind = "interval" if any(p.kind == "interval" and len(p.support) == 1 for p in (before, after)) else None
            return _predicate(raw, terms, domain, forced_kind=kind)
        return None

    shifted_prime = re.fullmatch(
        rf"(?:(?:the )?number is |numbers? (?:equal to|of the form) )?(?:a )?prime(?: number)?\s+(plus|minus|\+|-)\s+({_NUMBER_TOKEN_RE.pattern})",
        normalized,
    )
    if shifted_prime:
        offset = _parse_number(shifted_prime.group(2))
        if offset is not None:
            if shifted_prime.group(1) in {"minus", "-"}:
                offset = -offset
            return _predicate(raw, [_term(f"_is_prime(x - ({offset}))", domain)], domain, forced_kind="rule")

    coefficient_prime = re.fullmatch(
        rf"(?:numbers? (?:of the form )?)?({_NUMBER_TOKEN_RE.pattern})\s*\*?\s*n,? where n is (?:a )?prime(?: number)?",
        normalized,
    )
    if coefficient_prime:
        coefficient = _parse_number(coefficient_prime.group(1))
        if coefficient is not None and coefficient > 0:
            return _predicate(raw, [_term(f"x % {coefficient} == 0 and _is_prime(x // {coefficient})", domain)], domain, forced_kind="rule")
        return None
    if re.search(r"\bdistinct prime factors?\b", normalized) and "multiplicity" in normalized:
        return None

    # A stated total is a constraint on the complete set, not a number to add.
    total = re.fullmatch(rf"(.+),\s*totall?ing\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:numbers|integers)", normalized)
    if total:
        count = _parse_number(total.group(2))
        sequence = _parse_first_n_pattern(total.group(1), domain, total=count)
        if sequence is not None:
            return _predicate(raw, [sequence], domain, forced_kind="rule")
        if re.search(r"\bfirst\b", total.group(1)):
            return None
        base = _parse_hypothesis_text_cached(total.group(1), domain)
        if base is not None and count == len(base.support):
            return ParsedPredicate(base.kind, _library_label(raw), base.support, base.expr)
        return None

    exclusion = re.fullmatch(rf"(.+?)\s+(?:exclusive of|excluding)\s+(?P<nums>{_number_list_pattern()})", normalized)
    if exclusion:
        base = _parse_hypothesis_text_cached(exclusion.group(1), domain)
        if base is None:
            return None
        excluded = _term(f"x not in {_set_literal(_extract_numbers(exclusion.group('nums')))}", domain)
        return _predicate(raw, [Term(base.support, base.expr, base.kind == "interval"), excluded], domain)

    multiple_union = _parse_multiple_or_singleton_union(normalized, domain)
    if multiple_union is not None:
        return _predicate(raw, [multiple_union], domain, forced_kind="rule")

    simple_exclusion = _parse_simple_exclusion(raw, normalized, domain)
    if simple_exclusion is not None:
        return simple_exclusion
    if _has_exclusion_marker(normalized):
        return None

    finite = _parse_explicit_set(normalized, domain)
    if finite is not None:
        return _predicate(raw, [finite], domain, forced_kind="rule")

    added_numbers = re.fullmatch(rf"(.+?)\s+plus (?:the )?(?:specific )?numbers?\s+(?P<nums>{_number_list_pattern()})", normalized)
    if added_numbers:
        base = _parse_hypothesis_text_cached(added_numbers.group(1), domain)
        if base is None:
            return None
        extra = _term(f"x in {_set_literal(_extract_numbers(added_numbers.group('nums')))}", domain)
        return _predicate(raw, [_combine_terms([Term(base.support, base.expr), extra], "or")], domain, forced_kind="rule")

    only = _parse_only_singleton(normalized, domain)
    if only is not None:
        return _predicate(raw, [only], domain, forced_kind="interval")
    # Remaining procedural output conditions and numerical commentary cannot
    # become extra conjuncts inside just one side of an OR.
    if re.search(r"\bor\b", normalized) and re.search(r"\band (?:the program )?(?:only )?(?:outputs?|says?|returns?)\s+['\"]?yes\b", normalized):
        return None
    if re.search(r"\bor\b", normalized) and re.search(r"[.,]\s+\d+\s+is\s+(?:not\s+)?(?:a\s+)?(?:divisible|multiple)\b", normalized):
        return None

    integer_subset = re.fullmatch(
        rf"(?:the )?(?:numbers? are )?(?:the )?first\s+(?P<count>{_NUMBER_TOKEN_RE.pattern})\s+"
        rf"(?:positive )?integers?(?:\s*\(1 to (?P=count)\))?\s+"
        rf"(?:that are|that satisfy the condition)\s+(?P<condition>.+)", normalized,
    )
    if integer_subset:
        count = _parse_number(integer_subset.group("count"))
        condition = _parse_hypothesis_text_cached(integer_subset.group("condition"), domain)
        if count is not None and condition is not None:
            return _predicate(raw, [Term(condition.support, condition.expr), _term(f"1 <= x <= {count}", domain, interval=True)], domain)
        return None

    first_n = _parse_first_n_pattern(normalized, domain)
    if first_n is not None:
        return _predicate(raw, [first_n], domain, forced_kind="rule")
    if re.search(rf"\bfirst\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:(?:perfect\s+)?(?:squares?|cubes?)|primes?|multiples?|even|odd)\b", normalized):
        return None

    if "prime" in normalized and re.search(r"\b\d+(?:st|nd|rd|th)\b", normalized):
        if not _ORDINAL_PRIME_RE.fullmatch(normalized) and not re.search(_ORDINAL_INTEGER_CLAUSE, normalized):
            return None

    bounded_form = _parse_bounded_form(normalized, domain)
    if bounded_form is not None:
        return _predicate(raw, [bounded_form], domain, forced_kind="rule")
    # Bounds on a root or on one digit position are not bounds on x; abstain
    # when no exact parser owns them instead of dropping or misapplying them.
    if re.search(
        r"\b(?:squares?|cubes?) of (?:the )?(?:positive )?(?:integers?|numbers?)\s+"
        r"(?:from|between|under|over|below|above|less|greater|more|up to|at most|at least|\d)", normalized,
    ):
        return None
    if re.search(r"\b(?:square|cube) roots?\b", normalized):
        integer_root = re.fullmatch(
            r"(?:numbers? )?(?:with (?:an? )?(?:(?:odd|even|positive) )?(?:integer|whole number) (?:square|cube) roots?"
            r"|whose (?:square|cube) roots? (?:is|are) (?:an? )?(?:(?:odd|even|positive) )?(?:integers?|whole numbers?))",
            normalized,
        )
        if not integer_root:
            return None
    if re.search(
        rf"\b(?:(?:first|last|leading|tens|units?|ones|hundreds) digit\s+(?:is\s+)?|digits\s+(?:are\s+)?)"
        rf"(?:(?:from\s+)?(?:{_NUMBER_TOKEN_RE.pattern})\s*(?:to|through|-)|between\s+(?:{_NUMBER_TOKEN_RE.pattern})\s+and)\s*(?:{_NUMBER_TOKEN_RE.pattern})", normalized,
    ):
        return None
    # Only digit and divisor counts have implemented parity terms. Prime-factor
    # counts are ambiguous between distinct factors and multiplicity.
    for count in re.finditer(r"\b(?:even|odd) number of (?!the form\b)(?:(?:positive|total) )?(\w+)", normalized):
        if count.group(1) not in {"divisors", "factors", "digits"} or "binary" in normalized:
            return None
        if count.group(1) == "digits" and re.search(r"\b(?:base|octal|hexadecimal|hex)\b", normalized):
            return None
        if count.group(1) in {"divisors", "factors"} and "multiplicity" in normalized:
            return None

    quotient = re.fullmatch(
        rf"numbers n where n is a multiple of (?P<base>{_NUMBER_TOKEN_RE.pattern}) "
        rf"and n/(?P=base) is (?:a )?(?:perfect )?square or (?:a )?(?:perfect )?cube", normalized,
    )
    if quotient:
        base = _parse_number(quotient.group("base"))
        if base and base > 0:
            term = _term(f"x % {base} == 0 and (_is_square(x // {base}) or _is_cube(x // {base}))", domain)
            return _predicate(raw, [term], domain, forced_kind="rule")
    if re.search(r"\b[nx]\s*/\s*\d+", normalized):
        return None

    stripped = _strip_boilerplate(normalized)
    parts = _split_or(stripped)
    if parts is None:
        # Explicit grouping fixes conjunction scope without splitting operand
        # lists such as "divisible by 3 and 7".
        grouped = _strip_outer_groups(stripped)
        for match in _top_level_matches(grouped, r"\band\b") or ():
            sides = [grouped[:match.start()].strip(), grouped[match.end():].strip()]
            if not any(_strip_outer_groups(side) != side for side in sides):
                continue
            predicates = [_parse_hypothesis_text_cached(side, domain) for side in sides]
            if all(predicates):
                terms = [Term(p.support, p.expr) for p in predicates if p is not None]
                return _predicate(raw, terms, domain, forced_kind="rule")
            return None
        return None
    if len(parts) > 1:
        predicates = [_parse_hypothesis_text_cached(part, domain) for part in parts]
        if all(predicates):
            terms = [Term(p.support, p.expr, interval=p.kind == "interval") for p in predicates if p is not None]
            return _predicate(raw, [_combine_terms(terms, "or")], domain, forced_kind="rule")
        return None

    if re.search(rf"\bprime(?: number)?\s+(?:plus|minus|\+|-)\s+(?:{_NUMBER_TOKEN_RE.pattern})|(?:{_NUMBER_TOKEN_RE.pattern})\s*\*?\s*n,? where n is (?:a )?prime\b", normalized):
        return None
    terms = _parse_conjunctive_terms(stripped, domain)
    if not terms and stripped != normalized:
        terms = _parse_conjunctive_terms(normalized, domain)
    or_union = _parse_modulo_interval_or_union(stripped, domain)
    if or_union is not None:
        return _predicate(raw, [or_union], domain, forced_kind="rule")
    if not terms:
        return None
    if all(term.interval for term in terms):
        # An interval-only fallback must describe x itself, not merely mention
        # a domain alongside an unrecognized property.
        remainder = re.sub(_DIGIT_COUNT_CLAUSE, " ", stripped)
        remainder = re.sub(r"\b\d+(?:st|nd|rd|th)\b|\b\d0s\b", " ", remainder)
        remainder = _NUMBER_TOKEN_RE.sub(" ", remainder)
        allowed = {
            "a", "an", "the", "number", "numbers", "integer", "integers", "positive", "all", "any",
            "is", "are", "must", "be", "that", "which", "those", "these", "in", "within", "from", "to", "between",
            "and", "or", "at", "least", "most", "no", "less", "fewer", "more", "greater", "than",
            "equal", "equals", "above", "below", "under", "over", "up", "through", "strictly", "inclusive", "closed", "interval", "range",
            "exactly", "only", "first", "input", "always", "x", "n",
        }
        if any(word not in allowed for word in re.findall(r"[a-z]+", remainder)):
            return None
    return _predicate(raw, terms, domain)


def _candidate_texts(text: str, domain: int) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = _strip_boilerplate(_normalize_struct_text(candidate))
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    quoted_rule = re.search(
        r"(?:rule is\b|rule:|checks? if\b|checks? whether\b)\s*(['\"])(?P<body>.+?)\1",
        text,
        flags=re.IGNORECASE,
    )
    if quoted_rule:
        tail = _strip_boilerplate(text[quoted_rule.end():]).strip(" .;:,?!")
        if not tail:
            add(quoted_rule.group("body"))
            return candidates

        comments = re.fullmatch(r"(?:\s*\([^()]*\)\s*[.,;:]?)+", tail)
        if comments:
            parentheticals = re.findall(r"\(([^()]*)\)", tail)

            def is_example_comment(comment: str) -> bool:
                comment = comment.strip()
                if re.match(r"(?:e\.g\.|for example\b|examples?\s*:|union of\b)", comment, re.IGNORECASE):
                    return True
                if re.fullmatch(r"\d+(?:\s*[,;]\s*\d+)*", comment):
                    return True
                return bool(re.search(r"(?:^|[,;]\s*)\d+\s*(?::|=|\b(?:is|are|means?|gives?|has)\b|\+)", comment))

            if parentheticals and all(is_example_comment(comment) for comment in parentheticals):
                add(quoted_rule.group("body"))
                return candidates
            body = _parse_hypothesis_text_cached(quoted_rule.group("body"), domain)
            if body is not None and body.kind == "interval" and all(
                comment.strip() in {"inclusive", "all integers in that range"} for comment in parentheticals
            ) and not re.search(r"\b(?:strictly|exclusive|excluding|greater than|less than)\b", quoted_rule.group("body")):
                add(quoted_rule.group("body"))
                return candidates
            explanations = [_parse_hypothesis_text_cached(comment, domain) for comment in parentheticals]
            if body and explanations and all(p and p.support == body.support for p in explanations):
                add(quoted_rule.group("body"))
                return candidates
        included_examples = re.fullmatch(rf"\([\d\s,.;]+\),?\s+which includes\s+(?P<nums>{_number_list_pattern()})", tail)
        if included_examples:
            body = _parse_hypothesis_text_cached(quoted_rule.group("body"), domain)
            if body is not None and set(_extract_numbers(included_examples.group("nums"))) <= body.support:
                add(quoted_rule.group("body"))
            return candidates
        constraint = _strip_outer_groups(tail)
        if re.match(r"(?:and|or)\b", constraint):
            add(f"({quoted_rule.group('body')}) {constraint}")
        return candidates
    # Numeric illustrations are redundant; parenthesized constraints are not.
    text = re.sub(
        r"\s*,?\s+and (?:the program )?(?:outputs?|says?|returns?) ['\"]?yes['\"]? "
        r"(?:if it is|if true|if it (?:is|matches) either|for (?:these|those|(?:any|all) of them))"
        r"(?:, otherwise ['\"]?no['\"]?)?$", "", text,
    )
    no_parenthetical = re.sub(
        r"\s*\((?:(?:e\.g\.|i\.e\.|for example)\s*)?[\d\s,.;]+\)", "", text,
    )
    if re.search(rf"\bfirst\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:prime|multiple|perfect|square|cube|even|odd)", text):
        # Sequence enumerations assert cardinality and endpoints; their owner
        # verifies them before clipping to the finite domain.
        no_parenthetical = text
    if re.search(rf"(?:greater|less|more|fewer) than\s+(?:{_NUMBER_TOKEN_RE.pattern})", no_parenthetical):
        no_parenthetical = re.sub(r"\s*\(not including the endpoint\)", "", no_parenthetical)
    for fragment in (no_parenthetical, text):
        for separator in (" because ", " since ", " explanation:", " examples:"):
            if separator in fragment:
                add(fragment.split(separator, 1)[0])
        add(fragment)
    return candidates


def _structure_payload(payload: dict, domain: int, indexer: DomainIndexer, stats: Counter) -> dict:
    structured = {key: value for key, value in payload.items() if key != "prefix_cache"}
    prefix_cache = {}
    for prefix, items in payload["prefix_cache"].items():
        structured_items = []
        for item in items:
            stats["items"] += 1
            predicate = parse_hypothesis_text(item.get("hypothesis", ""), domain)
            next_item = dict(item)
            if predicate is None:
                next_item["type"] = "other"
                next_item.pop("index", None)
                stats["other"] += 1
            else:
                next_item["type"] = predicate.kind
                next_item["index"] = indexer.index_for(predicate)
                stats[predicate.kind] += 1
            structured_items.append(next_item)
        prefix_cache[prefix] = structured_items
    structured["prefix_cache"] = prefix_cache
    return structured


def _parse_conjunctive_terms(text: str, domain: int) -> list[Term]:
    terms: list[Term] = []

    terms.extend(_parse_digit_sum_terms(text, domain))
    if not terms and re.search(rf"{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX}", text):
        return []
    text = _strip_digit_sum_clauses(text)
    terms.extend(_parse_multiple_terms(text, domain))
    terms.extend(_parse_terms(text, domain, _simple_term_rules()))
    for parser in (
        _parse_digit_count_terms,
        _parse_interval_terms,
        _parse_modulo_terms,
        _parse_form_terms,
        _parse_shifted_power_terms,
        _parse_sum_representation_terms,
        _parse_consecutive_sum_terms,
    ):
        terms.extend(parser(text, domain))

    number_text = _strip_non_subject_property_clauses(text)
    if re.search(r"\bnot\s+(?:a\s+)?prime\b|\bnot prime\b|\bnon-prime\b", number_text):
        terms.append(_term("not _is_prime(x)", domain))
    elif re.search(r"\bprime(?:s| numbers?| number)?\b", number_text):
        terms.append(_term("_is_prime(x)", domain))

    if re.search(r"\bcomposite(?:s| numbers?| number)?\b", number_text):
        terms.append(_term("x > 1 and not _is_prime(x)", domain))
    if re.search(r"\bnot\s+(?:an?\s+)?odd(?: numbers?| number)?\b", number_text):
        terms.append(_term("x % 2 == 0", domain))
    elif re.search(r"\bnot\s+(?:an?\s+)?even(?: numbers?| number)?\b", number_text):
        terms.append(_term("x % 2 == 1", domain))
    elif re.search(r"\beven(?: numbers?| number)?\b", number_text):
        terms.append(_term("x % 2 == 0", domain))
    elif re.search(r"\bodd(?: numbers?| number)?\b", number_text):
        terms.append(_term("x % 2 == 1", domain))

    shifted_square_context = re.search(r"\b(?:perfect\s+)?squares?\s+(?:plus|\+|multiplied|times|ends?)\b|\bsum of two squares\b", text)
    if re.search(r"\bnot\s+(?:a\s+)?(?:perfect\s+)?squares?\b", text):
        terms.append(_term("not _is_square(x)", domain))
    elif not shifted_square_context and re.search(r"\b(?:perfect\s+)?squares?\b|\bsquare numbers?\b", text):
        terms.append(_term("_is_square(x)", domain))

    digit_cube_context = re.search(r"\bsum of (?:the |its )?cubes of (?:the |its )?digits\b", text)
    if re.search(r"\bnot\s+(?:a\s+)?(?:perfect\s+)?cubes?\b", text):
        terms.append(_term("not _is_cube(x)", domain))
    elif not digit_cube_context and re.search(r"\b(?:perfect\s+)?cubes?\b|\bcube numbers?\b", text):
        terms.append(_term("_is_cube(x)", domain))

    # "perfect prime power" / "prime power" form an elif group; "perfect power" is independent.
    if "perfect prime power" in text:
        terms.append(_term("_is_prime_power(x) and not _is_prime(x)", domain))
    elif "prime power" in text:
        terms.append(_term("_is_prime_power(x)", domain))

    keyword_text = re.sub(_FIBONACCI_DIGIT_CLAUSE, " ", text)
    for pattern, expr in _KEYWORD_TERMS:
        if re.search(pattern, keyword_text):
            terms.append(_term(expr, domain))
    if "distinct digits" in text and not re.search(rf"exactly\s+({_NUMBER_TOKEN_RE.pattern})\s+distinct digits", text):
        terms.append(_term("len(set(str(abs(x)))) == len(str(abs(x)))", domain))

    if re.fullmatch(r"(?:(?:all|any|the|numbers that are)\s+)?positive integers?", text):
        terms.append(_term(f"1 <= x <= {domain}", domain, interval=True))
    if re.fullmatch(r"(?:all|any)\s+numbers?|\bintegers?\b|\bany integer\b", text):
        terms.append(_term(f"1 <= x <= {domain}", domain, interval=True))

    for match in re.finditer(rf"perfect\s+({_NUMBER_TOKEN_RE.pattern})(?:st|nd|rd|th)?\s+powers?", text):
        exponent = _parse_number(match.group(1))
        if exponent is not None and exponent > 1:
            expr = "x == 1" if exponent > math.log2(domain) else f"x > 0 and any(n ** {exponent} == x for n in range(1, {domain + 2}))"
            terms.append(_term(expr, domain))
    for ordinal, exponent in {"fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}.items():
        if re.search(rf"\bperfect\s+{ordinal}\s+powers?\b", text):
            terms.append(_term(f"x > 0 and any(n ** {exponent} == x for n in range(1, {domain + 2}))", domain))
    # Must run after the ordinal loop: "perfect fourth powers" triggers both, and original ordering
    # places this expr after the ordinal expr in the conjunction.
    if re.search(r"\bfourth powers?\b", text):
        terms.append(_term("any(n ** 4 == x for n in range(1, math.isqrt(x) + 2))", domain))
    for match in re.finditer(rf"(exactly|at least|at most)\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:distinct\s+)?prime factors?(?:,?\s*\(?\s*(?:counted with|counting|with) multiplicity\)?)?", text):
        count = _parse_number(match.group(2))
        if count is not None:
            factor_count = (
                "sum(1 for p in _distinct_prime_factors(x) for k in range(1, x.bit_length()) if x % (p ** k) == 0)"
                if "multiplicity" in match.group(0) else "len(_distinct_prime_factors(x))"
            )
            op = {"exactly": "==", "at least": ">=", "at most": "<="}[match.group(1)]
            terms.append(_term(f"{factor_count} {op} {count}", domain))
    for match in re.finditer(_DIVISOR_COUNT_CLAUSE, text):
        count_expr = "sum(1 for d in range(1, x + 1) if x % d == 0)"
        if match.group("test") in {"even", "odd"}:
            expr = f"{count_expr} % 2 == {int(match.group('test') == 'odd')}"
        else:
            value = _parse_number(match.group("value"))
            if value is None:
                continue
            op = {"greater than": ">", "less than": "<", "at least": ">=", "at most": "<=", "greater than or equal to": ">=", "less than or equal to": "<=", None: "==", "equal to": "=="}[match.group("op")]
            expr = f"{count_expr} {op} {value}"
        terms.append(_term(expr, domain))
    if re.search(_FACTOR_EXPONENT_CLAUSE, text):
        terms.append(_term("_is_square(x)", domain))
    if re.search(r"\bno odd prime factors\b", text):
        terms.append(_term("all(p == 2 for p in _distinct_prime_factors(x))", domain))
    for match in re.finditer(rf"exactly\s+({_NUMBER_TOKEN_RE.pattern})\s+distinct digits?", text):
        count = _parse_number(match.group(1))
        if count is not None:
            terms.append(_term(f"len(set(str(abs(x)))) == {count}", domain))
    for match in re.finditer(rf"binary representation (?:contains?|has) exactly\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:1s|ones|set bits)", text):
        count = _parse_number(match.group(1))
        if count is not None:
            terms.append(_term(f"x.bit_count() == {count}", domain))

    if re.search(r"\bfactorials?\b", text):
        support = {math.factorial(n) for n in range(1, 8)}
        support = {x for x in support if 1 <= x <= domain}
        if support:
            terms.append(Term(frozenset(support), f"x in {_set_literal(support)}"))

    for match in re.finditer(rf"(?P<proper>proper )?divisors? of\s+(?P<nums>{_number_list_pattern()})", text):
        nums = _extract_numbers(match.group("nums"))
        joiner = f" {_connector(match.group('nums'))} "
        exprs = [f"({value} % x == 0" + (f" and x < {value})" if match.group("proper") else ")") for value in nums]
        if exprs:
            terms.append(_term(joiner.join(exprs), domain))
    for match in re.finditer(rf"exactly\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:positive\s+)?divisors", text):
        count = _parse_number(match.group(1))
        if count is not None:
            terms.append(_term(f"sum(1 for d in range(1, x + 1) if x % d == 0) == {count}", domain))

    for match in re.finditer(rf"(?P<neg>not\s+(?:a\s+)?)?powers? of\s+(?P<nums>{_number_list_pattern()})", text):
        power_bases = [base for base in _extract_numbers(match.group("nums")) if base > 0]
        if power_bases:
            expr = " or ".join(f"(x > 0 and any({base} ** n == x for n in range(0, 16)))" for base in power_bases)
            terms.append(_term(f"not ({expr})" if match.group("neg") else expr, domain))

    exact = _parse_exact_singleton(text, domain)
    if exact is not None:
        terms.append(exact)
    for match in re.finditer(_ORDINAL_INTEGER_CLAUSE, text):
        values = [int(n) for n in re.findall(r"\d+", match.group(0))]
        terms.append(_term(f"x in {_set_literal(values)}", domain, interval=len(values) == 1))

    return _dedupe_terms(terms)


# Constant-expression keyword triggers: each (pattern, expr) emits _term(expr, domain) when pattern
# matches text. Entries are independent (no elif coupling); precedence-coupled keywords (parity,
# square/cube, prime, prime-power) live as explicit elif blocks above.
_FIBONACCI_EXPR = "(math.isqrt(5 * x * x + 4) ** 2 == 5 * x * x + 4) or (math.isqrt(5 * x * x - 4) ** 2 == 5 * x * x - 4)"
_KEYWORD_TERMS: tuple[tuple[str, str], ...] = (
    ("fibonacci", _FIBONACCI_EXPR),
    ("lucas", "_is_lucas(x)"),
    ("mersenne", "x > 0 and ((x + 1) & x) == 0"),
    ("semiprime", "_is_semiprime(x)"),
    ("harshad|niven", "x % _digit_sum(x) == 0"),
    ("triangular", "_is_triangular(x)"),
    ("pentagonal", "any(n * (3 * n - 1) // 2 == x for n in range(1, x + 1))"),
    ("hexagonal", "any(n * (2 * n - 1) == x for n in range(1, x + 1))"),
    ("palindrome|palindromic", "str(abs(x)) == str(abs(x))[::-1]"),
    ("perfect power", "_is_perfect_power(x)"),
    (r"\brepunit\b", "set(str(abs(x))) == {'1'}"),
    (r"\bsquare of a prime\b|\bprime square\b", "_is_square(x) and _is_prime(math.isqrt(x))"),
    (r"\bn\^2\b(?!\s*=)|\bn squared\b", "_is_square(x)"),
    (r"\bn\^3\b(?!\s*=)|\bn cubed\b", "_is_cube(x)"),
    (r"\bpowers of integers\b|\bpowers of whole numbers\b", "_is_perfect_power(x)"),
    (r"\bexactly two divisors\b", "_is_prime(x)"),
    (r"(?<!no )\brepeated digits\b", "len(set(str(abs(x)))) < len(str(abs(x)))"),
    (r"\brepdigit\b|\ball digits are (?:the )?same\b", "len(set(str(abs(x)))) == 1"),
    (r"\b(?:all (?:the )?digits are distinct|no repeated digits)\b", "len(set(str(abs(x)))) == len(str(abs(x)))"),
    (r"\bround numbers?\b", "x % 10 == 0"),
    (r"\bone[-\s]?hot\b|\bone set bit\b|exactly one bit set\b", "x > 0 and (x & (x - 1)) == 0"),
    (r"\babundant numbers?\b", "sum(d for d in range(1, x) if x % d == 0) > x"),
    (r"\bperfect numbers?\b", "x > 1 and sum(d for d in range(1, x) if x % d == 0) == x"),
    (r"\beven number of (?:(?:positive|total) )?(?:divisors|factors)\b", "sum(1 for d in range(1, x + 1) if x % d == 0) % 2 == 0"),
    (r"\bodd number of (?:(?:positive|total) )?(?:divisors|factors)\b", "sum(1 for d in range(1, x + 1) if x % d == 0) % 2 == 1"),
    (r"\beven number of digits\b", "len(str(abs(x))) % 2 == 0"),
    (r"\bodd number of digits\b", "len(str(abs(x))) % 2 == 1"),
    (r"\bat least one even digit\b|\bwith an even digit\b", "any(int(ch) % 2 == 0 for ch in str(abs(x)))"),
    (r"\bat least one odd digit\b|\bwith an odd digit\b", "any(int(ch) % 2 == 1 for ch in str(abs(x)))"),
    (r"\bone even digit and one odd digit\b", "10 <= x <= 99 and {int(str(x)[0]) % 2, int(str(x)[1]) % 2} == {0, 1}"),
    (r"\btens digit is greater than (?:the )?(?:units|ones) digit\b", "10 <= x <= 99 and x // 10 > x % 10"),
    (r"\btens digit is less than (?:the )?(?:units|ones) digit\b", "10 <= x <= 99 and x // 10 < x % 10"),
    (r"\bdigits are in (?:ascending|increasing) order\b|\b(?:ascending|increasing) digits\b", "all(a < b for a, b in zip(str(x), str(x)[1:]))"),
    (r"\bdigits are in (?:descending|decreasing) order\b|\b(?:descending|decreasing) digits\b", "all(a > b for a, b in zip(str(x), str(x)[1:]))"),
)


def _parse_terms(text: str, domain: int, rules: Iterable[_TermRule]) -> list[Term]:
    terms = []
    for pattern, build in rules:
        for match in re.finditer(pattern, text):
            term = build(match, text, domain)
            if term is not None:
                terms.append(term)
    return terms


def _parse_multiple_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(rf"prime factor of\s+(?P<nums>{_number_list_pattern()})", text):
        values = [value for value in _extract_numbers(match.group("nums")) if value > 1]
        if values:
            joiner = f" {_connector(match.group('nums'))} "
            terms.append(_term(joiner.join(f"x % {value} == 0" for value in values), domain))

    # Generic divisibility claims take priority before narrower xor/exclusion forms.
    terms.extend(_parse_terms(text, domain, (
        (
            rf"(?P<neg>not\s+(?:an?\s+)?)?(?:(?:multiples?|multiple)\s+of|divisible\s+by)\s+(?:(?P<quantifier>both|either)\s+)?(?P<nums>{_number_list_pattern()})",
            _build_multiple_divisibility_term,
        ),
        (
            rf"(?:multiples?|multiple)\s+of\s+({_NUMBER_TOKEN_RE.pattern})\s+or\s+({_NUMBER_TOKEN_RE.pattern}).*?\bbut\s+not\s+both\b",
            _build_multiple_xor_term,
        ),
        (
            rf"(?:divisible\s+by|multiple\s+of)\s+(?P<yes>{_number_list_pattern()})\s*,?\s+but\s+not\s+(?:by\s+)?(?:a\s+)?(?:multiple\s+of\s+)?(?P<no>{_number_list_pattern()})",
            _build_multiple_but_not_term,
        ),
    )))

    if re.search(r"\b(?:multiple of|divisible by) 2 where the quotient is odd\b", text):
        terms.append(_term("x % 4 == 2", domain))
    return terms


def _build_multiple_divisibility_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    # Keep offset, comparison, and digit-sum clauses owned by their specific parsers.
    after = text[match.end():match.end() + 24]
    if re.match(r"\s+(?:plus|with (?:a )?remainder)\s+", after):
        return None
    if not match.group("neg") and re.match(rf"\s*,?\s+but\s+not\s+(?:by\s+)?(?:a\s+)?(?:multiple\s+of\s+)?(?:{_NUMBER_TOKEN_RE.pattern})", text[match.end():]):
        return None
    before = text[max(0, match.start() - 24):match.start()]
    if re.search(r"(?:more|less) than (?:a )?$", before):
        return None
    if "digit sum" in before or "digits sum" in before or "sum of" in before:
        return None

    operands = _extract_numbers(match.group("nums"))
    nums = [n for n in operands if n != 0]
    if not nums:
        return None
    connector = _connector(match.group("nums"))
    if match.group("neg"):
        joiner = " or " if match.group("quantifier") == "both" else " and "
        expr = joiner.join(f"x % {n} != 0" for n in nums)
    elif connector == "and":
        expr = " and ".join(f"x % {n} == 0" for n in nums)
    elif re.search(r"\band\b", match.group("nums")):
        # Mixed explicit operators retain AND-before-OR precedence, so a zero
        # operand cannot be dropped without changing the expression.
        if len(nums) != len(operands):
            return None
        expr = _NUMBER_TOKEN_RE.sub(lambda m: f"(x % {_parse_number(m.group(0))} == 0)", match.group("nums"))
        expr = re.sub(r",\s*(?=and\b|or\b)", " ", expr)
        expr = re.sub(r"[,/]", " or ", expr)
    else:
        expr = " or ".join(f"x % {n} == 0" for n in nums)
    return _term(expr, domain)


def _build_multiple_xor_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    left = _parse_number(match.group(1))
    right = _parse_number(match.group(2))
    if left not in {None, 0} and right not in {None, 0}:
        return _term(f"(x % {left} == 0) != (x % {right} == 0)", domain)
    return None


def _build_multiple_but_not_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    yes = [n for n in _extract_numbers(match.group("yes")) if n]
    no = [n for n in _extract_numbers(match.group("no")) if n]
    if yes and no:
        joiner = f" {_connector(match.group('yes'))} "
        included = joiner.join(f"x % {n} == 0" for n in yes)
        excluded = " and ".join(f"x % {n} != 0" for n in no)
        return _term(f"({included}) and ({excluded})", domain)
    return None


def _build_ending_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    # A root or squared-value suffix condition does not constrain x's last digit.
    if re.search(r"\bsquares? of (?:an? )?(?:numbers?|integers?)\s*$|\b(?:whose|its|their) square\s*$", text[:match.start()]):
        return None
    nums = _extract_numbers(match.group("nums"))
    if not nums:
        return None
    if all(0 <= n <= 9 for n in nums):
        op = "not in" if match.group("neg") else "in"
        expr = f"x % 10 {op} {_set_literal(nums)}"
    elif match.group("neg"):
        expr = " and ".join(f"not str(abs(x)).endswith('{n}')" for n in nums)
    else:
        expr = " or ".join(f"str(abs(x)).endswith('{n}')" for n in nums)
    return _term(expr, domain)


def _build_contains_digit_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    if re.search(r"\b(?:factorization|binary)\b", text[:match.start()]):
        return None
    nums = [n for n in _extract_numbers(match.group("nums")) if 0 <= n <= 9]
    if not nums:
        return None
    connector = _connector(match.group("nums"))
    if match.groupdict().get("neg"):
        expr = " and ".join(f"'{n}' not in str(abs(x))" for n in nums)
    else:
        joiner = " and " if connector == "and" else " or "
        expr = joiner.join(f"'{n}' in str(abs(x))" for n in nums)
    return _term(expr, domain)


def _build_tens_digit_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    nums = [n for n in _extract_numbers(match.group("nums")) if 0 <= n <= 9]
    if nums:
        return _term(f"x >= 10 and (x // 10) % 10 in {_set_literal(nums)}", domain)
    return None


def _build_starting_digit_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    if re.match(r"\s+(?:through|to)\s+\d", text[match.end():]):
        return None
    nums = _extract_numbers(match.group("nums"))
    if nums and all(1 <= n <= 9 for n in nums):
        return _term(" or ".join(f"str(x).startswith('{n}')" for n in nums), domain)
    return None


# Built lazily because the patterns interpolate _number_list_pattern() which is defined below.
@lru_cache(maxsize=1)
def _simple_term_rules() -> tuple[_TermRule, ...]:
    return (
        (
            rf"(?P<neg>not\s+)?(?:ends?\s+(?:in|with)|ending\s+(?:in|with)|last digit(?:\s+is|\s+equals?)?|(?:units|ones) digit(?:\s+is|\s+equals?)?)\s+(?:the\s+)?(?:digits?\s+)?(?P<nums>{_number_list_pattern()})",
            _build_ending_term,
        ),
        (
            rf"(?P<neg>not\s+)?(?:contains?|containing|decimal representation contains|digits? include|with (?:a\s+)?digits?|has (?:a\s+)?digits?|having (?:a\s+)?digits?)\s+(?:at least one\s+)?(?:the\s+)?(?:digits?\s+)?(?P<nums>{_number_list_pattern()})",
            _build_contains_digit_term,
        ),
        (
            rf"at least one (?:decimal )?digit (?:equal to|equals?|is)\s+(?P<nums>{_number_list_pattern()})",
            _build_contains_digit_term,
        ),
        (
            rf"(?:contains?|containing)\s+(?P<neg>no)\s+digits?\s+(?P<nums>{_number_list_pattern()})",
            _build_contains_digit_term,
        ),
        (
            rf"tens digit(?:\s+(?:is|equals?|equal to|of))?\s+(?P<nums>{_number_list_pattern()})",
            _build_tens_digit_term,
        ),
        (
            rf"(?:starts? with|starting with|beginning with|(?:first|leading) digit(?: is| equals?)?)\s+(?P<nums>{_number_list_pattern()})",
            _build_starting_digit_term,
        ),
    )


def _parse_digit_sum_terms(text: str, domain: int) -> list[Term]:
    prefix = _DIGIT_SUM_PREFIX
    # Numeric digit-sum constraints take priority before parity/prime/square properties.
    terms = _parse_terms(text, domain, (
        (rf"{prefix}\s+(?:is|equals?|equal to|to|of)?\s*(?:either |one of )?(?P<nums>{_number_list_pattern()})", _build_digit_sum_exact_term),
        (rf"{prefix}\s+(?:(?:is|to)\s+)?(?:a\s+)?(?:not\s+)?(?:divisible by|multiple of)\s+(?P<nums>{_number_list_pattern()})", _build_digit_sum_multiple_term),
        (rf"{prefix}\s+(?:is\s+)?(?:>|greater than|more than)\s+({_NUMBER_TOKEN_RE.pattern})", _build_digit_sum_greater_term),
        (rf"{prefix}\s+(?:is\s+)?(?:<|less than|fewer than)\s+({_NUMBER_TOKEN_RE.pattern})", _build_digit_sum_less_term),
        (rf"{prefix}\s+(?:is\s+)?(?:<=|less than or equal to|at most|no more than)\s+({_NUMBER_TOKEN_RE.pattern})", _build_digit_sum_at_most_term),
        (rf"{prefix}\s+(?:is\s+)?(?:>=|greater than or equal to|at least|no less than)\s+({_NUMBER_TOKEN_RE.pattern})", _build_digit_sum_at_least_term),
        (rf"{prefix}\s+(?:is\s+)?({_NUMBER_TOKEN_RE.pattern})\s+or\s+less\b", _build_digit_sum_at_most_term),
        (rf"{prefix}\s+(?:is\s+)?({_NUMBER_TOKEN_RE.pattern})\s+or\s+greater\b", _build_digit_sum_at_least_term),
    ))

    for match in re.finditer(rf"{prefix}\s+(?:is\s+|are\s+|to a value\s+)?(?P<strict>strictly\s+)?(?:between|from)\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:and|to)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        lo, hi = _parse_number(match.group(2)), _parse_number(match.group(3))
        if lo is not None and hi is not None:
            op = "<" if match.group("strict") else "<="
            terms.append(_term(f"{lo} {op} _digit_sum(x) {op} {hi}", domain))

    for pattern, expr in (
        (rf"{prefix}\s+(?:(?:is|to)\s+)?(?:an?\s+)?even\b", "_digit_sum(x) % 2 == 0"),
        (rf"{prefix}\s+(?:(?:is|to)\s+)?(?:an?\s+)?odd\b", "_digit_sum(x) % 2 == 1"),
        (rf"{prefix}\s+(?:(?:is|to)\s+)?(?:an?\s+)?prime\b", "_is_prime(_digit_sum(x))"),
        (rf"\bprime {prefix}\b", "_is_prime(_digit_sum(x))"),
        (rf"\b(?:perfect )?square {prefix}\b", "_is_square(_digit_sum(x))"),
        (rf"\beven {prefix}\b", "_digit_sum(x) % 2 == 0"),
        (rf"\bodd {prefix}\b", "_digit_sum(x) % 2 == 1"),
        (rf"{prefix}\s+(?:is\s+|to\s+)?(?:a\s+)?(?:perfect\s+)?square\b", "_is_square(_digit_sum(x))"),
    ):
        match = re.search(rf"{pattern}(?:\s+or\s+(?P<nums>{_number_list_pattern()}))?", text)
        if match:
            if match.group("nums"):
                expr = f"({expr}) or _digit_sum(x) in {_set_literal(_extract_numbers(match.group('nums')))}"
            terms.append(_term(expr, domain))

    product_prefix = _DIGIT_PRODUCT_PREFIX
    terms.extend(_parse_terms(text, domain, (
        (rf"{product_prefix}\s+(?:is|equals?|to)?\s*(?P<nums>{_number_list_pattern()})", _build_digit_product_exact_term),
    )))
    for match in re.finditer(rf"{product_prefix}\s+(?:is\s+)?(odd|even)(?:\s+or\s+(?P<nums>{_number_list_pattern()}))?", text):
        expr = f"_digit_product(x) % 2 == {int(match.group(1) == 'odd')}"
        if match.group("nums"):
            expr += f" or _digit_product(x) in {_set_literal(_extract_numbers(match.group('nums')))}"
        terms.append(_term(expr, domain))
    return terms


def _build_digit_sum_exact_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    if re.match(r"\s+or\s+(?:less|greater)\b", text[match.end():]):
        return None
    nums = [n for n in _extract_numbers(match.group("nums")) if n != 0]
    if nums:
        return _term(f"_digit_sum(x) in {_set_literal(nums)}", domain)
    return None


def _build_digit_sum_multiple_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    negated = re.search(r"\bnot\s+(?:divisible by|a multiple of|multiple of)\b", match.group(0))
    nums = _extract_numbers(match.group("nums"))
    if not nums:
        return None
    connector = _connector(match.group("nums"))
    joiner = " and " if connector == "and" else " or "
    op = "!=" if negated else "=="
    return _term(joiner.join(f"_digit_sum(x) % {n} {op} 0" for n in nums), domain)


def _build_digit_sum_bound_term(match: re.Match[str], domain: int, op: str) -> Term | None:
    value = _parse_number(match.group(1))
    if value is not None:
        return _term(f"_digit_sum(x) {op} {value}", domain)
    return None


def _build_digit_sum_greater_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    return _build_digit_sum_bound_term(match, domain, ">")


def _build_digit_sum_less_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    return _build_digit_sum_bound_term(match, domain, "<")


def _build_digit_sum_at_most_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    return _build_digit_sum_bound_term(match, domain, "<=")


def _build_digit_sum_at_least_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    return _build_digit_sum_bound_term(match, domain, ">=")


def _build_digit_product_exact_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    nums = _extract_numbers(match.group("nums"))
    if nums:
        return _term(f"_digit_product(x) in {_set_literal(nums)}", domain)
    return None


def _parse_digit_count_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(_DIGIT_COUNT_CLAUSE, text):
        count = 1 if match.group("single") else _parse_number(match.group("hyphen_count") or match.group("count"))
        if count is not None and count > 0:
            expr = "False" if count > len(str(domain)) else f"{10 ** (count - 1)} <= x <= {10 ** count - 1}"
            terms.append(_term(expr, domain, interval=True))

    for match in re.finditer(rf"(?:at least one|a) digit\s+(greater than|less than|equal to)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        value = _parse_number(match.group(2))
        if value is not None:
            op = {"greater than": ">", "less than": "<", "equal to": "=="}[match.group(1)]
            terms.append(_term(f"any(int(d) {op} {value} for d in str(abs(x)))", domain))
    for match in re.finditer(rf"digits (?:are )?(?:all |only )?(?:in|drawn from (?:the )?set|belonging to)\s*\{{(?P<nums>{_number_list_pattern()})\}}", text):
        digits = _extract_numbers(match.group("nums"))
        if digits and all(0 <= n <= 9 for n in digits):
            terms.append(_term(f"all(int(d) in {_set_literal(digits)} for d in str(abs(x)))", domain))
    if re.search(_FIBONACCI_DIGIT_CLAUSE, text):
        terms.append(_term("all(int(d) in {0, 1, 2, 3, 5, 8} for d in str(x))", domain))
    for match in re.finditer(_DIGIT_PROPERTY_CLAUSE, text):
        position = match.group("position") or match.group("adjective_position")
        properties = match.group("properties") or match.group("adjectives")
        value = "int(str(x)[0])" if position in {"first", "leading"} else "(x // 10) % 10" if position == "tens" else "x % 10"
        alternatives = []
        for prop in properties.split(" or "):
            expr = f"_is_prime({value})" if prop.endswith("prime") else f"({value}) % 2 == {int(prop.endswith('odd'))}"
            alternatives.append(f"not ({expr})" if prop.startswith("not ") else f"({expr})")
        expr = " or ".join(alternatives)
        if match.group("adjective_neg"):
            expr = f"not ({expr})"
        if position == "tens":
            expr = f"x >= 10 and ({expr})"
        terms.append(_term(expr, domain))
    for match in re.finditer(rf"(?:tens|units?|ones) digit\s+(?:is )?(greater than|less than|equal to)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        value = _parse_number(match.group(2))
        if value is not None:
            op = {"greater than": ">", "less than": "<", "equal to": "=="}[match.group(1)]
            expr = f"x >= 10 and (x // 10) % 10 {op} {value}" if match.group(0).startswith("tens") else f"x % 10 {op} {value}"
            terms.append(_term(expr, domain))
    for match in re.finditer(rf"(?:containing|contains?|with) at least one (?:of (?:the )?digits\s+|digit from\s*\{{)(?P<nums>{_number_list_pattern()})\}}?", text):
        digits = _extract_numbers(match.group("nums"))
        if digits and all(0 <= n <= 9 for n in digits):
            terms.append(_term(f"any(int(d) in {_set_literal(digits)} for d in str(abs(x)))", domain))
    return terms


def _parse_interval_terms(text: str, domain: int) -> list[Term]:
    text = re.sub(_DIVISOR_COUNT_CLAUSE, " ", text)
    terms = []
    for match in re.finditer(rf"\bfirst\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:positive )?integers?\b", text):
        count = _parse_number(match.group(1))
        if count is not None:
            terms.append(_term(f"1 <= x <= {count}", domain, interval=True))
    range_patterns = (
        r"(?<![\w.])(-?\d+)\s*\.\.\s*(-?\d+)(?![\w.])",
        rf"(?:numbers?\s+)?from\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:to|through)\s+({_NUMBER_TOKEN_RE.pattern})",
        rf"(?<!from )({_NUMBER_TOKEN_RE.pattern})\s+through\s+({_NUMBER_TOKEN_RE.pattern})",
        rf"between\s+({_NUMBER_TOKEN_RE.pattern})\s+and\s+({_NUMBER_TOKEN_RE.pattern})",
        rf"(?:in\s+the\s+)?interval\s+({_NUMBER_TOKEN_RE.pattern})\s+to\s+({_NUMBER_TOKEN_RE.pattern})",
        rf"interval\s*\[?\s*({_NUMBER_TOKEN_RE.pattern})\s*,\s*({_NUMBER_TOKEN_RE.pattern})\s*\]?",
        rf"range\s*\[?\s*({_NUMBER_TOKEN_RE.pattern})\s*,\s*({_NUMBER_TOKEN_RE.pattern})\s*\]?",
        rf"range\s+({_NUMBER_TOKEN_RE.pattern})\s*(?:-|to)\s*({_NUMBER_TOKEN_RE.pattern})",
    )
    for pattern in range_patterns:
        for match in re.finditer(pattern, text):
            before = text[max(0, match.start() - 28):match.start()]
            if re.search(r"\bnot\s+(?:a\s+)?(?:number\s+)?(?:from|between|in\s+)?\s*$", before):
                continue
            lo = _parse_number(match.group(1))
            hi = _parse_number(match.group(2))
            if lo is not None and hi is not None:
                if lo > hi:
                    lo, hi = hi, lo
                op = "<" if re.search(r"\bstrictly\s*$", before) else "<="
                terms.append(_term(f"{lo} {op} x {op} {hi}", domain, interval=True))

    for match in re.finditer(rf"\[\s*({_NUMBER_TOKEN_RE.pattern})\s*,\s*({_NUMBER_TOKEN_RE.pattern})\s*\]", text):
        lo, hi = _parse_number(match.group(1)), _parse_number(match.group(2))
        if lo is not None and hi is not None:
            terms.append(_term(f"{lo} <= x <= {hi}", domain, interval=True))

    one_sided = (
        (rf"(?:greater|more)\s+than\s+or\s+equal\s+to\s+({_NUMBER_TOKEN_RE.pattern})", ">="),
        (rf"(?:less|fewer)\s+than\s+or\s+equal\s+to\s+({_NUMBER_TOKEN_RE.pattern})", "<="),
        (rf"(?:greater|more)\s+than\s+({_NUMBER_TOKEN_RE.pattern})", ">"),
        (rf"(?:less|fewer)\s+than\s+({_NUMBER_TOKEN_RE.pattern})", "<"),
        (rf"(?:at least|no less than)\s+({_NUMBER_TOKEN_RE.pattern})", ">="),
        (rf"(?:at most|no more than)\s+({_NUMBER_TOKEN_RE.pattern})", "<="),
        (rf"\b(?:above|over)\s+({_NUMBER_TOKEN_RE.pattern})", ">"),
        (rf"\b(?:below|under)\s+({_NUMBER_TOKEN_RE.pattern})", "<"),
        (rf"\bup to\s+({_NUMBER_TOKEN_RE.pattern})", "<="),
    )
    for pattern, op in one_sided:
        for match in re.finditer(pattern, text):
            # "5, 10, ... up to 100" ends an elided sequence, not a range on x.
            if re.search(r"\b(?:digit|digits|divisors|factors|adds?|adding|sums?|summing)\s*$|(?:\.\.\.|…)\s*,?\s*$", text[:match.start()]):
                continue
            if re.match(r"\s+(?:of (?:the )?)?(?:distinct\s+)?(?:digits?|(?:positive )?divisors?|(?:prime )?factors?|hypothes[ei]s|examples?|guesses|answers)\b", text[match.end():]):
                continue
            if re.search(r"\b(?:contains?|containing)\s*$", text[:match.start()]):
                continue
            if _digit_sum_context_before(text, match.start()):
                continue
            value = _parse_number(match.group(1))
            if value is not None:
                terms.append(_term(f"x {op} {value}", domain, interval=True))

    for match in re.finditer(rf"({_NUMBER_TOKEN_RE.pattern})\s*<=\s*[nx]\s*<=\s*({_NUMBER_TOKEN_RE.pattern})", text):
        lo = _parse_number(match.group(1))
        hi = _parse_number(match.group(2))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            terms.append(_term(f"{lo} <= x <= {hi}", domain, interval=True))

    for match in re.finditer(rf"[nx]\s*(<=|>=|<|>)\s*({_NUMBER_TOKEN_RE.pattern})", text):
        value = _parse_number(match.group(2))
        if value is not None:
            terms.append(_term(f"x {match.group(1)} {value}", domain, interval=True))

    for match in re.finditer(rf"({_NUMBER_TOKEN_RE.pattern})\s+or\s+greater\b", text):
        if _digit_sum_context_before(text, match.start()):
            continue
        value = _parse_number(match.group(1))
        if value is not None:
            terms.append(_term(f"x >= {value}", domain, interval=True))
    for match in re.finditer(rf"({_NUMBER_TOKEN_RE.pattern})\s+or\s+less\b", text):
        if _digit_sum_context_before(text, match.start()):
            continue
        value = _parse_number(match.group(1))
        if value is not None:
            terms.append(_term(f"x <= {value}", domain, interval=True))

    if re.search(r"\bnumbers? in the \d0s\b", text):
        decades = sorted({int(value) * 10 for value in re.findall(r"\b(\d)0s\b", text)})
        if decades:
            if decades == list(range(decades[0], decades[-1] + 1, 10)):
                terms.append(_term(f"{decades[0]} <= x <= {decades[-1] + 9}", domain, interval=True))
            else:
                terms.append(_combine_terms([_term(f"{value} <= x <= {value + 9}", domain, interval=True) for value in decades], "or"))
    return terms


def _parse_modulo_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(rf"remainder (?:of )?({_NUMBER_TOKEN_RE.pattern})\s+(?:when divided by|upon division by|on division by)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        residue, modulus = _parse_number(match.group(1)), _parse_number(match.group(2))
        if modulus and residue is not None:
            terms.append(_term(f"x % {modulus} == {residue % modulus}", domain))
    for match in re.finditer(rf"divisible by\s+({_NUMBER_TOKEN_RE.pattern})\s+with (?:a )?remainder\s+({_NUMBER_TOKEN_RE.pattern})", text):
        modulus, residue = _parse_number(match.group(1)), _parse_number(match.group(2))
        if modulus and residue is not None:
            terms.append(_term(f"x % {modulus} == {residue % modulus}", domain))
    for match in re.finditer(rf"multiples?\s+of\s+({_NUMBER_TOKEN_RE.pattern})\s+plus\s+(?P<nums>{_number_list_pattern()})", text):
        modulus = _parse_number(match.group(1))
        residues = _extract_numbers(match.group("nums"))
        if residues and modulus:
            joiner = f" {_connector(match.group('nums'))} "
            terms.append(_term(joiner.join(f"x % {modulus} == {residue % modulus}" for residue in residues), domain))

    for match in re.finditer(rf"[nx]\s*(?:%|mod)\s*({_NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*(?P<nums>{_number_list_pattern()})", text):
        modulus = _parse_number(match.group(1))
        residues = _extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(_term(f"x % {modulus} in {_set_literal(r % modulus for r in residues)}", domain))

    for match in re.finditer(rf"not\s+congruent to\s+(?P<nums>{_number_list_pattern()})\s+(?:mod|modulo)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        modulus = _parse_number(match.group(2))
        residues = _extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(_term(f"x % {modulus} not in {_set_literal(r % modulus for r in residues)}", domain))

    for match in re.finditer(rf"mod\s+({_NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*(?P<nums>{_number_list_pattern()})", text):
        modulus = _parse_number(match.group(1))
        residues = _extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(_term(f"x % {modulus} in {_set_literal(r % modulus for r in residues)}", domain))

    for match in re.finditer(rf"(?:congruent to|mod(?:ulo)?)\s+({_NUMBER_TOKEN_RE.pattern})\s+(?:mod|modulo)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        if re.search(r"\bnot\s+$", text[max(0, match.start() - 8):match.start()]):
            continue
        residue = _parse_number(match.group(1))
        modulus = _parse_number(match.group(2))
        if residue is not None and modulus:
            terms.append(_term(f"x % {modulus} == {residue % modulus}", domain))

    for match in re.finditer(rf"congruent to\s+(?P<nums>{_number_list_pattern()})\s+(?:mod|modulo)\s+({_NUMBER_TOKEN_RE.pattern})", text):
        if re.search(r"\bnot\s+$", text[max(0, match.start() - 8):match.start()]):
            continue
        modulus = _parse_number(match.group(2))
        residues = _extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(_term(f"x % {modulus} in {_set_literal(r % modulus for r in residues)}", domain))
    return terms


def _parse_modulo_interval_or_union(text: str, domain: int) -> Term | None:
    mod_expr = rf"[nx]\s*(?:%|mod)\s*({_NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*({_NUMBER_TOKEN_RE.pattern})"
    interval_expr = rf"({_NUMBER_TOKEN_RE.pattern})\s*<=\s*[nx]\s*<=\s*({_NUMBER_TOKEN_RE.pattern})"
    match = re.search(rf"{mod_expr}\s+or\s+{interval_expr}", text)
    if match:
        modulus = _parse_number(match.group(1))
        residue = _parse_number(match.group(2))
        lo = _parse_number(match.group(3))
        hi = _parse_number(match.group(4))
        if modulus and residue is not None and lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return _combine_terms([_term(f"x % {modulus} == {residue % modulus}", domain), _term(f"{lo} <= x <= {hi}", domain, interval=True)], "or")

    match = re.search(rf"(?:multiples?|multiple)\s+of\s+({_NUMBER_TOKEN_RE.pattern})\s+or\s+(?:numbers?\s+)?from\s+({_NUMBER_TOKEN_RE.pattern})\s+to\s+({_NUMBER_TOKEN_RE.pattern})", text)
    if match:
        modulus = _parse_number(match.group(1))
        lo = _parse_number(match.group(2))
        hi = _parse_number(match.group(3))
        if modulus and lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return _combine_terms([_term(f"x % {modulus} == 0", domain), _term(f"{lo} <= x <= {hi}", domain, interval=True)], "or")
    return None


def _parse_form_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(rf"prime factorization (?:contains?|containing|includes?)\s+({_NUMBER_TOKEN_RE.pattern})\s*\^\s*({_NUMBER_TOKEN_RE.pattern})", text):
        base, power = _parse_number(match.group(1)), _parse_number(match.group(2))
        if base is not None and power is not None and _is_prime(base) and power > 0:
            terms.append(_term(f"x % {base ** power} == 0", domain))
    if re.search(r"\bnot\s+(?:a\s+)?powers? of 2\b|\bnot\s+2\^n\b", text):
        terms.append(_term("not (x > 0 and (x & (x - 1)) == 0)", domain))
        return terms
    if re.search(r"\b(?:numbers of the form )?2\^n\b|\bpowers? of 2\b", text):
        terms.append(_term("x > 0 and (x & (x - 1)) == 0", domain))

    for match in re.finditer(r"\b[nx]\s*=\s*(\d+)\s*[a-z]\b", text):
        terms.append(_term(f"x % {int(match.group(1))} == 0", domain))

    for match in re.finditer(r"\b[nx]\s*=\s*(\d+)\s*\*?\s*[a-z]\s*\^\s*2\b", text):
        coeff = int(match.group(1))
        terms.append(_term(f"x % {coeff} == 0 and _is_square(x // {coeff})", domain))

    for match in re.finditer(r"\b[nx]\s*=\s*(\d+)\s*\^\s*[a-z](?:\s*where\s*[a-z]\s*>=\s*(\d+))?", text):
        base = int(match.group(1))
        start = int(match.group(2) or 0)
        terms.append(_term(f"any({base} ** n == x for n in range({start}, 16))", domain))

    for match in re.finditer(r"\b[nx]\s*\^\s*(\d+)\s*=\s*(\d+)\b", text):
        power = int(match.group(1))
        value = int(match.group(2))
        root = round(value ** (1 / power))
        support = {n for n in (root - 1, root, root + 1) if n > 0 and n ** power == value}
        if support:
            terms.append(Term(frozenset(n for n in support if n <= domain), f"x in {_set_literal(support)}", interval=len(support) == 1))

    for match in re.finditer(r"\b(\d+)\s*n\s*([+-])\s*(\d+)\b", text):
        modulus = int(match.group(1))
        offset = int(match.group(3)) * (1 if match.group(2) == "+" else -1)
        terms.append(_term(f"(x - ({offset})) % {modulus} == 0", domain))
    for match in re.finditer(r"\b(\d+)\s*[a-z]\s*([+-])\s*(\d+)\b", text):
        modulus = int(match.group(1))
        offset = int(match.group(3)) * (1 if match.group(2) == "+" else -1)
        terms.append(_term(f"(x - ({offset})) % {modulus} == 0", domain))
    for match in re.finditer(rf"({_NUMBER_TOKEN_RE.pattern})\s+(more|less)\s+than\s+a\s+multiple\s+of\s+({_NUMBER_TOKEN_RE.pattern})", text):
        offset = _parse_number(match.group(1))
        direction = match.group(2)
        modulus = _parse_number(match.group(3))
        if offset is not None and modulus:
            residue = offset if direction == "more" else -offset
            terms.append(_term(f"x % {modulus} == {residue % modulus}", domain))
    for match in re.finditer(r"\b(\d+)\s*n\b(?!\s*[+-])", text):
        terms.append(_term(f"x % {int(match.group(1))} == 0", domain))
    return terms


def _parse_shifted_power_terms(text: str, domain: int) -> list[Term]:
    terms = []
    if re.search(r"\b(?:twice|two times|2 times) (?:a )?prime\b", text):
        terms.append(_term("x % 2 == 0 and _is_prime(x // 2)", domain))
    for match in re.finditer(r"\b(\d+)\s*\^\s*n\s*([+*])\s*(\d+)\b", text):
        base, op, value = int(match.group(1)), match.group(2), int(match.group(3))
        if op == "+":
            terms.append(_term(f"any({base} ** n + {value} == x for n in range(0, 16))", domain))
        else:
            terms.append(_term(f"any({base} ** n * {value} == x for n in range(0, 16))", domain))

    for match in re.finditer(r"\bperfect squares?\s+(?:plus|\+)\s+(-?\d+)\b", text):
        offset = int(match.group(1))
        terms.append(_term(f"x > {offset} and _is_square(x - ({offset}))", domain))

    for match in re.finditer(rf"\bsquare of\s+({_NUMBER_TOKEN_RE.pattern})\b", text):
        value = _parse_number(match.group(1))
        if value is not None:
            squared = value * value
            if 1 <= squared <= domain:
                terms.append(_term(f"x == {squared}", domain, interval=True))

    if re.search(r"\bperfect square multiplied by a prime\b|\bperfect square times a prime\b", text):
        terms.append(_term("any(a * a * p == x for a in range(1, math.isqrt(x) + 1) for p in range(2, x + 1) if _is_prime(p))", domain))
    if re.search(r"\bprime or (?:a )?(?:perfect )?square of a prime\b|\bprime numbers? or squares? of primes?\b", text):
        terms.append(_term("_is_prime(x) or (_is_square(x) and _is_prime(math.isqrt(x)))", domain))
    if re.search(r"\b(?:perfect )?squares? or prime numbers?\b|\bprime numbers? or (?:perfect )?squares?\b", text):
        terms.append(_term("_is_square(x) or _is_prime(x)", domain))
    if re.search(r"\b(?:is|equals?)\s+(?:the\s+)?(?:sum|product) of (?:its|the) digits\b", text):
        terms.append(_term("x == _digit_sum(x) or x == _digit_product(x)", domain))
    if re.search(r"\bsum of (?:the |its )?cubes of (?:the |its )?digits\b", text):
        terms.append(_term("x == sum(int(ch) ** 3 for ch in str(abs(x)))", domain))
    for match in re.finditer(rf"\b(?:number )?(?:equals?|is)\s+(?:the\s+)?digit sum times\s+({_NUMBER_TOKEN_RE.pattern})", text):
        multiplier = _parse_number(match.group(1))
        if multiplier is not None:
            terms.append(_term(f"x == _digit_sum(x) * {multiplier}", domain))
    if re.search(r"\bproduct of the sum of (?:its|the) digits and the sum of (?:its|the) digits\b", text):
        terms.append(_term("x == _digit_sum(x) ** 2", domain))
    if re.search(r"\b(?:100\s*-\s*[nx]|100 minus the number).*?\bodd\b", text):
        terms.append(_term("x % 2 == 1", domain))
    if re.search(r"\b(?:100\s*-\s*[nx]|100 minus the number).*?\beven\b", text):
        terms.append(_term("x % 2 == 0", domain))

    for match in re.finditer(rf"squares? of (?:an? )?(?:numbers?|integers?) ending in\s+(?P<nums>{_number_list_pattern()})", text):
        nums = [n for n in _extract_numbers(match.group("nums")) if 0 <= n <= 9]
        if nums:
            terms.append(_term(f"_is_square(x) and math.isqrt(x) % 10 in {_set_literal(nums)}", domain))

    for match in re.finditer(rf"perfect cube(?:s)? (?:where the base is less than|of a number that is either)\s+(?P<nums>{_number_list_pattern()})", text):
        nums = [n for n in _extract_numbers(match.group("nums")) if n > 0]
        if nums:
            if "less than" in match.group(0):
                support = {n ** 3 for n in range(1, max(nums))}
            else:
                support = {n ** 3 for n in nums}
            support = {x for x in support if 1 <= x <= domain}
            if support:
                terms.append(Term(frozenset(support), f"x in {_set_literal(support)}"))
    for match in re.finditer(rf"\bprime(?: number)?s? whose square ends in\s+(?P<nums>{_number_list_pattern()})", text):
        nums = _extract_numbers(match.group("nums"))
        nums = [n for n in nums if 0 <= n <= 9]
        if nums:
            terms.append(_term(f"_is_prime(x) and (x * x) % 10 in {_set_literal(nums)}", domain))
    return terms


def _parse_sum_representation_terms(text: str, domain: int) -> list[Term]:
    terms = []
    if re.search(r"\bsum of two distinct primes\b", text):
        terms.append(_term("any(a + b == x and a != b and _is_prime(a) and _is_prime(b) for a in range(2, x) for b in range(2, x))", domain))
    elif re.search(r"\bsum of two primes\b", text):
        terms.append(_term("any(a + b == x and _is_prime(a) and _is_prime(b) for a in range(2, x) for b in range(2, x))", domain))
    if re.search(r"\bsum of two squares\b", text):
        terms.append(_term("any(a * a + b * b == x for a in range(0, math.isqrt(x) + 1) for b in range(0, math.isqrt(x) + 1))", domain))
    if re.search(r"\bperfect square plus a prime\b|\bsquare plus a prime\b", text):
        terms.append(_term("any(_is_square(a) and _is_prime(x - a) for a in range(1, x))", domain))
    if re.search(r"\bprime and a square of a prime\b|\bprime plus (?:a )?square of a prime\b", text):
        terms.append(_term("any(_is_prime(p) and _is_square(q) and _is_prime(math.isqrt(q)) and p + q == x for p in range(2, x) for q in range(1, x))", domain))

    ordinal_primes = _ORDINAL_PRIME_RE.fullmatch(text)
    nths = [int(n) for n in re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", text)] if ordinal_primes else []
    if nths:
        primes = [x for x in range(2, max(domain * 3, 128)) if _is_prime(x)]
        support = {primes[n - 1] for n in nths if 0 < n <= len(primes) and 1 <= primes[n - 1] <= domain}
        if support:
            terms.append(Term(frozenset(support), f"x in {_set_literal(support)}", interval=len(support) == 1))
    return terms


def _parse_consecutive_sum_terms(text: str, domain: int) -> list[Term]:
    terms = []
    if re.search(r"sum of two consecutive odd integers", text):
        terms.append(_term("x % 4 == 0", domain))
    elif re.search(r"sum of two consecutive even integers", text):
        terms.append(_term("x % 4 == 2", domain))
    elif re.search(r"sum of two consecutive integers where the first is odd", text):
        terms.append(_term("x % 4 == 3", domain))
    elif re.search(r"sum of two consecutive integers where the first is even", text):
        terms.append(_term("x % 4 == 1", domain))
    elif re.search(r"sum of two consecutive integers", text):
        terms.append(_term("x % 2 == 1", domain))

    for match in re.finditer(rf"sum of two consecutive integers where the first is\s+({_NUMBER_TOKEN_RE.pattern})", text):
        first = _parse_number(match.group(1))
        if first is not None:
            terms.append(_term(f"x == {2 * first + 1}", domain, interval=True))

    for match in re.finditer(rf"sum of two (?:positive )?integers where the difference between (?:the )?two numbers is\s+({_NUMBER_TOKEN_RE.pattern})", text):
        diff = _parse_number(match.group(1))
        if diff is not None:
            terms.append(_term(f"x > {diff} and (x - {diff}) % 2 == 0", domain))
    return terms


def _parse_simple_exclusion(raw: str, text: str, domain: int) -> ParsedPredicate | None:
    patterns = (
        rf"(?:the\s+)?numbers?\s+(?:is\s+|are\s+)?not\s+(?:equal\s+to\s+)?(?P<nums>{_number_list_pattern()})",
        rf"(?:the\s+)?numbers?\s+(?:is\s+|are\s+)?not\s+(?:the\s+)?numbers?\s+(?P<nums>{_number_list_pattern()})",
        rf"numbers?\s+not\s+equal\s+to\s+(?P<nums>{_number_list_pattern()})",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            nums = [n for n in _extract_numbers(match.group("nums")) if 1 <= n <= domain]
            if nums:
                return _predicate(raw, [_term(f"x not in {_set_literal(nums)}", domain)], domain, forced_kind="rule")
    return None


def _has_exclusion_marker(text: str) -> bool:
    text = re.sub(_DIGIT_PROPERTY_CLAUSE, " ", text)
    # These atoms already implement negation; an unrecognized exclusion still
    # rejects the whole description instead of silently keeping its positive part.
    atom = (
        rf"(?:prime(?: numbers?)?|odd(?: numbers?)?|even(?: numbers?)?|"
        rf"(?:perfect )?squares?|(?:perfect )?cubes?|"
        rf"(?:multiples? of|divisible by|powers? of|congruent to|"
        rf"ending in|ends? in|contains?|containing)\s+(?:(?:both|either)\s+)?{_number_list_pattern()})"
    )
    if re.search(r"\bnot\b.*\bor\b.*\bsimultaneously\b", text):
        return True
    rest = re.sub(rf"\bnot\s+(?:an?\s+)?{atom}\b", " ", text)
    rest = re.sub(rf"(?:divisible by|multiples? of)\s+{_number_list_pattern()}\s*,?\s+but not by\s+{_number_list_pattern()}", " ", rest)
    return bool(re.search(r"\b(?:not|except|excluding|exclusive of)\b", rest))


def _parse_explicit_set(text: str, domain: int) -> Term | None:
    text = re.sub(r",? which are the only numbers in the range \d+ to \d+ that satisfy the condition$", "", text)
    subject = r"(?:(?:(?:only )?(?:the )?numbers?(?: (?:is|are|that are))?|the accepted set is)\s+)?"
    values = rf"(?P<nums>{_number_list_pattern()})"
    patterns = (
        rf"{subject}(?:(?:is )?in(?: (?:the )?set)?\s*)?\{{\s*{values}\s*\}}",
        rf"{subject}(?:in (?:the )?(?:set|list))\s*\[\s*{values}\s*\]",
        rf"{subject}(?:(?:one of|either|exactly these|in the set|in the list)\s+)?{values}(?: specifically)?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            nums = _extract_numbers(match.group("nums"))
            if len(nums) >= 2:
                support = frozenset(n for n in nums if 1 <= n <= domain)
                return Term(support, f"x in {_set_literal(nums)}")
    return None


def _parse_bounded_form(text: str, domain: int) -> Term | None:
    token = _NUMBER_TOKEN_RE.pattern
    subject = r"(?:(?:numbers?|integers?)\s+(?:(?:of the form|equal to)\s+)?)?"
    bounds = rf"(?:from|between)\s+({token})\s+(?:to|and)\s+({token})(?: inclusive)?"
    match = re.fullmatch(
        r"(?:(?:the )?(?:input )?numbers?(?: (?:is|are))? )?(?:a )?(?:multiples? of|divisible by)\s+(\d+)\.(\d+)", text,
    )
    if match:
        numerator = int(match.group(1) + match.group(2))
        denominator = 10 ** len(match.group(2))
        return _term(f"(x * {denominator}) % {numerator} == 0", domain) if numerator else None
    match = re.fullmatch(rf"(?:numbers? )?{bounds} in increments of ({token})", text)
    if match:
        lo, hi, step = (_parse_number(match.group(i)) for i in (1, 2, 3))
        if lo is not None and hi is not None and step is not None and step > 0:
            return _term(f"{lo} <= x <= {hi} and (x - {lo}) % {step} == 0", domain)
    match = re.fullmatch(rf"{subject}({token})\s*\*?\s*n,? where n is (?P<nums>{_number_list_pattern()})", text)
    if match:
        factor = _parse_number(match.group(1))
        if factor is not None:
            return _term(f"x in {_set_literal(factor * n for n in _extract_numbers(match.group('nums')))}", domain)
    match = re.fullmatch(rf"{subject}n\^([23])\s+for\s+(?:(?:an?|some)?\s*integer\s+)?n\s+{bounds}", text)
    if match:
        power = int(match.group(1))
        lo, hi = _parse_number(match.group(2)), _parse_number(match.group(3))
        if lo is not None and hi is not None:
            return _term(f"any(n ** {power} == x for n in range({lo}, {hi + 1}))", domain)
    # A bound after "squares of integers" constrains the root, not x.
    match = re.fullmatch(
        rf"(?:(?:the )?numbers? (?:that|which) are )?(?:the )?(?:perfect )?(?P<power>squares|cubes) of (?:the )?(?:positive )?(?:integers|numbers)\s+"
        rf"(?:(?:from\s+)?(?P<lo>{token})\s*(?:to|through|-)\s*(?P<hi>{token})|between\s+(?P<lo2>{token})\s+and\s+(?P<hi2>{token})"
        rf"|(?P<op>under|below|less than or equal to|less than|at most|up to|greater than or equal to|greater than|more than|above|over|at least)\s+(?P<bound>{token}))",
        text,
    )
    if match:
        power = 2 if match.group("power") == "squares" else 3
        if match.group("op"):
            bound = _parse_number(match.group("bound"))
            lo, hi = {
                "under": (1, bound - 1), "below": (1, bound - 1), "less than": (1, bound - 1),
                "less than or equal to": (1, bound), "at most": (1, bound), "up to": (1, bound),
                "greater than": (bound + 1, domain), "more than": (bound + 1, domain),
                "above": (bound + 1, domain), "over": (bound + 1, domain),
                "greater than or equal to": (bound, domain), "at least": (bound, domain),
            }[match.group("op")] if bound is not None else (None, None)
        else:
            lo = _parse_number(match.group("lo") or match.group("lo2"))
            hi = _parse_number(match.group("hi") or match.group("hi2"))
        if lo is not None and hi is not None:
            return _term(f"any(n ** {power} == x for n in range({lo}, {hi + 1}))", domain)
    match = re.fullmatch(rf"{subject}({token})\s+times\s+(?:an?|some) integer\s+{bounds}", text)
    if match:
        factor, lo, hi = (_parse_number(match.group(i)) for i in (1, 2, 3))
        if factor is not None and lo is not None and hi is not None:
            return _term(f"any({factor} * n == x for n in range({lo}, {hi + 1}))", domain)
    return None


def _parse_multiple_or_singleton_union(text: str, domain: int) -> Term | None:
    for match in re.finditer(rf"[nx]\s*(?:%|mod)\s*({_NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*0\s+or\s+[nx]\s*(?:==|=|is)\s*({_NUMBER_TOKEN_RE.pattern})", text):
        modulus = _parse_number(match.group(1))
        single = _parse_number(match.group(2))
        if modulus and single is not None:
            return _term(f"x % {modulus} == 0 or x == {single}", domain)

    pattern = re.compile(
        rf"(?:multiples?|multiple)\s+of\s+(?P<nums>{_number_list_pattern()})"
        rf"\s*,?\s*or\s+(?:the\s+)?number\s+is\s+(?P<single>{_NUMBER_TOKEN_RE.pattern})"
    )
    match = pattern.search(text)
    if not match:
        return None
    nums = [n for n in _extract_numbers(match.group("nums")) if n != 0]
    single = _parse_number(match.group("single"))
    if not nums or single is None:
        return None
    expr_parts = [f"x % {n} == 0" for n in nums]
    expr_parts.append(f"x == {single}")
    return _term(" or ".join(expr_parts), domain)


def _parse_only_singleton(text: str, domain: int) -> Term | None:
    patterns = (
        rf"(?:returns?|outputs?|says?)\s+'?yes'?\s+only\s+for\s+({_NUMBER_TOKEN_RE.pattern})",
        rf"(?:accepts?|returns?|outputs?)\s+only\s+(?:the\s+)?(?:number\s+)?({_NUMBER_TOKEN_RE.pattern})",
        rf"only\s+accepts?\s+(?:the\s+)?(?:number\s+)?({_NUMBER_TOKEN_RE.pattern})",
        rf"only\s+the\s+number\s+({_NUMBER_TOKEN_RE.pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if re.search(r"\bnot\s+$", text[max(0, match.start() - 8):match.start()]):
                continue
            value = _parse_number(match.group(1))
            if value is not None and 1 <= value <= domain:
                return _term(f"x == {value}", domain, interval=True)
    return None


def _parse_first_n_pattern(text: str, domain: int, *, total: int | None = None) -> Term | None:
    match = re.fullmatch(
        rf"(?:(?:the )?numbers?(?: are)? )?(?:the )?first\s+({_NUMBER_TOKEN_RE.pattern})\s+"
        rf"(?:multiples of\s+({_NUMBER_TOKEN_RE.pattern})|prime numbers?|primes|(?:perfect )?(?P<power>squares?|cubes?)|(?P<parity>odd|even) numbers?)"
        rf"(?:\s+(?:are\s+)?\((?P<examples>[\d\s,.;…]+)\))?"
        rf"(?:\s+plus (?:the )?(?:specific )?numbers?\s+(?P<extras>{_number_list_pattern()}))?", text,
    )
    if match:
        count = _parse_number(match.group(1))
        base = _parse_number(match.group(2)) if match.group(2) else None
        if count is not None and (base is None or base > 0):
            if base:
                sequence = list(range(base, domain + 1, base))
            elif match.group("parity"):
                sequence = list(range(1 if match.group("parity") == "odd" else 2, domain + 1, 2))
            else:
                power = match.group("power") or ""
                accepts = _is_square if power.startswith("square") else _is_cube if power.startswith("cube") else _is_prime
                sequence = [x for x in range(1, domain + 1) if accepts(x)]
            examples = match.group("examples")
            if examples is not None:
                values = [int(n) for n in re.findall(r"\d+", examples)]
                elided = "..." in examples or "…" in examples
                if not values or (not elided and len(values) != count):
                    return None
                positions = [*range(1, len(values)), count] if elided else range(1, count + 1)
                if base:
                    expected = [base * n for n in positions]
                elif match.group("power"):
                    power = 2 if match.group("power").startswith("square") else 3
                    expected = [n ** power for n in positions]
                elif match.group("parity"):
                    expected = [2 * n - int(match.group("parity") == "odd") for n in positions]
                else:
                    primes = list(sequence)
                    candidate = domain + 1
                    while len(primes) < max(positions, default=0):
                        if _is_prime(candidate):
                            primes.append(candidate)
                        candidate += 1
                    expected = [primes[n - 1] for n in positions if 1 <= n <= len(primes)]
                if values != expected:
                    return None
            support = frozenset(sequence[:max(0, count)])
            extras = set(_extract_numbers(match.group("extras"))) if match.group("extras") else set()
            if total is not None:
                # The declared sequence count precedes domain clipping. Check
                # added values against the full sequence to avoid double-counting.
                if base:
                    duplicates = {x for x in extras if 0 < x <= count * base and x % base == 0}
                elif match.group("parity"):
                    parity = int(match.group("parity") == "odd")
                    duplicates = {x for x in extras if 0 < x <= 2 * count - parity and x % 2 == parity}
                elif match.group("power"):
                    power = 2 if match.group("power").startswith("square") else 3
                    accepts = _is_square if power == 2 else _is_cube
                    duplicates = {x for x in extras if 0 < x <= count ** power and accepts(x)}
                else:
                    primes = list(sequence)
                    candidate = domain + 1
                    while extras and len(primes) < count:
                        if _is_prime(candidate):
                            primes.append(candidate)
                        candidate += 1
                    duplicates = extras & set(primes[:max(0, count)])
                if total != max(0, count) + len(extras - duplicates):
                    return None
            support |= frozenset(x for x in extras if 1 <= x <= domain)
            return Term(support, f"x in {_set_literal(support)}")
    if total is not None:
        return None
    match = re.search(rf"first\s+({_NUMBER_TOKEN_RE.pattern})\s+(odd|even) numbers?", text)
    if match and not re.search(r"\bplus\b", text[match.end():]):
        count = _parse_number(match.group(1))
        parity = match.group(2)
        if count is not None:
            start = 1 if parity == "odd" else 2
            support = {start + 2 * i for i in range(count)}
            support = frozenset(x for x in support if 1 <= x <= domain)
            return Term(support, f"x in {_set_literal(support)}")

    match = re.search(rf"first\s+({_NUMBER_TOKEN_RE.pattern})\s+even numbers?.*?plus(?: the)? number\s+({_NUMBER_TOKEN_RE.pattern})", text)
    if match:
        count = _parse_number(match.group(1))
        extra = _parse_number(match.group(2))
        if count is not None and extra is not None:
            support = {2 * i for i in range(1, count + 1)}
            support.add(extra)
            support = frozenset(x for x in support if 1 <= x <= domain)
            return Term(support, f"x in {_set_literal(support)}")

    match = re.search(rf"first\s+({_NUMBER_TOKEN_RE.pattern})\s+integers?.*?either even or perfect squares?", text)
    if match:
        count = _parse_number(match.group(1))
        if count is not None:
            support = frozenset(x for x in range(1, min(count, domain) + 1) if x % 2 == 0 or _is_square(x))
            return Term(support, f"x in {_set_literal(support)}")
    return None


def _parse_exact_singleton(text: str, domain: int) -> Term | None:
    if re.search(rf"\bnot\s+(?:equal to\s+|(?:the\s+)?number\s+)?(?:{_NUMBER_TOKEN_RE.pattern})\b", text):
        return None
    if re.fullmatch(rf"{_NUMBER_TOKEN_RE.pattern}", text):
        value = _parse_number(text)
        if value is not None and 1 <= value <= domain:
            return _term(f"x == {value}", domain, interval=True)

    patterns = (
        rf"(?:the )?(?:numbers?|integers?)(?: (?:is|are|that is|that are))?\s+(?:equals?|equal to|exactly|only)\s+({_NUMBER_TOKEN_RE.pattern})",
        rf"(?:only|exactly|equals?|(?<!or )equal to|number is|the number is|input is always)\s+(?:exactly\s+)?({_NUMBER_TOKEN_RE.pattern})\b(?!\s+(?:or\s+(?:less|greater)|divisors?|digits?|prime factors?))",
        rf"(?:is|must be)\s+(?:the\s+)?number\s+({_NUMBER_TOKEN_RE.pattern})\b(?!\s+(?:or\s+(?:less|greater)|divisors?|digits?|prime factors?))",
        rf"\b[nx]\s*(?:==|=)\s*({_NUMBER_TOKEN_RE.pattern})\b(?!\s*(?:\*|[a-z]\s*\^|\s+or\s+(?:less|greater)|\s+(?:divisors?|digits?|prime factors?)))",
        rf"\bnumbers? from\s+({_NUMBER_TOKEN_RE.pattern})\s+to\s+\1\b",
        rf"(?:(?:the )?numbers? (?:is|are) )?(?:exactly )?(?:the )?({_NUMBER_TOKEN_RE.pattern})(?:st|nd|rd|th)\s+positive integer",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            value = _parse_number(match.group(1))
            if value is not None and 1 <= value <= domain:
                return _term(f"x == {value}", domain, interval=True)
    return None


def _predicate(raw_label: str, terms: list[Term], domain: int, forced_kind: str | None = None) -> ParsedPredicate | None:
    if not terms:
        return None
    combined = terms[0] if len(terms) == 1 else _combine_terms(terms, "and")
    kind = forced_kind or ("interval" if all(term.interval for term in terms) else "rule")
    return ParsedPredicate(
        kind=kind,
        label=_library_label(raw_label),
        support=frozenset(x for x in combined.support if 1 <= x <= domain),
        expr=combined.expr,
    )


def _combine_terms(terms: list[Term], op: str) -> Term:
    if op == "or":
        support = frozenset().union(*(term.support for term in terms))
        expr = " or ".join(f"({term.expr})" for term in terms)
        return Term(support, expr)
    support = set(terms[0].support)
    for term in terms[1:]:
        support &= term.support
    expr = " and ".join(f"({term.expr})" for term in terms)
    return Term(frozenset(support), expr, interval=all(term.interval for term in terms))


@lru_cache(maxsize=None)
def _term(expr: str, domain: int, *, interval: bool = False) -> Term:
    support = frozenset(x for x in range(1, domain + 1) if _eval_expr(expr, x))
    return Term(support, expr, interval=interval)


def _dedupe_terms(terms: Iterable[Term]) -> list[Term]:
    out = []
    seen: set[str] = set()
    for term in terms:
        if term.expr not in seen:
            seen.add(term.expr)
            out.append(term)
    return out


def _eval_expr(expr: str, x: int) -> bool:
    env = {
        "__builtins__": {"abs": abs, "all": all, "any": any, "int": int, "len": len, "range": range, "set": set, "str": str, "sum": sum, "zip": zip},
        "math": math,
        "_digit_product": _digit_product,
        "_digit_sum": _digit_sum,
        "_distinct_prime_factors": _distinct_prime_factors,
        "_is_cube": _is_cube,
        "_is_lucas": _is_lucas,
        "_is_perfect_power": _is_perfect_power,
        "_is_prime": _is_prime,
        "_is_prime_power": _is_prime_power,
        "_is_semiprime": _is_semiprime,
        "_is_square": _is_square,
        "_is_triangular": _is_triangular,
        "x": x,
    }
    return bool(eval(expr, env, {}))


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
            f"def {fn_name}(xs: list[int]) -> bool:\n"
            f"    {doc}\n"
            f"    return all({addition.expr} for x in xs)\n"
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


def _smallest_unused_index(used: set[int]) -> int:
    index = 0
    while index in used:
        index += 1
    return index


def _normalize_struct_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00b2", "^2").replace("\u00b3", "^3")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.?!;:")


def _self_refuting(text: str) -> bool:
    return bool(re.search(
        r"\b(?:this\s+)?hypothesis\s+is\s+(?:incorrect|invalid|wrong)\b|"
        r"\b(?:candidate|rule)\s+is\s+(?:incorrect|invalid|wrong)\b|"
        r"\b(?:incorrect|invalid|wrong)\b|"
        r"\bexamples?\s+(?:contradict|do not support)\b|"
        r"\bexamples?\s+(?:do\s+not|don't|does\s+not|doesn't)\s+fit\b|"
        r"\b(?:this|hypothesis|rule)\s+(?:does\s+not|doesn't|do\s+not|don't)\s+fit\b|"
        r"\b(?:does\s+not|doesn't|do\s+not|don't)\s+fit\b|"
        r"\b(?:does\s+not|doesn't|do\s+not|don't)\s+fit\s+(?:the\s+)?(?:data|pattern|examples?)\b|"
        r"\bfails?(?:\s+(?:on|for|because)\b|$)|"
        r"\b(?:this|hypothesis|rule)\s+is\s+false\b|"
        r"\b(?:true\s+)?rule\s+is\s+different\b|"
        r"\b(?:so|therefore)\s+(?:this\s+)?(?:hypothesis|rule)\s+is\s+(?:false|different)\b|"
        r"\bfalse positive\b|"
        r"\bprogram\s+has\s+a\s+bug\b|"
        r"\bconfidence\s+is\s+0(?:\.0+)?\b|"
        r"\bteacher\s+is\s+lying\b|"
        r"\b(?:typo|misleading examples?)\b",
        text,
    ))


_BOILERPLATE_REWRITES = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"^the rule is to (?:output|say) '?yes'?,?\s*(?:if|when)\s+", ""),
        (r"^the rule is to (?:check|accept)\s+(?:if|whether)\s+", ""),
        (r"^the rule is that\s+", ""),
        (r"^the rule is\s+", ""),
        (r"^the program checks (?:if|whether)\s+", ""),
        (r"^the program outputs '?yes'? (?:if|when)\s+", ""),
        (r"^the program accepts\s+", ""),
        (r"^if (?:the )?number\s+", "the number "),
        (r"^is the number\s+", "the number is "),
        (r"[.\s]*confidence:\s*(?:0(?:\.\d+)?|1(?:\.0+)?)$", ""),
    )
)


def _strip_boilerplate(text: str) -> str:
    for pattern, replacement in _BOILERPLATE_REWRITES:
        text = pattern.sub(replacement, text)
    text = text.strip(" '\"")
    return text


def _library_label(text: str) -> str:
    label = re.sub(r"\s+", " ", str(text).strip())
    return label if len(label) <= 240 else label[:237] + "..."


def _number_list_pattern() -> str:
    token = rf"(?:{_NUMBER_TOKEN_RE.pattern})"
    separator = r"(?:,?\s+(?:or|and)|,|/)"
    return rf"{token}(?:\s*{separator}\s*{token})*"


_DIGIT_SUM_PREFIX = r"(?:digit sums?|digits sum(?:ming)?|sum of (?:(?:its|the)\s+)?(?:decimal )?digits)"
_DIGIT_PRODUCT_PREFIX = r"(?:digit product|digits product|product of (?:(?:its|the)\s+)?digits|digits multiply)"
_FIBONACCI_DIGIT_CLAUSE = r"\b(?:decimal )?digits (?:are )?(?:all )?fibonacci(?: numbers)?\b"
_DIGIT_COUNT_CLAUSE = (
    rf"(?<!at least )(?<!at most )\b(?:(?P<single>single)[-\s]?digit|"
    rf"(?P<hyphen_count>{_NUMBER_TOKEN_RE.pattern})[-\s]digit|"
    rf"(?:with\s+)?(?:exactly\s+)?(?P<count>{_NUMBER_TOKEN_RE.pattern}) digits?)\b"
)
_DIGIT_POSITION = r"(?:first|leading|last|tens|units?|ones)"
_DIGIT_PROPERTY_CLAUSE = (
    rf"\b(?:(?P<position>{_DIGIT_POSITION}) digit (?:is|must be) (?:an? )?"
    rf"(?P<properties>(?:not (?:an? )?)?(?:even|odd|prime)(?: or (?:not )?(?:even|odd|prime)(?!\s+{_DIGIT_POSITION} digit))*)"
    rf"|(?:(?P<adjective_neg>no|without(?: an?)?|not(?: an?)?) )?(?P<adjectives>(?:even|odd|prime)(?: or (?:even|odd|prime))*) (?P<adjective_position>{_DIGIT_POSITION}) digit)\b"
)
_DIVISOR_COUNT_CLAUSE = (
    rf"\b(?:the )?(?:number|count) of (?:(?:its|the) )?(?:(?:positive|total) )?(?:divisors|factors) (?:is|must be) "
    rf"(?P<test>even|odd|(?:(?P<op>greater than or equal to|less than or equal to|greater than|less than|at least|at most|equal to) )?(?P<value>{_NUMBER_TOKEN_RE.pattern}))\b"
)
_FACTOR_EXPONENT_CLAUSE = r"\bprime factorization (?:contains?|containing|has|with) (?:only|all) even exponents\b"


def _strip_digit_sum_clauses(text: str) -> str:
    # Explicit subject clauses, including "and even numbers", constrain x
    # rather than its digit sum or product.
    return re.sub(
        rf"(?:(?:prime|even|odd|(?:perfect )?square)\s+)?(?:{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX}).*?(?=$|[;.]|\s+(?:and|or)\s+(?:that|which|whose|with|(?:the )?(?:(?:even|odd|prime|composite|(?:perfect )?(?:square|cube))\s+)?(?:numbers?|integers?)|(?:a )?multiple|divisible)\b)",
        " ", text,
    )


def _digit_sum_context_before(text: str, position: int) -> bool:
    before = text[max(0, position - 48):position]
    return any(marker in before for marker in ("digit sum", "digits sum", "sum of its digits", "sum of the digits"))


def _strip_non_subject_property_clauses(text: str) -> str:
    text = re.sub(r"\b(?:prime|even|odd|(?:perfect\s+)?square)\s+digit sum\b", "digit sum", text)
    text = _strip_digit_sum_clauses(text)
    if re.search(r"\b\d+(?:st|nd|rd|th)\b", text) and "prime" in text and not re.search(_ORDINAL_INTEGER_CLAUSE, text):
        text = re.sub(r"\bprime(?:s| numbers?| number)?\b", " ", text)
    for pattern in (_DIGIT_PROPERTY_CLAUSE, _DIVISOR_COUNT_CLAUSE, _FACTOR_EXPONENT_CLAUSE, r"\bno odd prime factors\b"):
        text = re.sub(pattern, " ", text)
    text = re.sub(r"\bprime factors?\b", "factor", text)
    patterns = (
        r"\b(?:twice|two times|2 times) (?:a )?prime\b",
        r"\b(?:perfect )?prime powers?\b",
        r"\bprime factorization\b",
        # The parity of a count never makes x itself even or odd.
        r"\b(?:even|odd) number of\b(?! the form\b)",
        r"\b(?:even|odd) tens digit\b",
        r"\btens digit is (?:even|odd|prime)\b",
        r"\b(?:units?|ones) digit is (?:even|odd|prime)\b",
        r"\bsum of two (?:distinct )?primes\b",
        r"\bsum of two squares\b",
        r"\b(?:perfect\s+)?square plus a prime\b",
        r"\bprime numbers? whose square ends in[^.;,]*",
        r"\bsquares? of (?:a )?prime\b",
        r"\bprime square\b",
        r"\bprime plus (?:a )?square of (?:a )?prime\b",
        r"\bprime and a square of (?:a )?prime\b",
    )
    for pattern in patterns:
        text = re.sub(pattern, " ", text)
    return text


def _numbers_after_keyword(text: str, keyword_pattern: str) -> list[int]:
    nums = []
    for match in re.finditer(rf"{keyword_pattern}\s+(?P<nums>{_number_list_pattern()})", text):
        nums.extend(_extract_numbers(match.group("nums")))
    return _unique(nums)


def _extract_numbers(text: str) -> list[int]:
    values = []
    for match in _NUMBER_TOKEN_RE.finditer(text):
        value = _parse_number(match.group(0))
        if value is not None:
            values.append(value)
    return _unique(values)


def _parse_number(token: str) -> int | None:
    token = token.lower().strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    token = token.replace("-", " ")
    if token.startswith(" "):
        signed = "-" + token.lstrip()
    else:
        signed = token
    if signed.lstrip("-").isdigit():
        return int(signed)
    parts = token.split()
    if not parts:
        return None
    if "hundred" in parts:
        idx = parts.index("hundred")
        prefix = 1 if idx == 0 else _parse_number(" ".join(parts[:idx]))
        suffix = _parse_number(" ".join(parts[idx + 1:])) if idx + 1 < len(parts) else 0
        if prefix is not None and suffix is not None:
            return prefix * 100 + suffix
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
        return _TENS[parts[0]] + _ONES[parts[1]]
    if token in _ONES:
        return _ONES[token]
    if token in _TENS:
        return _TENS[token]
    return None


def _connector(fragment: str) -> str:
    return "and" if " and " in f" {fragment} " and " or " not in f" {fragment} " else "or"


def _unique(values: Iterable[int]) -> list[int]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _set_literal(values: Iterable[int]) -> str:
    return "{" + ", ".join(str(v) for v in sorted(set(values))) + "}"


def _strip_outer_groups(text: str) -> str:
    """Remove parentheses only when they enclose the complete expression."""
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wraps_all = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return text
                if depth == 0 and index != len(text) - 1:
                    wraps_all = False
                    break
        if depth != 0 or not wraps_all:
            break
        text = text[1:-1].strip()
    return text


def _top_level_matches(text: str, pattern: str) -> list[re.Match[str]] | None:
    """Return `pattern` matches outside parentheses, or None for unbalanced grouping."""
    depth = 0
    matches = []
    for match in re.finditer(rf"\(|\)|{pattern}", text):
        if match.group(0) == "(":
            depth += 1
        elif match.group(0) == ")":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            matches.append(match)
    return matches if depth == 0 else None


def _split_or(text: str) -> list[str] | None:
    # Protect OR inside a numeric operand list, without hiding other OR clauses.
    number = rf"(?:{_NUMBER_TOKEN_RE.pattern})"
    operands = (
        rf"(?:multiples? of|divisible by|prime factor of|powers? of|"
        rf"ends? (?:in|with)|ending (?:in|with)|starts? with|starting with|beginning with|"
        rf"contains?|containing|digits?(?: include)?|(?:decimal )?digit equal to|(?:tens|units|ones|first|last|leading) digit|"
        rf"{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX}|congruent to)"
        rf"\s+(?:(?:is|equals?|equal to|of|to|either|both|one of|the digit|at least one)\s+)*{_number_list_pattern()}"
    )
    protected = text
    for pattern in (
        operands,
        rf"multiples? of\s+{number}\s+plus\s+{_number_list_pattern()}",
        rf"(?:{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX})\s+(?:is\s+)?(?:odd|even|prime)\s+or\s+{_number_list_pattern()}",
        r"\{[^{}]+\}",
        _ORDINAL_INTEGER_CLAUSE,
        _DIGIT_PROPERTY_CLAUSE,
        r"\b\d0s(?:\s*(?:,|or|and)\s*(?:the\s+)?\d0s)+",
        rf"[nx]\s*(?:%|mod)\s*{number}\s*(?:==|=|equals?|is)\s*{_number_list_pattern()}",
    ):
        protected = re.sub(pattern, lambda m: re.sub(r"\bor\b", "__list_or__", m.group(0)), protected)
    protected = re.sub(r"\bthan or equal to\b", "than __list_or__ equal to", protected)
    protected = re.sub(rf"({number})\s+or\s+(greater|less)\b", r"\1 __list_or__ \2", protected)
    protected = _strip_outer_groups(protected)
    split_points = _top_level_matches(protected, r"\bor\b")
    # An OR nested inside a mixed Boolean expression cannot be flattened into
    # the parser's conjunctive-term path. Abstain rather than broaden support.
    if split_points is None or (not split_points and re.search(r"\bor\b", protected)):
        return None

    parts = []
    start = 0
    for match in split_points:
        part = protected[start:match.start()].strip(" ,")
        if part:
            parts.append(_strip_outer_groups(part).replace("__list_or__", "or"))
        start = match.end()
    tail = protected[start:].strip(" ,")
    if tail:
        parts.append(_strip_outer_groups(tail).replace("__list_or__", "or"))
    if len(parts) <= 1:
        return parts
    prefix = _shared_subject_prefix(parts[0])
    return [parts[0], *[prefix + part if prefix and not _has_subject(part) else part for part in parts[1:]]]


def _shared_subject_prefix(text: str) -> str:
    for marker in ("the number is ", "number is ", "numbers are ", "numbers that are "):
        if text.startswith(marker):
            return marker
    return ""


def _has_subject(text: str) -> bool:
    return bool(re.match(r"(?:if )?(?:the )?(?:number|numbers)\b|(?:a |an )?(?:multiple|prime|perfect|square|cube|power)", text))


def _is_square(x: int) -> bool:
    root = math.isqrt(x)
    return root * root == x


def _is_cube(x: int) -> bool:
    return any(n ** 3 == x for n in range(1, math.isqrt(x) + 2))


def _is_triangular(x: int) -> bool:
    root = math.isqrt(8 * x + 1)
    return root * root == 8 * x + 1


def _digit_sum(x: int) -> int:
    return sum(int(ch) for ch in str(abs(x)))


def _digit_product(x: int) -> int:
    return math.prod(int(ch) for ch in str(abs(x)))


def _distinct_prime_factors(x: int) -> set[int]:
    return {p for p in range(2, x + 1) if x % p == 0 and _is_prime(p)}


def _is_prime_power(x: int) -> bool:
    return x > 1 and len(_distinct_prime_factors(x)) == 1


def _is_perfect_power(x: int) -> bool:
    return x == 1 or any(a ** b == x for a in range(2, math.isqrt(x) + 1) for b in range(2, 8))


def _is_semiprime(x: int) -> bool:
    return any(a * b == x and _is_prime(a) and _is_prime(b) for a in range(2, x + 1) for b in range(a, x + 1))


def _is_lucas(x: int) -> bool:
    a, b = 2, 1
    while a <= x:
        if a == x:
            return True
        a, b = b, a + b
    return x == 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stats = structure_results(args.results_root, task=args.task, dry_run=args.dry_run)
    for domain in sorted(stats):
        summary = ", ".join(f"{key}={value}" for key, value in sorted(stats[domain].items()))
        print(f"d{domain}: {summary}")


if __name__ == "__main__":
    main()
