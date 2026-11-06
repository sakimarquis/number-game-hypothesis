"""Term grammar for hypothesis structuring: parse rule text into executable Terms on 1..domain."""
from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import lru_cache

from analysis.hypothesis_text import (
    NUMBER_TOKEN_RE,
    number_list_pattern,
    blank,
    extract_numbers,
    parse_number,
    list_connector,
    set_literal,
    strip_outer_groups,
    top_level_matches,
    restore_operand_lists,
)
from tasks.hypotheses import is_cube as _is_cube, is_prime as _is_prime, is_square as _is_square


@dataclass(frozen=True)
class Term:
    support: frozenset[int]
    expr: str
    interval: bool = False
    # (start, end) of the phrase the term was parsed from, in the text given to
    # parse_conjunctive_terms; None when the term covers no single phrase.
    span: tuple[int, int] | None = None


_TermBuilder = Callable[[re.Match[str], str, int], Term | None]
_TermRule = tuple[str, _TermBuilder]


ORDINAL_PRIME_RE = re.compile(
    r"(?:(?:the )?numbers? are )?(?:the )?\d+(?:st|nd|rd|th)"
    r"(?:\s*(?:,|and|or)\s*\d+(?:st|nd|rd|th))*\s+(?:primes|prime numbers?)"
)
ORDINAL_INTEGER_CLAUSE = r"\b\d+(?:st|nd|rd|th)(?:\s*(?:,?\s+or)\s*\d+(?:st|nd|rd|th))*\s+positive integers?\b"


def parse_conjunctive_terms(text: str, domain: int) -> list[Term]:
    terms: list[Term] = []

    terms.extend(_parse_digit_sum_terms(text, domain))
    if not terms and re.search(rf"{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX}", text):
        return []
    text = _strip_digit_sum_clauses(text)
    offset_terms, text = _parse_offset_terms(text, domain)
    terms.extend(offset_terms)
    terms.extend(_parse_multiple_terms(text, domain))
    terms.extend(_parse_terms(text, domain, _SIMPLE_TERM_RULES))
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
    if match := re.search(r"\bnot\s+(?:a\s+)?prime\b|\bnot prime\b|\bnon-prime\b", number_text):
        terms.append(build_term("not _is_prime(x)", domain, span=match.span()))
    elif match := re.search(r"\bprime(?:s| numbers?| number)?\b", number_text):
        terms.append(build_term("_is_prime(x)", domain, span=match.span()))

    if match := re.search(r"\bhighly composite\b", number_text):
        terms.append(build_term("all(_divisor_count(y) < _divisor_count(x) for y in range(1, x))", domain, span=match.span()))
    elif match := re.search(r"\bcomposite(?:s| numbers?| number)?\b", number_text):
        terms.append(build_term("x > 1 and not _is_prime(x)", domain, span=match.span()))
    if match := re.search(r"\bnot\s+(?:an?\s+)?odd(?: numbers?| number)?\b", number_text):
        terms.append(build_term("x % 2 == 0", domain, span=match.span()))
    elif match := re.search(r"\bnot\s+(?:an?\s+)?even(?: numbers?| number)?\b", number_text):
        terms.append(build_term("x % 2 == 1", domain, span=match.span()))
    elif match := re.search(r"\beven(?: numbers?| number)?\b", number_text):
        terms.append(build_term("x % 2 == 0", domain, span=match.span()))
    elif match := re.search(r"\bodd(?: numbers?| number)?\b", number_text):
        terms.append(build_term("x % 2 == 1", domain, span=match.span()))

    shifted_square_context = re.search(r"\b(?:perfect\s+)?squares?\s+(?:plus|\+|multiplied|times|ends?)\b|\bsum of two squares\b", text)
    if match := re.search(r"\bnot\s+(?:a\s+)?(?:perfect\s+)?squares?\b(?!-free)", text):
        terms.append(build_term("not _is_square(x)", domain, span=match.span()))
    elif not shifted_square_context and (match := re.search(r"\b(?:perfect\s+)?squares?\b(?!-free)|\bsquare numbers?\b", text)):
        terms.append(build_term("_is_square(x)", domain, span=match.span()))

    digit_cube_context = re.search(r"\bsum of (?:the |its )?cubes of (?:the |its )?digits\b", text)
    if match := re.search(r"\bnot\s+(?:a\s+)?(?:perfect\s+)?cubes?\b", text):
        terms.append(build_term("not _is_cube(x)", domain, span=match.span()))
    elif not digit_cube_context and (match := re.search(r"\b(?:perfect\s+)?cubes?\b|\bcube numbers?\b", text)):
        terms.append(build_term("_is_cube(x)", domain, span=match.span()))

    # "perfect prime power" / "prime power" form an elif group; "perfect power" is independent.
    if match := re.search("perfect prime power", text):
        terms.append(build_term("_is_prime_power(x) and not _is_prime(x)", domain, span=match.span()))
    elif match := re.search("prime power", text):
        terms.append(build_term("_is_prime_power(x)", domain, span=match.span()))

    keyword_text = blank(_FIBONACCI_DIGIT_CLAUSE, text)
    for pattern, expr in _KEYWORD_TERMS:
        if match := re.search(pattern, keyword_text):
            terms.append(build_term(expr, domain, span=match.span()))
    uncovered = keyword_text
    for pattern, _ in _DIGIT_CLASS_TERMS:
        uncovered = re.sub(pattern, " ", uncovered)
    if re.search(_DIGIT_CLASS_CLAUSE, uncovered):
        return []
    distinct_count = rf"\b(?:exactly\s+)?({NUMBER_TOKEN_RE.pattern})\s+distinct (?:decimal |non-?zero )?digits?\b"
    if (match := re.search(r"\bdistinct (?:decimal |non-?zero )?digits\b|\b(?:all )?digits (?:are )?(?:all )?distinct\b", text)) and not re.search(distinct_count, text):
        terms.append(build_term("len(set(str(abs(x)))) == len(str(abs(x)))", domain, span=match.span()))

    if match := re.fullmatch(r"(?:(?:all|any|the|numbers that are)\s+)?positive integers?", text):
        terms.append(build_term(f"1 <= x <= {domain}", domain, interval=True, span=match.span()))
    if match := re.fullmatch(r"(?:all|any)\s+numbers?|\bintegers?\b|\bany integer\b", text):
        terms.append(build_term(f"1 <= x <= {domain}", domain, interval=True, span=match.span()))

    for match in re.finditer(rf"perfect\s+({NUMBER_TOKEN_RE.pattern})(?:st|nd|rd|th)?\s+powers?", text):
        exponent = parse_number(match.group(1))
        if exponent is not None and exponent > 1:
            expr = "x == 1" if exponent > math.log2(domain) else f"x > 0 and any(n ** {exponent} == x for n in range(1, {domain + 2}))"
            terms.append(build_term(expr, domain, span=match.span()))
    for ordinal, exponent in {"fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}.items():
        if match := re.search(rf"\bperfect\s+{ordinal}\s+powers?\b", text):
            terms.append(build_term(f"x > 0 and any(n ** {exponent} == x for n in range(1, {domain + 2}))", domain, span=match.span()))
    # Must run after the ordinal loop: "perfect fourth powers" triggers both, and original ordering
    # places this expr after the ordinal expr in the conjunction.
    if match := re.search(r"\bfourth powers?\b", text):
        terms.append(build_term("any(n ** 4 == x for n in range(1, math.isqrt(x) + 2))", domain, span=match.span()))
    for match in re.finditer(rf"(exactly|at least|at most)\s+({NUMBER_TOKEN_RE.pattern})\s+(?:distinct\s+)?prime factors?(?:,?\s*\(?\s*(?:counted with|counting|with) multiplicity\)?)?", text):
        count = parse_number(match.group(2))
        if count is not None:
            factor_count = _PRIME_FACTOR_COUNT if "multiplicity" in match.group(0) else "len(_distinct_prime_factors(x))"
            op = {"exactly": "==", "at least": ">=", "at most": "<="}[match.group(1)]
            terms.append(build_term(f"{factor_count} {op} {count}", domain, span=match.span()))
    for match in re.finditer(rf"\b(?:with|has|having|have) ({NUMBER_TOKEN_RE.pattern}) distinct prime factors\b", text):
        terms.append(build_term(f"len(_distinct_prime_factors(x)) == {parse_number(match.group(1))}", domain, span=match.span()))
    # "only prime factors are 2 or 5" bounds the factor set; an "and" list may also require each factor.
    only_factors = re.search(rf"\b(?:{_ONLY_PRIME_FACTORS}) (?P<nums>{number_list_pattern()})", text)
    if only_factors:
        if list_connector(only_factors.group("nums")) == "and":
            return []
        terms.append(build_term(f"x > 1 and _distinct_prime_factors(x) <= {set_literal(extract_numbers(only_factors.group('nums')))}", domain, span=only_factors.span()))
    for match in re.finditer(rf"\b(?:co-?prime|relatively prime) (?:to|with) ({NUMBER_TOKEN_RE.pattern})", text):
        terms.append(build_term(f"math.gcd(x, {parse_number(match.group(1))}) == 1", domain, span=match.span()))
    for match in re.finditer(rf"\bdigital root (?:is |of |equal to |equals? )?(?P<nums>{number_list_pattern()})", text):
        roots = [n for n in extract_numbers(match.group("nums")) if 1 <= n <= 9]
        if roots:
            terms.append(build_term(f"(x - 1) % 9 + 1 in {set_literal(roots)}", domain, span=match.span()))
    if match := re.search(r"\bhappy numbers?\b", text):
        support = {x for x in range(1, domain + 1) if _is_happy(x)}
        terms.append(Term(frozenset(support), f"x in {set_literal(support)}", span=match.span()))
    for match in re.finditer(_AND_NOT_VALUE_CLAUSE, text):
        terms.append(build_term(f"x != {parse_number(match.group('value'))}", domain, span=match.span()))
    for match in re.finditer(_DIVISOR_COUNT_CLAUSE, text):
        if match.group("test") in {"even", "odd"}:
            expr = f"_divisor_count(x) % 2 == {int(match.group('test') == 'odd')}"
        else:
            value = parse_number(match.group("value"))
            if value is None:
                continue
            op = {"greater than": ">", "less than": "<", "at least": ">=", "at most": "<=", "greater than or equal to": ">=", "less than or equal to": "<=", None: "==", "equal to": "=="}[match.group("op")]
            expr = f"_divisor_count(x) {op} {value}"
        terms.append(build_term(expr, domain, span=match.span()))
    if match := re.search(_FACTOR_EXPONENT_CLAUSE, text):
        terms.append(build_term("_is_square(x)", domain, span=match.span()))
    if match := re.search(r"\bno odd prime factors\b", text):
        terms.append(build_term("all(p == 2 for p in _distinct_prime_factors(x))", domain, span=match.span()))
    for match in re.finditer(distinct_count, text):
        terms.append(build_term(f"len(set(str(abs(x)))) == {parse_number(match.group(1))}", domain, span=match.span()))
    ones = r"(?:'1'?s?|1s?|ones|set bits?)(?![\w'])"
    for match in re.finditer(
        rf"binary representation (?:contains?|has) (?:exactly |only )?({NUMBER_TOKEN_RE.pattern})\s+{ones}"
        rf"|\b(?:exactly|only) ({NUMBER_TOKEN_RE.pattern})\s+{ones} in (?:its |their )?binary\b", text,
    ):
        terms.append(build_term(f"x.bit_count() == {parse_number(match.group(1) or match.group(2))}", domain, span=match.span()))
    for match in re.finditer(r"\b(even|odd) number of (?:1s|ones|set bits) in (?:its |their )?binary\b", text):
        terms.append(build_term(f"x.bit_count() % 2 == {int(match.group(1) == 'odd')}", domain, span=match.span()))
    for match in re.finditer(rf"binary representation (?:contains?|has) exactly ({NUMBER_TOKEN_RE.pattern}) (?:bits|digits)\b", text):
        terms.append(build_term(f"x.bit_length() == {parse_number(match.group(1))}", domain, span=match.span()))

    if match := re.search(r"\bfactorials?\b", text):
        support = {math.factorial(n) for n in range(1, 8)}
        support = {x for x in support if 1 <= x <= domain}
        if support:
            terms.append(Term(frozenset(support), f"x in {set_literal(support)}", span=match.span()))

    for match in re.finditer(rf"(?P<proper>proper )?divisors? of\s+(?P<nums>{number_list_pattern()})", text):
        nums = extract_numbers(match.group("nums"))
        joiner = f" {list_connector(match.group('nums'))} "
        exprs = [f"({value} % x == 0" + (f" and x < {value})" if match.group("proper") else ")") for value in nums]
        if exprs:
            terms.append(build_term(joiner.join(exprs), domain, span=match.span()))
    for match in re.finditer(
        rf"\b(?:(?P<op>exactly|at least|at most|more than|fewer than|less than)\s+(?P<count>{NUMBER_TOKEN_RE.pattern})"
        rf"|(?P<minimum>{NUMBER_TOKEN_RE.pattern}) or more)\s+(?:positive\s+)?(?:divisors|factors)\b"
        r"(?!\s+(?:of|besides|other than|excluding|apart from)\b)", text,
    ):
        op = {"exactly": "==", "at least": ">=", "at most": "<=", "more than": ">", "fewer than": "<", "less than": "<", None: ">="}[match.group("op")]
        count = parse_number(match.group("count") or match.group("minimum"))
        terms.append(build_term(f"_divisor_count(x) {op} {count}", domain, span=match.span()))

    for match in re.finditer(rf"(?P<neg>not\s+(?:a\s+)?)?powers? of\s+(?P<nums>{number_list_pattern()})", text):
        power_bases = [base for base in extract_numbers(match.group("nums")) if base > 0]
        if power_bases:
            expr = " or ".join(f"(x > 0 and any({base} ** n == x for n in range(0, 16)))" for base in power_bases)
            terms.append(build_term(f"not ({expr})" if match.group("neg") else expr, domain, span=match.span()))

    exact = _parse_exact_singleton(text, domain)
    if exact is not None:
        terms.append(exact)
    for match in re.finditer(ORDINAL_INTEGER_CLAUSE, text):
        values = [int(n) for n in re.findall(r"\d+", match.group(0))]
        terms.append(build_term(f"x in {set_literal(values)}", domain, interval=len(values) == 1, span=match.span()))

    return terms


# Constant-expression keyword triggers: each (pattern, expr) emits build_term(expr, domain) when pattern
# matches text. Entries are independent (no elif coupling); precedence-coupled keywords (parity,
# square/cube, prime, prime-power) live as explicit elif blocks above.
_FIBONACCI_EXPR = "(math.isqrt(5 * x * x + 4) ** 2 == 5 * x * x + 4) or (math.isqrt(5 * x * x - 4) ** 2 == 5 * x * x - 4)"
# Any other "even/odd/prime digit" phrase (e.g. "an even digit in the hundreds place") is unresolved.
_DIGIT_CLASS_CLAUSE = r"\b(?:even|odd|prime|composite) digits?\b"
_NOT_ENDING = r"(?<!ends in )(?<!ending in )(?<!end in )(?<!ends with )(?<!ending with )"
_DIGIT_CLASS_TERMS: tuple[tuple[str, str], ...] = (
    (r"\b(?:ends?|ending) (?:in|with) a prime digit\b|\ba prime digit in the (?:units?|ones) place\b", "x % 10 in {2, 3, 5, 7}"),
    (r"\ba prime digit in the tens place\b", "x >= 10 and (x // 10) % 10 in {2, 3, 5, 7}"),
    (rf"\b(?:at least one|one or more|{_NOT_ENDING}a|any) prime digits?\b(?! (?:in|at) the \w+ place\b| count\b)", "any(ch in '2357' for ch in str(abs(x)))"),
    (r"\b(?:with |only |all |only of |entirely of )prime digits\b(?! (?:in|at) the \w+ place\b| count\b)|\bprime digits only\b|\bdigits (?:are )?(?:all )?prime\b", "all(ch in '2357' for ch in str(abs(x)))"),
    (r"\b(?:both digits (?:are )?prime|digits are both prime(?: digits)?|two prime digits)\b","10 <= x <= 99 and all(ch in '2357' for ch in str(x))"),
    (r"\b(?:ends?|ending) (?:in|with) an? even digit\b|\ban? even digit in the (?:units?|ones) place\b", "x % 2 == 0"),
    (r"\b(?:ends?|ending) (?:in|with) an? odd digit\b|\ban? odd digit in the (?:units?|ones) place\b", "x % 2 == 1"),
    (r"\ban? even digit in the tens place\b", "x >= 10 and (x // 10) % 2 == 0"),
    (r"\ban? odd digit in the tens place\b", "x >= 10 and (x // 10) % 2 == 1"),
    (rf"\b(?:at least one|one or more|{_NOT_ENDING}an|any) even digits?\b(?! (?:in|at) the \w+ place\b)", "any(int(ch) % 2 == 0 for ch in str(abs(x)))"),
    (rf"\b(?:at least one|one or more|{_NOT_ENDING}an|any) odd digits?\b(?! (?:in|at) the \w+ place\b)", "any(int(ch) % 2 == 1 for ch in str(abs(x)))"),
    (r"\b(?:with |only |all |only of |entirely of )even digits\b(?! (?:in|at) the \w+ place\b)|\beven digits only\b|\bdigits (?:are )?(?:all )?even\b|\ball (?:of )?(?:its |the )?digits (?:are )?even\b|\bno odd digits\b", "all(int(ch) % 2 == 0 for ch in str(abs(x)))"),
    (r"\b(?:with |only |all |only of |entirely of )odd digits\b(?! (?:in|at) the \w+ place\b)|\bodd digits only\b|\bdigits (?:are )?(?:all )?odd\b|\ball (?:of )?(?:its |the )?digits (?:are )?odd\b|\bno even digits\b", "all(int(ch) % 2 == 1 for ch in str(abs(x)))"),
    (r"\bdigits are both (?:even or both odd|odd or both even)\b", "10 <= x <= 99 and x // 10 % 2 == x % 2"),
    (r"\b(?:both digits (?:are )?even|(?:(?:tens and units|units and tens) )?digits are both even(?! or both| primes?\b| prime digits)|two even digits)\b", "10 <= x <= 99 and x // 10 % 2 == 0 and x % 2 == 0"),
    (r"\b(?:both digits (?:are )?odd|(?:(?:tens and units|units and tens) )?digits are both odd(?! or both| primes?\b| prime digits)|two odd digits)\b", "10 <= x <= 99 and x // 10 % 2 == 1 and x % 2 == 1"),
    (r"\bone even(?: digit)? and one odd digit\b|\bone odd(?: digit)? and one even digit\b", "10 <= x <= 99 and {int(str(x)[0]) % 2, int(str(x)[1]) % 2} == {0, 1}"),
)
MULTIPLICITY = r",? \(?(?:counting|counted with|including|with) (?:multiplicity|duplicates)\b"
_PRIME_FACTOR_COUNT = "sum(1 for p in _distinct_prime_factors(x) for k in range(1, x.bit_length()) if x % (p ** k) == 0)"
_KEYWORD_TERMS: tuple[tuple[str, str], ...] = (
    ("fibonacci", _FIBONACCI_EXPR),
    ("lucas", "_is_lucas(x)"),
    ("mersenne", "x > 0 and ((x + 1) & x) == 0"),
    ("semiprime", "_is_semiprime(x)"),
    # The integer-root shapes _non_integer_root lets through.
    (r"\bwith (?:an? )?(?:(?:odd|even|positive) )?(?:integer|whole number) square roots?\b|\bwhose square roots? (?:is|are) (?:an? )?(?:(?:odd|even|positive) )?(?:integers?|whole numbers?)\b", "_is_square(x)"),
    (r"\bwith (?:an? )?(?:(?:odd|even|positive) )?(?:integer|whole number) cube roots?\b|\bwhose cube roots? (?:is|are) (?:an? )?(?:(?:odd|even|positive) )?(?:integers?|whole numbers?)\b", "_is_cube(x)"),
    ("harshad|niven", "x % _digit_sum(x) == 0"),
    ("triangular", "_is_triangular(x)"),
    ("pentagonal", "any(n * (3 * n - 1) // 2 == x for n in range(1, x + 1))"),
    ("hexagonal", "any(n * (2 * n - 1) == x for n in range(1, x + 1))"),
    ("palindrome|palindromic", "str(abs(x)) == str(abs(x))[::-1]"),
    ("perfect power", "_is_perfect_power(x)"),
    (r"\brepunit\b", "set(str(abs(x))) == {'1'}"),
    (r"\bsquare of a prime\b|\bprime square\b", "_is_square(x) and _is_prime(math.isqrt(x))"),
    (r"\bn\^2\b(?!\s*=)|\bn squared\b|\b[nx] = [a-z]\^2\b", "_is_square(x)"),
    (r"\bn\^3\b(?!\s*=)|\bn cubed\b|\b[nx] = [a-z]\^3\b", "_is_cube(x)"),
    (r"\bpowers of integers\b|\bpowers of whole numbers\b", "_is_perfect_power(x)"),
    (r"\bexactly two divisors\b", "_is_prime(x)"),
    (r"(?<!no )\brepeated (?:decimal )?digits\b", "len(set(str(abs(x)))) < len(str(abs(x)))"),
    (r"\brepdigit\b|\ball digits are (?:the )?same\b", "len(set(str(abs(x)))) == 1"),
    (r"\b(?:all (?:the )?digits are distinct|no repeated (?:decimal )?digits)\b", "len(set(str(abs(x)))) == len(str(abs(x)))"),
    (r"\bno (?:zero|0) digits?\b|\bcontains? no (?:zero|0)s?\b|\bwithout (?:a |any )?(?:zero|0)s?(?: digits?)?\b|\bnon-?zero digits\b", "'0' not in str(abs(x))"),
    (r"\b(?:with|has|having) a (?:zero|0) digit\b", "'0' in str(abs(x))"),
    (r"\bbinary representation (?:consists (?:only|entirely) of|is all) 1s\b", "x > 0 and ((x + 1) & x) == 0"),
    (r"\bround numbers?\b", "x % 10 == 0"),
    (r"\bone[-\s]?hot\b|\bone set bit\b|exactly one bit set\b", "x > 0 and (x & (x - 1)) == 0"),
    (r"\babundant numbers?\b", "sum(d for d in range(1, x) if x % d == 0) > x"),
    (r"\bperfect numbers?\b", "x > 1 and sum(d for d in range(1, x) if x % d == 0) == x"),
    (r"(?<!non-)(?<!non)\bdeficient(?: numbers?)?\b", "sum(d for d in range(1, x) if x % d == 0) < x"),
    (r"\beven number of (?:(?:positive|total) )?(?:divisors|factors)\b", "_divisor_count(x) % 2 == 0"),
    (r"\bodd number of (?:(?:positive|total) )?(?:divisors|factors)\b", "_divisor_count(x) % 2 == 1"),
    (r"\beven number of digits\b", "len(str(abs(x))) % 2 == 0"),
    (r"\bodd number of digits\b", "len(str(abs(x))) % 2 == 1"),
    # x // 10 counts the tens and has the parity of the tens digit.
    (r"\beven number of tens\b", "(x // 10) % 2 == 0"),
    (r"\bodd number of tens\b", "(x // 10) % 2 == 1"),
    (rf"\beven number of prime factors{MULTIPLICITY}", f"{_PRIME_FACTOR_COUNT} % 2 == 0"),
    (rf"\bodd number of prime factors{MULTIPLICITY}", f"{_PRIME_FACTOR_COUNT} % 2 == 1"),
    *_DIGIT_CLASS_TERMS,
    (r"\btens digit is greater than (?:the |their |its )?(?:units|ones) digit\b", "10 <= x <= 99 and x // 10 > x % 10"),
    (r"\btens digit is less than (?:the |their |its )?(?:units|ones) digit\b", "10 <= x <= 99 and x // 10 < x % 10"),
    (r"\btens digit is less than or equal to (?:the |their |its )?(?:units|ones) digit\b", "10 <= x <= 99 and x // 10 <= x % 10"),
    (r"\btens digit is greater than or equal to (?:the |their |its )?(?:units|ones) digit\b", "10 <= x <= 99 and x // 10 >= x % 10"),
    (r"(?<!non-)(?<!non)(?<!not )\bsquare-?free\b", "all(x % (p * p) for p in range(2, math.isqrt(x) + 1))"),
    (r"\bdigits (?:are )?in (?:strictly )?(?:ascending|increasing) order\b|\b(?:ascending|increasing) digits\b", "all(a < b for a, b in zip(str(x), str(x)[1:]))"),
    (r"\bdigits (?:are )?in (?:strictly )?(?:descending|decreasing) order\b|\b(?:descending|decreasing) digits\b", "all(a > b for a, b in zip(str(x), str(x)[1:]))"),
    (r"\bdigits (?:are )?in non-?decreasing order\b", "all(a <= b for a, b in zip(str(x), str(x)[1:]))"),
    (r"\bdigits (?:are )?in non-?increasing order\b", "all(a >= b for a, b in zip(str(x), str(x)[1:]))"),
)


def _parse_terms(text: str, domain: int, rules: Iterable[_TermRule]) -> list[Term]:
    terms = []
    for pattern, build in rules:
        for match in re.finditer(pattern, text):
            term = build(match, text, domain)
            if term is not None:
                terms.append(replace(term, span=match.span()))
    return terms


def _parse_multiple_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(rf"prime factor of\s+(?P<nums>{number_list_pattern()})", text):
        values = [value for value in extract_numbers(match.group("nums")) if value > 1]
        if values:
            joiner = f" {list_connector(match.group('nums'))} "
            terms.append(build_term(joiner.join(f"x % {value} == 0" for value in values), domain, span=match.span()))

    # Generic divisibility claims take priority before narrower xor/exclusion forms.
    terms.extend(_parse_terms(text, domain, (
        (
            # In "multiple of 3, and 69 is ...", 69 is the subject of a claim, not an operand.
            rf"(?P<neg>not\s+(?:an?\s+)?)?(?:(?:multiples?|multiple)\s+of|divisible\s+by)\s+(?:(?P<quantifier>both|either)\s+)?(?P<nums>{number_list_pattern()})(?!\s+(?:is|are)\b)",
            _build_multiple_divisibility_term,
        ),
        (
            rf"(?:multiples?|multiple)\s+of\s+({NUMBER_TOKEN_RE.pattern})\s+or\s+({NUMBER_TOKEN_RE.pattern}).*?\bbut\s+not\s+both\b",
            _build_multiple_xor_term,
        ),
        (
            rf"(?:divisible\s+by|multiple\s+of)\s+(?P<yes>{number_list_pattern()})\s*,?\s+but\s+not\s+(?:by\s+)?(?:a\s+)?(?:multiple\s+of\s+)?(?P<no>{number_list_pattern()})",
            _build_multiple_but_not_term,
        ),
    )))

    if match := re.search(r"\b(?:multiple of|divisible by) 2 where the quotient is odd\b", text):
        terms.append(build_term("x % 4 == 2", domain, span=match.span()))
    return terms


def _build_multiple_divisibility_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    # Keep offset, comparison, and digit-sum clauses owned by their specific parsers.
    after = text[match.end():match.end() + 24]
    if re.match(r"\s+(?:plus|with (?:a )?remainder)\s+", after):
        return None
    if not match.group("neg") and re.match(rf"\s*,?\s+but\s+not\s+(?:by\s+)?(?:a\s+)?(?:multiple\s+of\s+)?(?:{NUMBER_TOKEN_RE.pattern})", text[match.end():]):
        return None
    before = text[max(0, match.start() - 24):match.start()]
    if re.search(r"(?:more|greater|less) than (?:a )?$", before):
        return None
    if "digit sum" in before or "digits sum" in before or "sum of" in before:
        return None

    nums_text = match.group("nums")
    # In "multiple of 4 and 20 <= n", the last number belongs to the comparison.
    if re.match(r"\s*(?:[<>]=?|==?|≤|≥)", text[match.end():]):
        nums_text = re.sub(rf"\s*(?:,?\s+(?:or|and)|,|/)\s*(?:{NUMBER_TOKEN_RE.pattern})$", "", nums_text)
        if nums_text == match.group("nums"):
            return None
    operands = extract_numbers(nums_text)
    nums = [n for n in operands if n != 0]
    if not nums:
        # Only 0 is a multiple of 0, so the rule is precise and empty on 1..domain.
        return None if match.group("neg") else build_term("False", domain)
    connector = list_connector(nums_text)
    if match.group("neg"):
        joiner = " or " if match.group("quantifier") == "both" else " and "
        expr = joiner.join(f"x % {n} != 0" for n in nums)
    elif connector == "and":
        expr = " and ".join(f"x % {n} == 0" for n in nums)
    elif re.search(r"\band\b", nums_text):
        # Mixed explicit operators retain AND-before-OR precedence, so a zero
        # operand cannot be dropped without changing the expression.
        if len(nums) != len(operands):
            return None
        expr = NUMBER_TOKEN_RE.sub(lambda m: f"(x % {parse_number(m.group(0))} == 0)", nums_text)
        expr = re.sub(r",\s*(?=and\b|or\b)", " ", expr)
        expr = re.sub(r"[,/]", " or ", expr)
    else:
        expr = " or ".join(f"x % {n} == 0" for n in nums)
    return build_term(expr, domain)


def _build_multiple_xor_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    left = parse_number(match.group(1))
    right = parse_number(match.group(2))
    if left not in {None, 0} and right not in {None, 0}:
        return build_term(f"(x % {left} == 0) != (x % {right} == 0)", domain)
    return None


def _build_multiple_but_not_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    yes = [n for n in extract_numbers(match.group("yes")) if n]
    no = [n for n in extract_numbers(match.group("no")) if n]
    if yes and no:
        joiner = f" {list_connector(match.group('yes'))} "
        included = joiner.join(f"x % {n} == 0" for n in yes)
        excluded = " and ".join(f"x % {n} != 0" for n in no)
        return build_term(f"({included}) and ({excluded})", domain)
    return None


def _build_ending_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    # A root or squared-value suffix condition does not constrain x's last digit.
    if re.search(r"\bsquares? of (?:an? )?(?:numbers?|integers?)\s*$|\b(?:whose|its|their) square\s*$", text[:match.start()]):
        return None
    nums = extract_numbers(match.group("nums"))
    if not nums:
        return None
    if all(0 <= n <= 9 for n in nums):
        op = "not in" if match.group("neg") else "in"
        expr = f"x % 10 {op} {set_literal(nums)}"
    elif match.group("neg"):
        expr = " and ".join(f"not str(abs(x)).endswith('{n}')" for n in nums)
    else:
        expr = " or ".join(f"str(abs(x)).endswith('{n}')" for n in nums)
    return build_term(expr, domain)


def _build_contains_digit_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    if re.search(r"\b(?:factorization|binary)\b", text[:match.start()]):
        return None
    nums = [n for n in extract_numbers(match.group("nums")) if 0 <= n <= 9]
    if not nums:
        return None
    connector = list_connector(match.group("nums"))
    if match.groupdict().get("neg"):
        expr = " and ".join(f"'{n}' not in str(abs(x))" for n in nums)
    else:
        joiner = " and " if connector == "and" else " or "
        expr = joiner.join(f"'{n}' in str(abs(x))" for n in nums)
    return build_term(expr, domain)


def _build_tens_digit_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    nums = [n for n in extract_numbers(match.group("nums")) if 0 <= n <= 9]
    if nums:
        return build_term(f"x >= 10 and (x // 10) % 10 in {set_literal(nums)}", domain)
    return None


def _build_starting_digit_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    if re.match(r"\s+(?:through|to)\s+\d", text[match.end():]):
        return None
    nums = extract_numbers(match.group("nums"))
    if nums and all(1 <= n <= 9 for n in nums):
        return build_term(" or ".join(f"str(x).startswith('{n}')" for n in nums), domain)
    return None


_SIMPLE_TERM_RULES: tuple[_TermRule, ...] = (
    (
        rf"(?P<neg>not\s+)?(?:ends?\s+(?:in|with)|ending\s+(?:in|with)|last digit(?:\s+is|\s+equals?)?|(?:units|ones) digit(?:\s+is|\s+equals?)?)\s+(?:the\s+)?(?:digits?\s+)?(?P<nums>{number_list_pattern()})",
        _build_ending_term,
    ),
    (
        rf"(?P<neg>not\s+)?(?:contains?|containing|decimal representation contains|digits? include|with (?:a\s+)?digits?|has (?:a\s+)?digits?|having (?:a\s+)?digits?)\s+(?:at least one\s+)?(?:the\s+|an?\s+)?(?:digits?\s+)?(?P<nums>{number_list_pattern()})",
        _build_contains_digit_term,
    ),
    (
        rf"at least one (?:decimal )?digit (?:equal to|equals?|is)\s+(?P<nums>{number_list_pattern()})",
        _build_contains_digit_term,
    ),
    (
        rf"(?:contains?|containing)\s+(?P<neg>no)\s+digits?\s+(?P<nums>{number_list_pattern()})",
        _build_contains_digit_term,
    ),
    (
        rf"tens digit(?:\s+(?:is|equals?|equal to|of))?\s+(?P<nums>{number_list_pattern()})",
        _build_tens_digit_term,
    ),
    (
        rf"(?:starts? with|starting with|beginning with|(?:first|leading) digit(?: is| equals?)?)\s+(?P<nums>{number_list_pattern()})",
        _build_starting_digit_term,
    ),
)


def _parse_digit_sum_terms(text: str, domain: int) -> list[Term]:
    prefix = _DIGIT_SUM_PREFIX
    # Numeric digit-sum constraints take priority before parity/prime/square properties.
    terms = _parse_terms(text, domain, (
        (rf"{prefix}\s+(?:is|equals?|equal to|to|of)?\s*(?:either |one of )?(?P<nums>{number_list_pattern()})", _build_digit_sum_exact_term),
        (rf"{prefix}\s+(?:(?:is|to)\s+)?(?:a\s+)?(?:not\s+)?(?:divisible by|multiple of)\s+(?P<nums>{number_list_pattern()})", _build_digit_sum_multiple_term),
        (rf"{prefix}\s+(?:is\s+)?(?:>|greater than|more than)\s+({NUMBER_TOKEN_RE.pattern})", _build_digit_sum_greater_term),
        (rf"{prefix}\s+(?:is\s+)?(?:<|less than|fewer than)\s+({NUMBER_TOKEN_RE.pattern})", _build_digit_sum_less_term),
        (rf"{prefix}\s+(?:is\s+)?(?:<=|less than or equal to|at most|no more than)\s+({NUMBER_TOKEN_RE.pattern})", _build_digit_sum_at_most_term),
        (rf"{prefix}\s+(?:is\s+)?(?:>=|greater than or equal to|at least|no less than)\s+({NUMBER_TOKEN_RE.pattern})", _build_digit_sum_at_least_term),
        (rf"{prefix}\s+(?:is\s+)?({NUMBER_TOKEN_RE.pattern})\s+or\s+less\b", _build_digit_sum_at_most_term),
        (rf"{prefix}\s+(?:is\s+)?({NUMBER_TOKEN_RE.pattern})\s+or\s+greater\b", _build_digit_sum_at_least_term),
    ))

    for match in re.finditer(rf"{prefix}\s+(?:is\s+|are\s+|to a value\s+)?(?P<strict>strictly\s+)?(?:between|from)\s+({NUMBER_TOKEN_RE.pattern})\s+(?:and|to)\s+({NUMBER_TOKEN_RE.pattern})", text):
        lo, hi = parse_number(match.group(2)), parse_number(match.group(3))
        if lo is not None and hi is not None:
            op = "<" if match.group("strict") else "<="
            terms.append(build_term(f"{lo} {op} _digit_sum(x) {op} {hi}", domain, span=match.span()))

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
        match = re.search(rf"{pattern}(?:\s+or\s+(?P<nums>{number_list_pattern()}))?", text)
        if match:
            if match.group("nums"):
                expr = f"({expr}) or _digit_sum(x) in {set_literal(extract_numbers(match.group('nums')))}"
            terms.append(build_term(expr, domain, span=match.span()))

    product_prefix = _DIGIT_PRODUCT_PREFIX
    terms.extend(_parse_terms(text, domain, (
        (rf"{product_prefix}\s+(?:is|equals?|to)?\s*(?P<nums>{number_list_pattern()})", _build_digit_product_exact_term),
    )))
    for match in re.finditer(rf"{product_prefix}\s+(?:is\s+)?(odd|even)(?:\s+or\s+(?P<nums>{number_list_pattern()}))?", text):
        expr = f"_digit_product(x) % 2 == {int(match.group(1) == 'odd')}"
        if match.group("nums"):
            expr += f" or _digit_product(x) in {set_literal(extract_numbers(match.group('nums')))}"
        terms.append(build_term(expr, domain, span=match.span()))
    return terms


def _build_digit_sum_exact_term(match: re.Match[str], text: str, domain: int) -> Term | None:
    if re.match(r"\s+or\s+(?:less|greater)\b", text[match.end():]):
        return None
    nums = [n for n in extract_numbers(match.group("nums")) if n != 0]
    if nums:
        return build_term(f"_digit_sum(x) in {set_literal(nums)}", domain)
    return None


def _build_digit_sum_multiple_term(match: re.Match[str], _text: str, domain: int) -> Term | None:
    negated = re.search(r"\bnot\s+(?:divisible by|a multiple of|multiple of)\b", match.group(0))
    nums = extract_numbers(match.group("nums"))
    if not nums:
        return None
    connector = list_connector(match.group("nums"))
    joiner = " and " if connector == "and" else " or "
    op = "!=" if negated else "=="
    return build_term(joiner.join(f"_digit_sum(x) % {n} {op} 0" for n in nums), domain)


def _build_digit_sum_bound_term(match: re.Match[str], domain: int, op: str) -> Term | None:
    value = parse_number(match.group(1))
    if value is not None:
        return build_term(f"_digit_sum(x) {op} {value}", domain)
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
    nums = extract_numbers(match.group("nums"))
    if nums:
        return build_term(f"_digit_product(x) in {set_literal(nums)}", domain)
    return None


def _parse_digit_count_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(_DIGIT_COUNT_CLAUSE, text):
        count = 1 if match.group("single") else parse_number(match.group("hyphen_count") or match.group("count"))
        if count is not None and count > 0:
            expr = "False" if count > len(str(domain)) else f"{10 ** (count - 1)} <= x <= {10 ** count - 1}"
            terms.append(build_term(expr, domain, interval=True, span=match.span()))
    for match in re.finditer(_DIGIT_COUNT_BOUND_CLAUSE, text):
        op = {"at least": ">=", "at most": "<=", "more than": ">", "fewer than": "<", "less than": "<"}[match.group("op")]
        terms.append(build_term(f"len(str(x)) {op} {parse_number(match.group('limit'))}", domain, interval=True, span=match.span()))
    for match in re.finditer(_DIGIT_PLACE_CLAUSE, text):
        digits = extract_numbers(match.group("digits"))
        if list_connector(match.group("digits")) == "or" and all(0 <= n <= 9 for n in digits):
            expr = f"x >= 10 and (x // 10) % 10 in {set_literal(digits)}" if match.group("place") == "tens" else f"x % 10 in {set_literal(digits)}"
            terms.append(build_term(expr, domain, span=match.span()))

    for match in re.finditer(rf"(?:at least one|a) digit\s+(greater than|less than|equal to)\s+({NUMBER_TOKEN_RE.pattern})", text):
        value = parse_number(match.group(2))
        if value is not None:
            op = {"greater than": ">", "less than": "<", "equal to": "=="}[match.group(1)]
            terms.append(build_term(f"any(int(d) {op} {value} for d in str(abs(x)))", domain, span=match.span()))
    for match in re.finditer(rf"digits (?:are )?(?:all |only )?(?:in|drawn from (?:the )?set|belonging to)\s*\{{(?P<nums>{number_list_pattern()})\}}", text):
        digits = extract_numbers(match.group("nums"))
        if digits and all(0 <= n <= 9 for n in digits):
            terms.append(build_term(f"all(int(d) in {set_literal(digits)} for d in str(abs(x)))", domain, span=match.span()))
    for match in re.finditer(
        rf"\b(?:consist(?:s|ing)?|composed|made up|contain(?:s|ing)?)\s+(?:only|entirely|solely)\s+(?:of\s+)?(?:the\s+)?digits?\s+"
        rf"(?:\{{\s*(?P<braced>{number_list_pattern()})\s*\}}|(?P<nums>{number_list_pattern()}))", text,
    ):
        digits = extract_numbers(match.group("braced") or match.group("nums"))
        if digits and all(0 <= n <= 9 for n in digits):
            terms.append(build_term(f"all(int(d) in {set_literal(digits)} for d in str(abs(x)))", domain, span=match.span()))
    # A difference between digits is defined only for two-digit numbers.
    for match in re.finditer(
        rf"\b(?:(?:absolute )?difference (?:between|of) (?:the |its )?(?:two )?digits (?:is|equals?)|(?:two )?(?:decimal )?digits differ by)"
        rf"\s+(?:exactly\s+)?(?P<nums>{number_list_pattern()})", text,
    ):
        gaps = extract_numbers(match.group("nums"))
        if gaps and all(0 <= n <= 9 for n in gaps):
            terms.append(build_term(f"10 <= x <= 99 and abs(x // 10 - x % 10) in {set_literal(gaps)}", domain, span=match.span()))
    if match := re.search(_FIBONACCI_DIGIT_CLAUSE, text):
        terms.append(build_term("all(int(d) in {0, 1, 2, 3, 5, 8} for d in str(x))", domain, span=match.span()))
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
        terms.append(build_term(expr, domain, span=match.span()))
    for match in re.finditer(rf"(?:tens|units?|ones) digit\s+(?:is )?(greater than|less than|equal to)\s+({NUMBER_TOKEN_RE.pattern})", text):
        value = parse_number(match.group(2))
        if value is not None:
            op = {"greater than": ">", "less than": "<", "equal to": "=="}[match.group(1)]
            expr = f"x >= 10 and (x // 10) % 10 {op} {value}" if match.group(0).startswith("tens") else f"x % 10 {op} {value}"
            terms.append(build_term(expr, domain, span=match.span()))
    for match in re.finditer(rf"(?:containing|contains?|with) at least one (?:of (?:the )?digits\s+|digit from\s*\{{)(?P<nums>{number_list_pattern()})\}}?", text):
        digits = extract_numbers(match.group("nums"))
        if digits and all(0 <= n <= 9 for n in digits):
            terms.append(build_term(f"any(int(d) in {set_literal(digits)} for d in str(abs(x)))", domain, span=match.span()))
    return terms


def _parse_interval_terms(text: str, domain: int) -> list[Term]:
    text = blank(_DIVISOR_COUNT_CLAUSE, text)
    terms = []
    for match in re.finditer(rf"\bfirst\s+({NUMBER_TOKEN_RE.pattern})\s+(?:positive )?integers?\b", text):
        count = parse_number(match.group(1))
        if count is not None:
            terms.append(build_term(f"1 <= x <= {count}", domain, interval=True, span=match.span()))
    range_patterns = (
        r"(?<![\w.])(-?\d+)\s*\.\.\s*(-?\d+)(?![\w.])",
        rf"(?:numbers?\s+)?from\s+({NUMBER_TOKEN_RE.pattern})\s+(?:to|through)\s+({NUMBER_TOKEN_RE.pattern})",
        rf"(?<!from )({NUMBER_TOKEN_RE.pattern})\s+through\s+({NUMBER_TOKEN_RE.pattern})",
        rf"between\s+({NUMBER_TOKEN_RE.pattern})\s+and\s+({NUMBER_TOKEN_RE.pattern})",
        rf"(?:in\s+the\s+)?interval\s+({NUMBER_TOKEN_RE.pattern})\s+to\s+({NUMBER_TOKEN_RE.pattern})",
        rf"interval\s*\[?\s*({NUMBER_TOKEN_RE.pattern})\s*,\s*({NUMBER_TOKEN_RE.pattern})\s*\]?",
        rf"range\s*\[?\s*({NUMBER_TOKEN_RE.pattern})\s*,\s*({NUMBER_TOKEN_RE.pattern})\s*\]?",
        rf"(?:range|within)\s+({NUMBER_TOKEN_RE.pattern})\s*(?:-|to)\s*({NUMBER_TOKEN_RE.pattern})",
    )
    for pattern in range_patterns:
        for match in re.finditer(pattern, text):
            before = text[max(0, match.start() - 28):match.start()]
            if re.search(r"\bnot\s+(?:a\s+)?(?:number\s+)?(?:from|between|in\s+)?\s*$", before):
                continue
            lo = parse_number(match.group(1))
            hi = parse_number(match.group(2))
            if lo is not None and hi is not None:
                if lo > hi:
                    lo, hi = hi, lo
                op = "<" if re.search(r"\bstrictly\s*$", before) else "<="
                terms.append(build_term(f"{lo} {op} x {op} {hi}", domain, interval=True, span=match.span()))

    for match in re.finditer(rf"\[\s*({NUMBER_TOKEN_RE.pattern})\s*,\s*({NUMBER_TOKEN_RE.pattern})\s*\]", text):
        lo, hi = parse_number(match.group(1)), parse_number(match.group(2))
        if lo is not None and hi is not None:
            terms.append(build_term(f"{lo} <= x <= {hi}", domain, interval=True, span=match.span()))

    one_sided = (
        (rf"(?:greater|more)\s+than\s+or\s+equal\s+to\s+({NUMBER_TOKEN_RE.pattern})", ">="),
        (rf"(?:less|fewer)\s+than\s+or\s+equal\s+to\s+({NUMBER_TOKEN_RE.pattern})", "<="),
        (rf"(?:greater|more)\s+than\s+({NUMBER_TOKEN_RE.pattern})", ">"),
        (rf"(?:less|fewer)\s+than\s+({NUMBER_TOKEN_RE.pattern})", "<"),
        (rf"(?:at least|no less than)\s+({NUMBER_TOKEN_RE.pattern})", ">="),
        (rf"(?:at most|no more than)\s+({NUMBER_TOKEN_RE.pattern})", "<="),
        (rf"\b(?:above|over)\s+({NUMBER_TOKEN_RE.pattern})", ">"),
        (rf"\b(?:below|under)\s+({NUMBER_TOKEN_RE.pattern})", "<"),
        (rf"\bup to\s+({NUMBER_TOKEN_RE.pattern})", "<="),
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
            value = parse_number(match.group(1))
            if value is not None:
                terms.append(build_term(f"x {op} {value}", domain, interval=True, span=match.span()))

    for match in re.finditer(rf"({NUMBER_TOKEN_RE.pattern})\s*<=\s*[nx]\s*<=\s*({NUMBER_TOKEN_RE.pattern})", text):
        lo = parse_number(match.group(1))
        hi = parse_number(match.group(2))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            terms.append(build_term(f"{lo} <= x <= {hi}", domain, interval=True, span=match.span()))

    for match in re.finditer(rf"[nx]\s*(<=|>=|<|>)\s*({NUMBER_TOKEN_RE.pattern})", text):
        value = parse_number(match.group(2))
        if value is not None:
            terms.append(build_term(f"x {match.group(1)} {value}", domain, interval=True, span=match.span()))

    # "divisible by 3 and equal to 90 or 96" names the members.
    for match in re.finditer(rf"\band (?:is )?(?:equal to|equals?)\s+(?P<nums>{number_list_pattern()})", text):
        values = extract_numbers(match.group("nums"))
        terms.append(build_term(f"x in {set_literal(values)}", domain, interval=len(values) == 1, span=match.span()))

    for match in re.finditer(rf"({NUMBER_TOKEN_RE.pattern})\s+or\s+greater\b", text):
        if _digit_sum_context_before(text, match.start()):
            continue
        value = parse_number(match.group(1))
        if value is not None:
            terms.append(build_term(f"x >= {value}", domain, interval=True, span=match.span()))
    for match in re.finditer(rf"({NUMBER_TOKEN_RE.pattern})\s+or\s+less\b", text):
        if _digit_sum_context_before(text, match.start()):
            continue
        value = parse_number(match.group(1))
        if value is not None:
            terms.append(build_term(f"x <= {value}", domain, interval=True, span=match.span()))

    if match := re.search(r"\bnumbers? in the \d0s\b", text):
        decades = sorted({int(value) * 10 for value in re.findall(r"\b(\d)0s\b", text)})
        if decades:
            if decades == list(range(decades[0], decades[-1] + 1, 10)):
                terms.append(build_term(f"{decades[0]} <= x <= {decades[-1] + 9}", domain, interval=True, span=match.span()))
            else:
                decade_terms = [build_term(f"{value} <= x <= {value + 9}", domain, interval=True) for value in decades]
                terms.append(replace(combine_terms(decade_terms, "or"), span=match.span()))
    return terms


def _parse_modulo_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(rf"remainder (?:of )?({NUMBER_TOKEN_RE.pattern})\s+(?:when divided by|upon division by|on division by)\s+({NUMBER_TOKEN_RE.pattern})", text):
        residue, modulus = parse_number(match.group(1)), parse_number(match.group(2))
        if modulus and residue is not None:
            terms.append(build_term(f"x % {modulus} == {residue % modulus}", domain, span=match.span()))
    for match in re.finditer(rf"divisible by\s+({NUMBER_TOKEN_RE.pattern})\s+with (?:a )?remainder\s+({NUMBER_TOKEN_RE.pattern})", text):
        modulus, residue = parse_number(match.group(1)), parse_number(match.group(2))
        if modulus and residue is not None:
            terms.append(build_term(f"x % {modulus} == {residue % modulus}", domain, span=match.span()))
    for match in re.finditer(rf"remainder when divided by\s+({NUMBER_TOKEN_RE.pattern})\s+(?:is|equals?)\s+({NUMBER_TOKEN_RE.pattern})", text):
        modulus, residue = parse_number(match.group(1)), parse_number(match.group(2))
        if modulus and residue is not None:
            terms.append(build_term(f"x % {modulus} == {residue % modulus}", domain, span=match.span()))
    # Bare "3 mod 5". "congruent to 1 or 3 mod 5" lists and "3 mod 5 = 3"-style remarks have their own readings.
    for match in re.finditer(rf"(?<![-\w])({NUMBER_TOKEN_RE.pattern})\s+(?:mod|modulo)\s+({NUMBER_TOKEN_RE.pattern})\b(?!\s*(?:==|=|equals?\b|is\b))", text):
        if re.search(r"(?:congruent to|,|\bor)\s*$", text[:match.start()]):
            continue
        residue, modulus = parse_number(match.group(1)), parse_number(match.group(2))
        if modulus and residue is not None:
            terms.append(build_term(f"x % {modulus} == {residue % modulus}", domain, span=match.span()))
    for match in re.finditer(rf"multiples?\s+of\s+({NUMBER_TOKEN_RE.pattern})\s+plus\s+(?P<nums>{number_list_pattern()})", text):
        modulus = parse_number(match.group(1))
        residues = extract_numbers(match.group("nums"))
        if residues and modulus:
            joiner = f" {list_connector(match.group('nums'))} "
            terms.append(build_term(joiner.join(f"x % {modulus} == {residue % modulus}" for residue in residues), domain, span=match.span()))

    for match in re.finditer(rf"[nx]\s*(?:%|mod)\s*({NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*(?P<nums>{number_list_pattern()})", text):
        modulus = parse_number(match.group(1))
        residues = extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(build_term(f"x % {modulus} in {set_literal(r % modulus for r in residues)}", domain, span=match.span()))

    for match in re.finditer(rf"not\s+congruent to\s+(?P<nums>{number_list_pattern()})\s+(?:mod|modulo)\s+({NUMBER_TOKEN_RE.pattern})", text):
        modulus = parse_number(match.group(2))
        residues = extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(build_term(f"x % {modulus} not in {set_literal(r % modulus for r in residues)}", domain, span=match.span()))

    for match in re.finditer(rf"mod\s+({NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*(?P<nums>{number_list_pattern()})", text):
        modulus = parse_number(match.group(1))
        residues = extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(build_term(f"x % {modulus} in {set_literal(r % modulus for r in residues)}", domain, span=match.span()))

    for match in re.finditer(rf"(?:congruent to|mod(?:ulo)?)\s+({NUMBER_TOKEN_RE.pattern})\s+(?:mod|modulo)\s+({NUMBER_TOKEN_RE.pattern})", text):
        if re.search(r"\bnot\s+$", text[max(0, match.start() - 8):match.start()]):
            continue
        residue = parse_number(match.group(1))
        modulus = parse_number(match.group(2))
        if residue is not None and modulus:
            terms.append(build_term(f"x % {modulus} == {residue % modulus}", domain, span=match.span()))

    for match in re.finditer(rf"congruent to\s+(?P<nums>{number_list_pattern()})\s+(?:mod|modulo)\s+({NUMBER_TOKEN_RE.pattern})", text):
        if re.search(r"\bnot\s+$", text[max(0, match.start() - 8):match.start()]):
            continue
        modulus = parse_number(match.group(2))
        residues = extract_numbers(match.group("nums"))
        if modulus and residues:
            terms.append(build_term(f"x % {modulus} in {set_literal(r % modulus for r in residues)}", domain, span=match.span()))
    return terms


def parse_modulo_interval_or_union(text: str, domain: int) -> Term | None:
    mod_expr = rf"[nx]\s*(?:%|mod)\s*({NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*({NUMBER_TOKEN_RE.pattern})"
    interval_expr = rf"({NUMBER_TOKEN_RE.pattern})\s*<=\s*[nx]\s*<=\s*({NUMBER_TOKEN_RE.pattern})"
    match = re.search(rf"{mod_expr}\s+or\s+{interval_expr}", text)
    if match:
        modulus = parse_number(match.group(1))
        residue = parse_number(match.group(2))
        lo = parse_number(match.group(3))
        hi = parse_number(match.group(4))
        if modulus and residue is not None and lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return combine_terms([build_term(f"x % {modulus} == {residue % modulus}", domain), build_term(f"{lo} <= x <= {hi}", domain, interval=True)], "or")

    match = re.search(rf"(?:multiples?|multiple)\s+of\s+({NUMBER_TOKEN_RE.pattern})\s+or\s+(?:numbers?\s+)?from\s+({NUMBER_TOKEN_RE.pattern})\s+to\s+({NUMBER_TOKEN_RE.pattern})", text)
    if match:
        modulus = parse_number(match.group(1))
        lo = parse_number(match.group(2))
        hi = parse_number(match.group(3))
        if modulus and lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return combine_terms([build_term(f"x % {modulus} == 0", domain), build_term(f"{lo} <= x <= {hi}", domain, interval=True)], "or")
    return None


def _parse_form_terms(text: str, domain: int) -> list[Term]:
    terms = []
    for match in re.finditer(rf"prime factorization (?:contains?|containing|includes?)\s+({NUMBER_TOKEN_RE.pattern})\s*\^\s*({NUMBER_TOKEN_RE.pattern})", text):
        base, power = parse_number(match.group(1)), parse_number(match.group(2))
        if base is not None and power is not None and _is_prime(base) and power > 0:
            terms.append(build_term(f"x % {base ** power} == 0", domain, span=match.span()))
    if match := re.search(r"\bnot\s+(?:a\s+)?powers? of 2\b|\bnot\s+2\^n\b", text):
        terms.append(build_term("not (x > 0 and (x & (x - 1)) == 0)", domain, span=match.span()))
        return terms
    # "2^n + 1" and "2^n * 3" are shifted or scaled forms, not powers of 2.
    if match := re.search(r"\b(?:numbers of the form )?2\^n\b(?!\s*[+*-]\s*\d)|\bpowers? of 2\b", text):
        terms.append(build_term("x > 0 and (x & (x - 1)) == 0", domain, span=match.span()))

    for match in re.finditer(r"\b[nx]\s*=\s*(\d+)\s*[a-z]\b", text):
        terms.append(build_term(f"x % {int(match.group(1))} == 0", domain, span=match.span()))

    for match in re.finditer(r"\b[nx]\s*=\s*(\d+)\s*\*?\s*[a-z]\s*\^\s*2\b", text):
        coeff = int(match.group(1))
        terms.append(build_term(f"x % {coeff} == 0 and _is_square(x // {coeff})", domain, span=match.span()))

    for match in re.finditer(r"\b[nx]\s*=\s*(\d+)\s*\^\s*[a-z](?:\s*where\s*[a-z]\s*>=\s*(\d+))?", text):
        base = int(match.group(1))
        start = int(match.group(2) or 0)
        terms.append(build_term(f"any({base} ** n == x for n in range({start}, 16))", domain, span=match.span()))

    for match in re.finditer(r"\b[nx]\s*\^\s*(\d+)\s*=\s*(\d+)\b", text):
        power = int(match.group(1))
        value = int(match.group(2))
        root = round(value ** (1 / power))
        support = {n for n in (root - 1, root, root + 1) if n > 0 and n ** power == value}
        if support:
            terms.append(Term(frozenset(n for n in support if n <= domain), f"x in {set_literal(support)}", interval=len(support) == 1, span=match.span()))

    for match in re.finditer(r"\b(\d+)\s*n\s*([+-])\s*(\d+)\b", text):
        modulus = int(match.group(1))
        offset = int(match.group(3)) * (1 if match.group(2) == "+" else -1)
        terms.append(build_term(f"(x - ({offset})) % {modulus} == 0", domain, span=match.span()))
    for match in re.finditer(r"\b(\d+)\s*[a-z]\s*([+-])\s*(\d+)\b", text):
        modulus = int(match.group(1))
        offset = int(match.group(3)) * (1 if match.group(2) == "+" else -1)
        terms.append(build_term(f"(x - ({offset})) % {modulus} == 0", domain, span=match.span()))
    for match in re.finditer(rf"({NUMBER_TOKEN_RE.pattern})\s+(more|greater|less)\s+than\s+a\s+multiple\s+of\s+({NUMBER_TOKEN_RE.pattern})", text):
        offset = parse_number(match.group(1))
        direction = match.group(2)
        modulus = parse_number(match.group(3))
        if offset is not None and modulus:
            residue = -offset if direction == "less" else offset
            terms.append(build_term(f"x % {modulus} == {residue % modulus}", domain, span=match.span()))
    for match in re.finditer(r"\b(\d+)\s*n\b(?!\s*[+-])", text):
        terms.append(build_term(f"x % {int(match.group(1))} == 0", domain, span=match.span()))
    return terms


_OFFSET_CLAUSE = (
    rf"\b(?P<offset>{NUMBER_TOKEN_RE.pattern})\s+(?P<direction>more|greater|less)\s+than\s+an?\s+(?:positive\s+)?"
    r"(?:(?:perfect\s+)?(?P<base>square|cube)(?:\s+number)?|power\s+of\s+(?P<power>\d+))\b"
)
_SCALE_CLAUSE = rf"\b(?:twice|(?P<scale>{NUMBER_TOKEN_RE.pattern})\s+times)\s+an?\s+(?:perfect\s+)?square(?:\s+number)?\b"


def _parse_offset_terms(text: str, domain: int) -> tuple[list[Term], str]:
    """Shifted ("one more than a perfect square") and scaled ("twice a perfect square") base sets.

    Their spans are blanked from the returned text so the bare square, cube, or power term is not also conjoined.
    """
    terms = []
    for match in re.finditer(_OFFSET_CLAUSE, text):
        offset = parse_number(match.group("offset"))
        power = int(match.group("power") or 0)
        if not offset or (match.group("power") and power < 2):
            return [], text
        y = f"(x + {offset})" if match.group("direction") == "less" else f"(x - {offset})"
        test = f"any({power} ** n == {y} for n in range(0, 16))" if power else f"_is_{match.group('base')}({y})"
        terms.append(build_term(f"{y} > 0 and {test}", domain, span=match.span()))
    for match in re.finditer(_SCALE_CLAUSE, text):
        scale = 2 if match.group("scale") is None else parse_number(match.group("scale"))
        if not scale:
            return [], text
        terms.append(build_term(f"x % {scale} == 0 and _is_square(x // {scale})", domain, span=match.span()))
    return terms, blank(_SCALE_CLAUSE, blank(_OFFSET_CLAUSE, text))


def _parse_shifted_power_terms(text: str, domain: int) -> list[Term]:
    terms = []
    if match := re.search(r"\b(?:twice|two times|2 times) (?:a )?prime\b", text):
        terms.append(build_term("x % 2 == 0 and _is_prime(x // 2)", domain, span=match.span()))
    for match in re.finditer(r"\b(\d+)\s*\^\s*n\s*([+*-])\s*(\d+)\b", text):
        # A bound on n, an equation, or "multiples of 2^n+2" is not the plain form n >= 0.
        if re.match(r"\s*(?:=|(?:for|where)\b[^.;]*?(?:[<>≤≥]|at least|at most|greater|less|positive|odd|even|from|between|\bin\s*(?:\[|\()))", text[match.end():]) \
                or re.search(r"\bmultiples? of\s*$", text[:match.start()]):
            continue
        base, op, value = int(match.group(1)), match.group(2), int(match.group(3))
        terms.append(build_term(f"any({base} ** n {op} {value} == x for n in range(0, 16))", domain, span=match.span()))

    for match in re.finditer(r"\bperfect squares?\s+(?:plus|\+)\s+(-?\d+)\b", text):
        offset = int(match.group(1))
        terms.append(build_term(f"x > {offset} and _is_square(x - ({offset}))", domain, span=match.span()))

    for match in re.finditer(rf"\bsquare of\s+({NUMBER_TOKEN_RE.pattern})\b", text):
        value = parse_number(match.group(1))
        if value is not None:
            squared = value * value
            if 1 <= squared <= domain:
                terms.append(build_term(f"x == {squared}", domain, interval=True, span=match.span()))

    if match := re.search(r"\bperfect square multiplied by a prime\b|\bperfect square times a prime\b", text):
        terms.append(build_term("any(a * a * p == x for a in range(1, math.isqrt(x) + 1) for p in range(2, x + 1) if _is_prime(p))", domain, span=match.span()))
    if match := re.search(r"\bprime or (?:a )?(?:perfect )?square of a prime\b|\bprime numbers? or squares? of primes?\b", text):
        terms.append(build_term("_is_prime(x) or (_is_square(x) and _is_prime(math.isqrt(x)))", domain, span=match.span()))
    if match := re.search(r"\b(?:perfect )?squares? or prime numbers?\b|\bprime numbers? or (?:perfect )?squares?\b", text):
        terms.append(build_term("_is_square(x) or _is_prime(x)", domain, span=match.span()))
    if match := re.search(r"\b(?:is|equals?)\s+(?:the\s+)?(?:sum|product) of (?:its|the) digits\b", text):
        terms.append(build_term("x == _digit_sum(x) or x == _digit_product(x)", domain, span=match.span()))
    if match := re.search(r"\bsum of (?:the |its )?cubes of (?:the |its )?digits\b", text):
        terms.append(build_term("x == sum(int(ch) ** 3 for ch in str(abs(x)))", domain, span=match.span()))
    for match in re.finditer(rf"\b(?:number )?(?:equals?|is)\s+(?:the\s+)?digit sum times\s+({NUMBER_TOKEN_RE.pattern})", text):
        multiplier = parse_number(match.group(1))
        if multiplier is not None:
            terms.append(build_term(f"x == _digit_sum(x) * {multiplier}", domain, span=match.span()))
    if match := re.search(r"\bproduct of the sum of (?:its|the) digits and the sum of (?:its|the) digits\b", text):
        terms.append(build_term("x == _digit_sum(x) ** 2", domain, span=match.span()))
    if match := re.search(r"\b(?:100\s*-\s*[nx]|100 minus the number).*?\bodd\b", text):
        terms.append(build_term("x % 2 == 1", domain, span=match.span()))
    if match := re.search(r"\b(?:100\s*-\s*[nx]|100 minus the number).*?\beven\b", text):
        terms.append(build_term("x % 2 == 0", domain, span=match.span()))

    for match in re.finditer(rf"squares? of (?:an? )?(?:numbers?|integers?) ending in\s+(?P<nums>{number_list_pattern()})", text):
        nums = [n for n in extract_numbers(match.group("nums")) if 0 <= n <= 9]
        if nums:
            terms.append(build_term(f"_is_square(x) and math.isqrt(x) % 10 in {set_literal(nums)}", domain, span=match.span()))

    for match in re.finditer(rf"perfect cube(?:s)? (?:where the base is less than|of a number that is either)\s+(?P<nums>{number_list_pattern()})", text):
        nums = [n for n in extract_numbers(match.group("nums")) if n > 0]
        if nums:
            if "less than" in match.group(0):
                support = {n ** 3 for n in range(1, max(nums))}
            else:
                support = {n ** 3 for n in nums}
            support = {x for x in support if 1 <= x <= domain}
            if support:
                terms.append(Term(frozenset(support), f"x in {set_literal(support)}", span=match.span()))
    for match in re.finditer(rf"\bprime(?: number)?s? whose square ends in\s+(?P<nums>{number_list_pattern()})", text):
        nums = extract_numbers(match.group("nums"))
        nums = [n for n in nums if 0 <= n <= 9]
        if nums:
            terms.append(build_term(f"_is_prime(x) and (x * x) % 10 in {set_literal(nums)}", domain, span=match.span()))
    return terms


def _parse_sum_representation_terms(text: str, domain: int) -> list[Term]:
    terms = []
    if match := re.search(r"\bsum of two distinct (?:primes|prime numbers)\b", text):
        terms.append(build_term("any(a + b == x and a != b and _is_prime(a) and _is_prime(b) for a in range(2, x) for b in range(2, x))", domain, span=match.span()))
    elif match := re.search(r"\bsum of two (?:primes|prime numbers)\b", text):
        terms.append(build_term("any(a + b == x and _is_prime(a) and _is_prime(b) for a in range(2, x) for b in range(2, x))", domain, span=match.span()))
    if match := re.search(r"\bsum of two squares\b", text):
        terms.append(build_term("any(a * a + b * b == x for a in range(0, math.isqrt(x) + 1) for b in range(0, math.isqrt(x) + 1))", domain, span=match.span()))
    if match := re.search(r"\bperfect square plus a prime\b|\bsquare plus a prime\b", text):
        terms.append(build_term("any(_is_square(a) and _is_prime(x - a) for a in range(1, x))", domain, span=match.span()))
    if match := re.search(r"\bprime and a square of a prime\b|\bprime plus (?:a )?square of a prime\b", text):
        terms.append(build_term("any(_is_prime(p) and _is_square(q) and _is_prime(math.isqrt(q)) and p + q == x for p in range(2, x) for q in range(1, x))", domain, span=match.span()))

    ordinal_primes = ORDINAL_PRIME_RE.fullmatch(text)
    nths = [int(n) for n in re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", text)] if ordinal_primes else []
    if nths:
        primes = [x for x in range(2, max(domain * 3, 128)) if _is_prime(x)]
        support = {primes[n - 1] for n in nths if 0 < n <= len(primes) and 1 <= primes[n - 1] <= domain}
        if support:
            terms.append(Term(frozenset(support), f"x in {set_literal(support)}", interval=len(support) == 1, span=ordinal_primes.span()))
    return terms


def _parse_consecutive_sum_terms(text: str, domain: int) -> list[Term]:
    terms = []
    if match := re.search(r"sum of two consecutive odd integers", text):
        terms.append(build_term("x % 4 == 0", domain, span=match.span()))
    elif match := re.search(r"sum of two consecutive even integers", text):
        terms.append(build_term("x % 4 == 2", domain, span=match.span()))
    elif match := re.search(r"sum of two consecutive integers where the first is odd", text):
        terms.append(build_term("x % 4 == 3", domain, span=match.span()))
    elif match := re.search(r"sum of two consecutive integers where the first is even", text):
        terms.append(build_term("x % 4 == 1", domain, span=match.span()))
    elif match := re.search(r"sum of two consecutive integers", text):
        terms.append(build_term("x % 2 == 1", domain, span=match.span()))

    for match in re.finditer(rf"sum of two consecutive integers where the first is\s+({NUMBER_TOKEN_RE.pattern})", text):
        first = parse_number(match.group(1))
        if first is not None:
            terms.append(build_term(f"x == {2 * first + 1}", domain, interval=True, span=match.span()))

    for match in re.finditer(rf"sum of two (?:positive )?integers where the difference between (?:the )?two numbers is\s+({NUMBER_TOKEN_RE.pattern})", text):
        diff = parse_number(match.group(1))
        if diff is not None:
            terms.append(build_term(f"x > {diff} and (x - {diff}) % 2 == 0", domain, span=match.span()))
    return terms


def has_exclusion_marker(text: str) -> bool:
    text = re.sub(_DIGIT_PROPERTY_CLAUSE, " ", text)
    # These atoms already implement negation; an unrecognized exclusion still
    # rejects the whole description instead of silently keeping its positive part.
    atom = (
        rf"(?:prime(?: numbers?)?|odd(?: numbers?)?|even(?: numbers?)?|"
        rf"(?:perfect )?squares?|(?:perfect )?cubes?|"
        rf"(?:multiples? of|divisible by|powers? of|congruent to|"
        rf"ending in|ends? in|contains?|containing)\s+(?:(?:both|either)\s+)?(?:the )?(?:digits? )?{number_list_pattern()})"
    )
    if re.search(r"\bnot\b.*\bor\b.*\bsimultaneously\b", text):
        return True
    rest = re.sub(_AND_NOT_VALUE_CLAUSE, " ", text)
    rest = re.sub(rf"\bnot\s+(?:an?\s+)?{atom}\b", " ", rest)
    rest = re.sub(rf"(?:divisible by|multiples? of)\s+{number_list_pattern()}\s*,?\s+but not (?:by\s+)?(?:a\s+)?(?:multiple\s+of\s+)?{number_list_pattern()}", " ", rest)
    return bool(re.search(r"\b(?:not|except|excluding|exclusive of)\b", rest))


def parse_explicit_set(text: str, domain: int) -> Term | None:
    text = re.sub(r",? which are the only numbers in the range \d+ to \d+ that satisfy the condition$", "", text)
    subject = r"(?:(?:(?:only )?(?:the )?(?:specific )?numbers?(?: (?:is|are|that are))?|the accepted set is|the set)\s+)?"
    values = rf"(?P<nums>{number_list_pattern()})"
    patterns = (
        rf"{subject}(?:(?:is )?in(?: (?:the )?set)?\s*)?\{{\s*{values}\s*\}}",
        rf"{subject}(?:in (?:the )?(?:set|list))\s*\[\s*{values}\s*\]",
        rf"{subject}(?:(?:one of|either|exactly these|exactly|in the set|in the list)\s+)?{values}(?: specifically)?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            nums = extract_numbers(match.group("nums"))
            if len(nums) >= 2:
                support = frozenset(n for n in nums if 1 <= n <= domain)
                return Term(support, f"x in {set_literal(nums)}")
    return None


def parse_bounded_form(text: str, domain: int) -> Term | None:
    token = NUMBER_TOKEN_RE.pattern
    subject = r"(?:(?:numbers?|integers?)\s+(?:(?:of the form|equal to)\s+)?)?"
    bounds = rf"(?:from|between)\s+({token})\s+(?:to|and)\s+({token})(?: inclusive)?"
    match = re.fullmatch(
        r"(?:(?:the )?(?:input )?numbers?(?: (?:is|are|must be))? )?(?:a )?(?:multiples? of|divisible by)\s+(\d+)\.(\d+)", text,
    )
    if match:
        numerator = int(match.group(1) + match.group(2))
        denominator = 10 ** len(match.group(2))
        return build_term(f"(x * {denominator}) % {numerator} == 0", domain) if numerator else None
    match = re.fullmatch(rf"(?:numbers? )?{bounds} in increments of ({token})", text)
    if match:
        lo, hi, step = (parse_number(match.group(i)) for i in (1, 2, 3))
        if lo is not None and hi is not None and step is not None and step > 0:
            return build_term(f"{lo} <= x <= {hi} and (x - {lo}) % {step} == 0", domain)
    match = re.fullmatch(rf"{subject}({token})\s*\*?\s*n,? where n is (?P<nums>{number_list_pattern()})", text)
    if match:
        factor = parse_number(match.group(1))
        if factor is not None:
            return build_term(f"x in {set_literal(factor * n for n in extract_numbers(match.group('nums')))}", domain)
    match = re.fullmatch(rf"{subject}n\^([23])\s+for\s+(?:(?:an?|some)?\s*integer\s+)?n\s+{bounds}", text)
    if match:
        power = int(match.group(1))
        lo, hi = parse_number(match.group(2)), parse_number(match.group(3))
        if lo is not None and hi is not None:
            return build_term(f"any(n ** {power} == x for n in range({lo}, {hi + 1}))", domain)
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
            bound = parse_number(match.group("bound"))
            lo, hi = {
                "under": (1, bound - 1), "below": (1, bound - 1), "less than": (1, bound - 1),
                "less than or equal to": (1, bound), "at most": (1, bound), "up to": (1, bound),
                "greater than": (bound + 1, domain), "more than": (bound + 1, domain),
                "above": (bound + 1, domain), "over": (bound + 1, domain),
                "greater than or equal to": (bound, domain), "at least": (bound, domain),
            }[match.group("op")] if bound is not None else (None, None)
        else:
            lo = parse_number(match.group("lo") or match.group("lo2"))
            hi = parse_number(match.group("hi") or match.group("hi2"))
        if lo is not None and hi is not None:
            return build_term(f"any(n ** {power} == x for n in range({lo}, {hi + 1}))", domain)
    match = re.fullmatch(rf"{subject}({token})\s+times\s+(?:an?|some) integer\s+{bounds}", text)
    if match:
        factor, lo, hi = (parse_number(match.group(i)) for i in (1, 2, 3))
        if factor is not None and lo is not None and hi is not None:
            return build_term(f"any({factor} * n == x for n in range({lo}, {hi + 1}))", domain)
    return None


def parse_multiple_or_singleton_union(text: str, domain: int) -> Term | None:
    for match in re.finditer(rf"[nx]\s*(?:%|mod)\s*({NUMBER_TOKEN_RE.pattern})\s*(?:==|=|equals?|is)\s*0\s+or\s+[nx]\s*(?:==|=|is)\s*({NUMBER_TOKEN_RE.pattern})", text):
        modulus = parse_number(match.group(1))
        single = parse_number(match.group(2))
        if modulus and single is not None:
            return build_term(f"x % {modulus} == 0 or x == {single}", domain)

    pattern = re.compile(
        rf"(?:multiples?|multiple)\s+of\s+(?P<nums>{number_list_pattern()})"
        rf"\s*,?\s*or\s+(?:the\s+)?number\s+is\s+(?P<single>{NUMBER_TOKEN_RE.pattern})"
    )
    match = pattern.search(text)
    if not match:
        return None
    nums = [n for n in extract_numbers(match.group("nums")) if n != 0]
    single = parse_number(match.group("single"))
    if not nums or single is None:
        return None
    expr_parts = [f"x % {n} == 0" for n in nums]
    expr_parts.append(f"x == {single}")
    return build_term(" or ".join(expr_parts), domain)


def parse_only_singleton(text: str, domain: int) -> Term | None:
    patterns = (
        rf"(?:returns?|outputs?|says?)\s+'?yes'?\s+only\s+for\s+({NUMBER_TOKEN_RE.pattern})",
        rf"(?:accepts?|returns?|outputs?)\s+only\s+(?:the\s+)?(?:number\s+)?({NUMBER_TOKEN_RE.pattern})",
        rf"only\s+accepts?\s+(?:the\s+)?(?:number\s+)?({NUMBER_TOKEN_RE.pattern})",
        rf"only\s+the\s+number\s+({NUMBER_TOKEN_RE.pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if re.search(r"\bnot\s+$", text[max(0, match.start() - 8):match.start()]):
                continue
            value = parse_number(match.group(1))
            if value is not None and 1 <= value <= domain:
                return build_term(f"x == {value}", domain, interval=True)
    return None


def parse_first_n_pattern(text: str, domain: int, *, total: int | None = None) -> Term | None:
    match = re.fullmatch(
        rf"(?:(?:the )?numbers?(?: are)? )?(?:the )?first\s+({NUMBER_TOKEN_RE.pattern})\s+"
        rf"(?:multiples of\s+({NUMBER_TOKEN_RE.pattern})|prime numbers?|primes|(?:perfect )?(?P<power>squares?|cubes?)|(?P<parity>odd|even) numbers?)"
        rf"(?:\s+plus (?:the )?(?:specific )?numbers?\s+(?P<extras>{number_list_pattern()}))?", text,
    )
    if match:
        count = parse_number(match.group(1))
        base = parse_number(match.group(2)) if match.group(2) else None
        if count is not None and (base is None or base > 0):
            parity, power = match.group("parity", "power")
            if base:
                members = (base * k for k in itertools.count(1))
            elif parity:
                members = (2 * k - (parity == "odd") for k in itertools.count(1))
            elif power:
                members = (k ** (2 if power.startswith("square") else 3) for k in itertools.count(1))
            else:
                members = filter(_is_prime, itertools.count(1))
            extras = set(extract_numbers(match.group("extras"))) if match.group("extras") else set()
            # Members ascend; none past the domain and every extra can matter.
            limit = max([domain, *extras])
            first = list(itertools.takewhile(lambda x: x <= limit, itertools.islice(members, max(0, count))))
            support = frozenset(x for x in first if x <= domain)
            if total is not None:
                # The declared sequence count precedes domain clipping. Check
                # added values against the full sequence to avoid double-counting.
                if total != max(0, count) + len(extras - set(first)):
                    return None
            support |= frozenset(x for x in extras if 1 <= x <= domain)
            return Term(support, f"x in {set_literal(support)}")
    if total is not None:
        return None
    match = re.search(rf"first\s+({NUMBER_TOKEN_RE.pattern})\s+(odd|even) numbers?", text)
    if match and not re.search(r"\bplus\b", text[match.end():]):
        count = parse_number(match.group(1))
        parity = match.group(2)
        if count is not None:
            start = 1 if parity == "odd" else 2
            support = {start + 2 * i for i in range(count)}
            support = frozenset(x for x in support if 1 <= x <= domain)
            return Term(support, f"x in {set_literal(support)}")

    match = re.search(rf"first\s+({NUMBER_TOKEN_RE.pattern})\s+even numbers?.*?plus(?: the)? number\s+({NUMBER_TOKEN_RE.pattern})", text)
    if match:
        count = parse_number(match.group(1))
        extra = parse_number(match.group(2))
        if count is not None and extra is not None:
            support = {2 * i for i in range(1, count + 1)}
            support.add(extra)
            support = frozenset(x for x in support if 1 <= x <= domain)
            return Term(support, f"x in {set_literal(support)}")

    match = re.search(rf"first\s+({NUMBER_TOKEN_RE.pattern})\s+integers?.*?either even or perfect squares?", text)
    if match:
        count = parse_number(match.group(1))
        if count is not None:
            support = frozenset(x for x in range(1, min(count, domain) + 1) if x % 2 == 0 or _is_square(x))
            return Term(support, f"x in {set_literal(support)}")
    return None


def _parse_exact_singleton(text: str, domain: int) -> Term | None:
    if re.search(rf"\bnot\s+(?:equal to\s+|(?:the\s+)?number\s+)?(?:{NUMBER_TOKEN_RE.pattern})\b", text):
        return None
    if re.fullmatch(rf"{NUMBER_TOKEN_RE.pattern}", text):
        value = parse_number(text)
        if value is not None and 1 <= value <= domain:
            return build_term(f"x == {value}", domain, interval=True, span=(0, len(text)))

    patterns = (
        rf"(?:the )?(?:numbers?|integers?)(?: (?:is|are|that is|that are))?\s+(?:equals?|equal to|exactly|only)\s+({NUMBER_TOKEN_RE.pattern})",
        rf"(?:only|exactly|equals?|(?<!or )equal to|number is|the number is|input is always)\s+(?:exactly\s+)?({NUMBER_TOKEN_RE.pattern})\b(?!\s+(?:or\s+(?:less|greater)|divisors?|digits?|prime factors?))",
        rf"(?:is|must be)\s+(?:the\s+)?number\s+({NUMBER_TOKEN_RE.pattern})\b(?!\s+(?:or\s+(?:less|greater)|divisors?|digits?|prime factors?))",
        rf"\b[nx]\s*(?:==|=)\s*({NUMBER_TOKEN_RE.pattern})\b(?!\s*(?:\*|[a-z]\s*\^|\s+or\s+(?:less|greater)|\s+(?:divisors?|digits?|prime factors?)))",
        rf"\bnumbers? from\s+({NUMBER_TOKEN_RE.pattern})\s+to\s+\1\b",
        rf"(?:(?:the )?numbers? (?:is|are) )?(?:exactly )?(?:the )?({NUMBER_TOKEN_RE.pattern})(?:st|nd|rd|th)\s+positive integer",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            value = parse_number(match.group(1))
            if value is not None and 1 <= value <= domain:
                return build_term(f"x == {value}", domain, interval=True, span=match.span())
    return None


def combine_terms(terms: list[Term], op: str) -> Term:
    if op == "or":
        support = frozenset().union(*(term.support for term in terms))
        expr = " or ".join(f"({term.expr})" for term in terms)
        return Term(support, expr)
    support = set(terms[0].support)
    for term in terms[1:]:
        support &= term.support
    expr = " and ".join(f"({term.expr})" for term in terms)
    return Term(frozenset(support), expr, interval=all(term.interval for term in terms))


def build_term(expr: str, domain: int, *, interval: bool = False, span: tuple[int, int] | None = None) -> Term:
    return Term(_support(expr, domain), expr, interval=interval, span=span)


@lru_cache(maxsize=None)
def _support(expr: str, domain: int) -> frozenset[int]:
    predicate = compile_expr(expr)
    return frozenset(x for x in range(1, domain + 1) if predicate(x))


def dedupe_terms(terms: Iterable[Term]) -> list[Term]:
    out = []
    seen: set[str] = set()
    for term in terms:
        if term.expr not in seen:
            seen.add(term.expr)
            out.append(term)
    return out


def compile_expr(expr: str) -> Callable[[int], object]:
    # Compile once per expression; evaluating the source string per x dominated runtime.
    env = {
        "__builtins__": {"abs": abs, "all": all, "any": any, "int": int, "len": len, "range": range, "set": set, "str": str, "sum": sum, "zip": zip},
        "math": math,
        "_digit_product": _digit_product,
        "_digit_sum": _digit_sum,
        "_divisor_count": _divisor_count,
        "_distinct_prime_factors": _distinct_prime_factors,
        "_is_cube": _is_cube,
        "_is_lucas": _is_lucas,
        "_is_perfect_power": _is_perfect_power,
        "_is_prime": _is_prime,
        "_is_prime_power": _is_prime_power,
        "_is_semiprime": _is_semiprime,
        "_is_square": _is_square,
        "_is_triangular": _is_triangular,
    }
    return eval(f"lambda x: {expr}", env)


_DIGIT_SUM_PREFIX = r"(?:digit sums?|digits (?:that )?sum(?:ming)?|sum of (?:(?:its|the)\s+)?(?:decimal )?digits)"
_DIGIT_PRODUCT_PREFIX = r"(?:digit product|digits product|product of (?:(?:its|the)\s+)?digits|digits multiply)"
_FIBONACCI_DIGIT_CLAUSE = r"\b(?:decimal )?digits (?:are )?(?:all )?fibonacci(?: numbers)?\b"
_DIGIT_COUNT_CLAUSE = (
    rf"(?<!at least )(?<!at most )(?<!more than )(?<!fewer than )(?<!less than )\b(?:(?P<single>single)[-\s]?digit|"
    rf"(?P<hyphen_count>{NUMBER_TOKEN_RE.pattern})[-\s]digit|"
    rf"(?:with\s+)?(?:exactly\s+)?(?P<count>{NUMBER_TOKEN_RE.pattern}) digits?)\b"
)
_DIGIT_POSITION = r"(?:first|leading|last|tens|units?|ones)"
_DIGIT_PROPERTY_CLAUSE = (
    rf"\b(?:(?P<position>{_DIGIT_POSITION}) digit (?:is|must be) (?:an? )?"
    rf"(?P<properties>(?:not (?:an? )?)?(?:even|odd|prime)(?: or (?:not )?(?:even|odd|prime)(?!\s+{_DIGIT_POSITION} digit))*)"
    rf"|(?:(?P<adjective_neg>no|without(?: an?)?|not(?: an?)?) )?(?P<adjectives>(?:even|odd|prime)(?: or (?:even|odd|prime))*) (?P<adjective_position>{_DIGIT_POSITION}) digit)\b"
)
_DIVISOR_COUNT_CLAUSE = (
    rf"\b(?:the )?(?:number|count) of (?:(?:its|the) )?(?:(?:positive|total) )?(?:divisors|factors) (?:is|must be) "
    rf"(?P<test>even|odd|(?:(?P<op>greater than or equal to|less than or equal to|greater than|less than|at least|at most|equal to) )?(?P<value>{NUMBER_TOKEN_RE.pattern}))\b"
)
_FACTOR_EXPONENT_CLAUSE = r"\bprime factorization (?:contains?|containing|has|with) (?:only|all) even exponents\b"
# "not prime and not 1": a trailing excluded value after a property, not a divisor or digit.
_AND_NOT_VALUE_CLAUSE = rf"(?:(?<=prime)|(?<=composite)|(?<=even)|(?<=odd)) and not (?:equal to )?(?P<value>{NUMBER_TOKEN_RE.pattern})$"
_DIGIT_COUNT_BOUND_CLAUSE = rf"\b(?:with )?(?P<op>at least|at most|more than|fewer than|less than) (?P<limit>{NUMBER_TOKEN_RE.pattern}) digits\b"
_DIGIT_PLACE_CLAUSE = rf"\b(?:with )?(?:an? |the digit )?(?P<digits>{number_list_pattern()}) in the (?P<place>ones|units?|tens) place\b"
_ONLY_PRIME_FACTORS = r"only prime factors? (?:is|are)|only prime factors?|prime factors are only"


def _strip_digit_sum_clauses(text: str) -> str:
    # Explicit subject clauses, including "and even numbers", constrain x
    # rather than its digit sum or product.
    return blank(
        rf"(?:(?:prime|even|odd|(?:perfect )?square)\s+)?(?:{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX}).*?(?=$|[;.]|\s+(?:and|or)\s+(?:that|which|whose|with|(?:the )?(?:(?:even|odd|prime|composite|(?:perfect )?(?:square|cube))\s+)?(?:numbers?|integers?)|(?:a )?multiple|divisible)\b)",
        text,
    )


def _digit_sum_context_before(text: str, position: int) -> bool:
    before = text[max(0, position - 48):position]
    return any(marker in before for marker in ("digit sum", "digits sum", "sum of its digits", "sum of the digits"))


def _strip_non_subject_property_clauses(text: str) -> str:
    text = blank(r"\b(?:prime|even|odd|(?:perfect\s+)?square)\s+digit sum\b", text, "digit sum")
    text = _strip_digit_sum_clauses(text)
    if re.search(r"\b\d+(?:st|nd|rd|th)\b", text) and "prime" in text and not re.search(ORDINAL_INTEGER_CLAUSE, text):
        text = blank(r"\bprime(?:s| numbers?| number)?\b", text)
    for pattern in (_DIGIT_PROPERTY_CLAUSE, _DIVISOR_COUNT_CLAUSE, _FACTOR_EXPONENT_CLAUSE, r"\bno odd prime factors\b"):
        text = blank(pattern, text)
    text = blank(r"\bprime factors?\b", text, "factor")
    patterns = (
        r"\b(?:twice|two times|2 times) (?:a )?prime\b",
        r"\b(?:perfect )?prime powers?\b",
        r"\bprime factorization\b",
        # The parity of a count never makes x itself even or odd.
        r"\b(?:even|odd) number of\b(?! the form\b)",
        *(pattern for pattern, _ in _DIGIT_CLASS_TERMS),
        _DIGIT_CLASS_CLAUSE,
        r"\b(?:all (?:of )?(?:its |the )?)?digits (?:are )?(?:all )?(?:even|odd)\b",
        r"\b(?:even|odd) tens digit\b",
        r"\btens digit is (?:even|odd|prime)\b",
        r"\b(?:units?|ones) digit is (?:even|odd|prime)\b",
        r"\brelatively prime\b",
        r"\bsum of two (?:distinct )?(?:primes|prime numbers)\b",
        r"\bsum of two squares\b",
        r"\b(?:perfect\s+)?square plus a prime\b",
        r"\bprime numbers? whose square ends in[^.;,]*",
        r"\bsquares? of (?:a )?prime\b",
        r"\bprime square\b",
        r"\bprime plus (?:a )?square of (?:a )?prime\b",
        r"\bprime and a square of (?:a )?prime\b",
    )
    for pattern in patterns:
        text = blank(pattern, text)
    return text


def protect_operand_lists(text: str) -> str:
    """Mask AND/OR inside numeric operand lists ("divisible by 3 or 7") so only clause connectives remain."""
    number = rf"(?:{NUMBER_TOKEN_RE.pattern})"
    operands = (
        rf"(?:multiples? of|divisible by|prime factor of|powers? of|"
        rf"ends? (?:in|with)|ending (?:in|with)|starts? with|starting with|beginning with|"
        rf"contains?|containing|digits?(?: include)?|(?:decimal )?digit equal to|(?:tens|units|ones|first|last|leading) digit|"
        rf"digits differ by|(?:absolute )?difference (?:between|of) (?:the |its )?(?:two )?digits|"
        rf"{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX}|congruent to|digital root|{_ONLY_PRIME_FACTORS})"
        rf"\s+(?:(?:is|equals?|equal to|of|to|either|both|one of|the digit|at least one|an?)\s+)*{number_list_pattern()}(?!\s+(?:is|are)\b)"
    )
    protected = text
    for pattern in (
        operands,
        rf"multiples? of\s+{number}\s+plus\s+{number_list_pattern()}",
        rf"(?:{_DIGIT_SUM_PREFIX}|{_DIGIT_PRODUCT_PREFIX})\s+(?:is\s+)?(?:odd|even|prime)\s+or\s+{number_list_pattern()}",
        r"\{[^{}]+\}",
        r"\bdigits are both (?:even or both odd|odd or both even)\b",
        ORDINAL_INTEGER_CLAUSE,
        _DIGIT_PROPERTY_CLAUSE,
        _DIGIT_PLACE_CLAUSE,
        r"\b\d0s(?:\s*(?:,|or|and)\s*(?:the\s+)?\d0s)+",
        rf"[nx]\s*(?:%|mod)\s*{number}\s*(?:==|=|equals?|is)\s*{number_list_pattern()}",
        rf"\band (?:is )?(?:equal to|equals?)\s+{number_list_pattern()}",
        rf"\bbetween\s+{number}\s+and\s+{number}",
    ):
        # Commas are masked too, so clause splitting cannot cut "3, 7, or 11".
        protected = re.sub(pattern, lambda m: re.sub(r"\b(and|or)\b|(,)", lambda k: f"__list_{k.group(1) or 'comma'}__", m.group(0)), protected)
    protected = re.sub(r"\bthan or equal to\b", "than __list_or__ equal to", protected)
    protected = re.sub(rf"({number})\s+or\s+(greater|less)\b", r"\1 __list_or__ \2", protected)
    return re.sub(rf"({number})\s+or\s+more(?=\s+(?:positive\s+)?(?:divisors|factors)\b)|\b(one) or more\b", lambda m: f"{m.group(1) or m.group(2)} __list_or__ more", protected)


def split_or(text: str) -> list[str] | None:
    protected = strip_outer_groups(protect_operand_lists(text))
    split_points = top_level_matches(protected, r"\bor\b")
    # An OR nested inside a mixed Boolean expression cannot be flattened into
    # the parser's conjunctive-term path. Abstain rather than broaden support.
    if split_points is None or (not split_points and re.search(r"\bor\b", protected)):
        return None

    parts = []
    start = 0
    for match in split_points:
        part = protected[start:match.start()].strip(" ,")
        if part:
            parts.append(restore_operand_lists(strip_outer_groups(part)))
        start = match.end()
    tail = protected[start:].strip(" ,")
    if tail:
        parts.append(restore_operand_lists(strip_outer_groups(tail)))
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


def _is_triangular(x: int) -> bool:
    root = math.isqrt(8 * x + 1)
    return root * root == 8 * x + 1


def _digit_sum(x: int) -> int:
    return sum(int(ch) for ch in str(abs(x)))


def _digit_product(x: int) -> int:
    return math.prod(int(ch) for ch in str(abs(x)))


def _divisor_count(x: int) -> int:
    return sum(1 for d in range(1, x + 1) if x % d == 0)


def _distinct_prime_factors(x: int) -> set[int]:
    return {p for p in range(2, x + 1) if x % p == 0 and _is_prime(p)}


def _is_prime_power(x: int) -> bool:
    return x > 1 and len(_distinct_prime_factors(x)) == 1


def _is_perfect_power(x: int) -> bool:
    return x == 1 or any(a ** b == x for a in range(2, math.isqrt(x) + 1) for b in range(2, 8))


def _is_semiprime(x: int) -> bool:
    return any(a * b == x and _is_prime(a) and _is_prime(b) for a in range(2, x + 1) for b in range(a, x + 1))


def _is_happy(x: int) -> bool:
    seen = set()
    while x != 1 and x not in seen:
        seen.add(x)
        x = sum(int(ch) ** 2 for ch in str(x))
    return x == 1


def _is_lucas(x: int) -> bool:
    a, b = 2, 1
    while a <= x:
        if a == x:
            return True
        a, b = b, a + b
    return x == 1
