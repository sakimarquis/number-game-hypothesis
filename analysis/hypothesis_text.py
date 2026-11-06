"""Generic text utilities for hypothesis structuring: number words, normalization, and set literals."""
from __future__ import annotations

import re
from collections.abc import Iterable


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
# (?!\d) stops backtracking from reading "17.5" as the integer "1".
NUMBER_TOKEN_RE = re.compile(rf"(?<![\w.])-?\d+(?!\d|\.\d)|\b(?:{_NUMBER_WORD})(?:[-\s](?:{_NUMBER_WORD}))?\b")


def number_list_pattern() -> str:
    token = rf"(?:{NUMBER_TOKEN_RE.pattern})"
    separator = r"(?:,?\s+(?:or|and)|,|/)"
    return rf"{token}(?:\s*{separator}\s*{token})*"


REMARK_MARKER = r"^\s*(?:or\b,?\s*)?(?:(?:i\.e\.|e\.g\.|that is|equivalently|for example|such as|which includes|specifically)[,:]?\s*)?"
# Notes that can never narrow the rule they follow.
IGNORABLE_REMARK = (
    r"(?:inclusive(?: of (?:both )?(?:ends|endpoints|bounds|\d+(?: and \d+)?))?|all (?:integers|numbers) in (?:that|the|this) range"
    r"|confidence:? [\d.]+%?|(?:alternative|redundant) phrasing|intersection|union|a union of (?:these|the) (?:\w+ )?(?:conditions|rules|intervals)"
    # References to other hypotheses in the same answer.
    r"|(?:same as|redundant|rephrased|duplicate|repeated)(?: (?:with|of|from|in|to))?"
    r"(?: (?:the )?(?:above|before|first|previous|prompt|logic|hypothesis(?: \d+)?|one))?(?:,? same confidence)?)\.?"
)
EXAMPLE_CLAIM = r"(?:^|[,;.]\s+)(?:since |because |as |but )?-?\d+(?:(?:,|,? and|,? or)\s*-?\d+)*\s*(?::|=|->|\+|%|[*^](?!\s*[a-z])|\b(?:is|are|has|have|gives?|means?|ends?|starts?|contains?|yes|no)\b)"
# Words a parsed description may contain beyond what its terms consume.
GLUE_WORDS = {
    "a", "an", "the", "number", "numbers", "integer", "integers", "positive", "all", "any",
    "is", "are", "must", "be", "that", "which", "those", "these", "in", "within", "from", "to", "between",
    "and", "or", "at", "least", "most", "no", "less", "fewer", "more", "greater", "than",
    "equal", "equals", "above", "below", "under", "over", "up", "through", "strictly", "inclusive", "closed", "interval", "range",
    "exactly", "only", "first", "input", "always", "x", "n",
    "but", "also", "both", "it", "they", "with", "has", "have", "having", "being", "lies", "program", "says", "outputs", "returns", "yes", "for", "if", "when",
    "of", "form", "whose", "where", "such", "either", "satisfy", "satisfying", "condition",
    "contain", "contains", "containing", "leave", "leaves", "leaving", "give", "gives", "giving", "decimal", "natural", "simultaneously",
    "can", "written", "expressed", "represented", "as", "representation", "representations", "do", "does", "k", "m",
    # Plural suffix left after unbounded keyword patterns ("semiprime" in "semiprimes").
    "s",
}
VAGUE_MARKERS = (
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


def normalize_struct_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00b2", "^2").replace("\u00b3", "^3")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.?!;:")


def self_refuting(text: str) -> bool:
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
        r"\b(?:typos?|misleading examples?)\b",
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
        (r"^linear transformation:\s*", ""),
        # The complement clause of a yes/no rule restates it.
        (r"\s*,?\s*(?:and )?'?no'? (?:otherwise|for all others?|for (?:all )?other numbers)$|\s*,\s*otherwise '?no'?$", ""),
        (r"\s*,?\s*to (?:say|output) '?yes'?$", ""),
    )
)


def strip_boilerplate(text: str) -> str:
    for pattern, replacement in _BOILERPLATE_REWRITES:
        text = pattern.sub(replacement, text)
    text = text.strip(" '\"")
    return text


def library_label(text: str) -> str:
    label = re.sub(r"\s+", " ", str(text).strip())
    return label if len(label) <= 240 else label[:237] + "..."


def blank(pattern: str, text: str, word: str = "") -> str:
    # Length-preserving re.sub: term spans found in the result index `text`.
    return re.sub(pattern, lambda m: word.rjust(len(m.group(0))), text)


def extract_numbers(text: str) -> list[int]:
    values = []
    for match in NUMBER_TOKEN_RE.finditer(text):
        value = parse_number(match.group(0))
        if value is not None:
            values.append(value)
    return _unique(values)


def parse_number(token: str) -> int | None:
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
        prefix = 1 if idx == 0 else parse_number(" ".join(parts[:idx]))
        suffix = parse_number(" ".join(parts[idx + 1:])) if idx + 1 < len(parts) else 0
        if prefix is not None and suffix is not None:
            return prefix * 100 + suffix
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
        return _TENS[parts[0]] + _ONES[parts[1]]
    if token in _ONES:
        return _ONES[token]
    if token in _TENS:
        return _TENS[token]
    return None


def list_connector(fragment: str) -> str:
    return "and" if " and " in f" {fragment} " and " or " not in f" {fragment} " else "or"


def _unique(values: Iterable[int]) -> list[int]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def set_literal(values: Iterable[int]) -> str:
    return "{" + ", ".join(str(v) for v in sorted(set(values))) + "}"


def strip_outer_groups(text: str) -> str:
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


def top_level_matches(text: str, pattern: str) -> list[re.Match[str]] | None:
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


def restore_operand_lists(text: str) -> str:
    return re.sub(r"__list_(and|or|comma)__", lambda m: "," if m.group(1) == "comma" else m.group(1), text)
