"""Bounded regex engine: correctness vs Python re + guaranteed bounded work."""

from __future__ import annotations

import re
import time

import pytest

from parquet_range.quality_regex import (
    MAX_STATES,
    PatternSyntaxError,
    compile_pattern,
)

PATTERNS = [
    r"^[A-Z][a-z]*$",
    r"a+b?c*",
    r"(ab|cd){1,2}e",
    r"[0-9]{2,4}",
    r"[^a-z]+",
    r"x(y(z|w))*v",
    r"\d+\.\d+",
    r"^\w+@(a|b)\.com$",
    r".{0,3}q",
    r"a{3}",
    r"(foo)+bar?",
    r"[a-cx-z]{2}",
    r"z|y|x",
    r"^$",
    r".",
    r"a|",
    r"^\d{3}-\d{2}-\d{4}$",
    r"[A-Za-z0-9_]+\.(csv|json|parquet)",
    r"^[+-]?(\d+(\.\d*)?|\.\d+)$",
    r"\s*hello\s+world\s*",
    r"a(bc)*d",
    r"(?:foo|bar|baz)+$",
]

INPUTS = [
    "", "a", "aa", "ab", "abc", "Abc", "XYZ", "123", "12.34", "xYzv",
    "foo@a.com", "foo@b.com", "foo@c.com", "foofoobar", "aaa", "aaq", "zq",
    "abab", "cdab", "a1_", "xyz", "cax", "hello world", "9" * 40,
    "123-45-6789", "12-45-6789", "report.csv", "report.parquet", "-1.5",
    "+.5", "1.", "  hello   world ", "ad", "abcbd", "fooBARbaz",
]


def test_nfa_matches_python_re_search():
    for pattern in PATTERNS:
        compiled = compile_pattern(pattern)
        reference = re.compile(pattern, re.ASCII)
        for text in INPUTS:
            got = compiled.search(text)
            want = reference.search(text) is not None
            assert got == want, f"{pattern!r} on {text!r}: {got} != {want}"


def test_anchors_are_absolute():
    # '$' is strict end-of-string (like \Z): no trailing-newline special case.
    assert not compile_pattern("abc$").search("abc\n")
    assert compile_pattern("abc$").search("abc")
    assert compile_pattern("^a").search("ba") is False
    assert compile_pattern("^$").search("")


def test_search_is_unanchored():
    assert compile_pattern("b+").search("aabbbcc")
    assert not compile_pattern("b+").search("aaacc")


def test_class_negation_and_ranges():
    p = compile_pattern("[^0-9]+")
    assert p.search("abc")
    assert not p.search("123")
    assert compile_pattern("[a-f0-9]+").search("deadbeef")
    assert not compile_pattern("[a-f0-9]+$").search("deadbeefg")


def test_predefined_classes_ascii_only():
    assert compile_pattern(r"\w+").search("hello_123")
    assert compile_pattern(r"\d").search("abc3")
    assert not compile_pattern(r"^\s*$").search("x")


def test_quantifier_braces():
    assert compile_pattern("a{2,3}$").search("aaa")
    # Search semantics: the last 3 a's of "aaaa" match a{2,3}$ (as in re).
    assert compile_pattern("a{2,3}$").search("aaaa")
    assert not compile_pattern("^a{2,3}$").search("aaaa")
    assert compile_pattern("a{2,}").search("a" * 50)
    assert compile_pattern("(ab){2}").search("xababy")


def test_evil_patterns_are_bounded():
    """Patterns that hang backtracking engines finish in milliseconds."""
    # (pattern, subject, expected) -- expected cross-checked against Python re
    # on short inputs (re itself hangs on the long ones, which is the point).
    evil = [
        ("(a+)+$", "a" * 5000 + "b", False),
        ("(a|a)*$", "a" * 5000 + "b", True),   # empty match at end position
        ("(a|aa)+$", "a" * 3000 + "b", False),
        ("(.*)*b", "a" * 2000, False),
        ("(x+x+)+y", "x" * 3000, False),
    ]
    for pattern, text, expected in evil:
        compiled = compile_pattern(pattern)
        start = time.monotonic()
        assert compiled.search(text) is expected
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"{pattern} took {elapsed:.2f}s"


def test_state_cap_rejects_huge_expansions():
    with pytest.raises(PatternSyntaxError):
        compile_pattern("a{1000}" + "b{1000}")  # > MAX_STATES states
    # Just under the cap is fine.
    compile_pattern("a{10}b{10}c{10}")


def test_pattern_length_cap():
    with pytest.raises(PatternSyntaxError):
        compile_pattern("a" * 3000)


def test_subject_length_cap():
    compiled = compile_pattern("a+")
    with pytest.raises(ValueError):
        compiled.search("a" * 70000)


def test_unsupported_constructs():
    bad = [
        r"(a)\1", r"(?=foo)", r"(?!foo)", r"(?<=x)y", r"(?<!x)y",
        r"(?P<name>x)", r"(?P=name)", r"(?i)x", r"(?#comment)x", r"(?>a)",
        r"a*+", r"a++", r"a?+", r"\bword", r"\Aabc", r"\Zabc", r"\z",
        r"\1", r"\0", r"\x4", r"\A", r"\q", r"a{2,1}", r"a{1001}",
        r"[z-a]", r"[", r"(", r")", r"a**", r"*a", "\\",
    ]
    for pattern in bad:
        with pytest.raises(PatternSyntaxError):
            compile_pattern(pattern)


def test_escaped_metacharacters():
    assert compile_pattern(r"a\+b").search("a+b")
    assert not compile_pattern(r"a\+b").search("aab")
    assert compile_pattern(r"\[x\]").search("[x]")
    assert compile_pattern(r"a\.b").search("a.b")
    assert not compile_pattern(r"a\.b").search("axb")
    assert compile_pattern(r"\x41+").search("AAA")
    assert compile_pattern(r"a\nb").search("a\nb")


def test_empty_alternation_branch():
    assert compile_pattern("a|").search("")
    assert compile_pattern("a|").search("a")
    assert compile_pattern("(|a)b").search("b")


def test_dot_excludes_newline():
    assert not compile_pattern("a.b").search("a\nb")
    assert compile_pattern("a.b").search("axb")
