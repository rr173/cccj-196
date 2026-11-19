"""Bounded-execution regular expressions for quality string patterns.

Data-quality contracts may come from teams that do not control the data, so a
pattern must never be able to stall an audit.  This module therefore does
*not* delegate matching to Python's backtracking :mod:`re` engine.  Instead a
pattern is compiled once (at contract load time) into a Thompson NFA and
executed with a Pike-VM simulation whose cost is strictly
``O(len(value) * nfa_states)`` -- there is no backtracking and therefore no
catastrophic exponential blow-up, by construction.

Supported syntax (a deliberate, fully-anchored-safe subset):

* literals and escaped metacharacters (``\\.``, ``\\*``, ``\\\\``, ...);
* ``.`` (any character except ``\\n``);
* character classes ``[abc]``, ``[a-z]``, ``[^a-z]`` with escapes inside;
* predefined ASCII classes ``\\d \\D \\w \\W \\s \\S``
  (``\\w`` = ``[A-Za-z0-9_]``, ``\\d`` = ``[0-9]``,
  ``\\s`` = ``[ \\t\\n\\r\\f\\v]`` -- ASCII semantics, never Unicode);
* anchors ``^`` (absolute start) and ``$`` (absolute end -- like ``\\Z``,
  there is no special trailing-newline behaviour);
* quantifiers ``*``, ``+``, ``?``, ``{m}``, ``{m,}``, ``{m,n}`` with
  ``0 <= m <= n <= MAX_QUANTIFIER``;
* groups ``(...)`` and ``(?:...)`` and alternation ``|``;
* escape shorthands ``\\n \\t \\r \\f \\v \\a`` and ``\\xHH``.

Rejected at compile time (raised as :class:`PatternSyntaxError`, which the
contract loader wraps into a contract-location error): backreferences,
look-ahead/look-behind, named groups, inline flags, atomic groups, possessive
quantifiers, word-boundary/zero-width assertions (``\\b \\B \\A \\Z \\z \\G``),
octal/unicode escapes, and anything else not listed above.

Hard resource bounds (all checked at compile time):

* pattern text length <= ``MAX_PATTERN_LEN``;
* quantifier bounds <= ``MAX_QUANTIFIER``;
* expanded NFA state count <= ``MAX_STATES``.

Matching semantics are *search* (unanchored): the pattern matches when it is
found anywhere in the value.  Use ``^...$`` to require a full-value match.
"""

from __future__ import annotations

import bisect

MAX_PATTERN_LEN = 2048
MAX_QUANTIFIER = 1000
MAX_STATES = 1024
#: Subject strings longer than this are never fed to the NFA; the rule
#: executor reports them with a dedicated reason code instead.
MAX_SUBJECT_LEN = 65536

_META = set(".^$*+?{}[]\\|()")
_CLASS_ESCAPES = {"d", "D", "w", "W", "s", "S"}
_SIMPLE_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a",
}
_UNSUPPORTED_ESCAPES = set("bBAZzGuU0123456789")

_ASCII_DIGIT = ((0x30, 0x39),)
_ASCII_WORD = ((0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A))
_ASCII_SPACE = ((0x09, 0x0D), (0x20, 0x20),)


class PatternSyntaxError(Exception):
    """A pattern is malformed or uses an unsupported construct."""

    def __init__(self, problem: str, position: int) -> None:
        super().__init__(f"at pattern offset {position}: {problem}")
        self.problem = problem
        self.position = position


# ------------------------------------------------------------------ AST
class _Lit:
    __slots__ = ("ch",)

    def __init__(self, ch: str) -> None:
        self.ch = ch


class _Dot:
    __slots__ = ()


class _Class:
    __slots__ = ("negated", "ranges")

    def __init__(self, negated: bool, ranges: list[tuple[int, int]]) -> None:
        self.negated = negated
        self.ranges = ranges


class _Anchor:
    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind  # "^" or "$"


class _Concat:
    __slots__ = ("parts",)

    def __init__(self, parts: list) -> None:
        self.parts = parts


class _Alt:
    __slots__ = ("branches",)

    def __init__(self, branches: list) -> None:
        self.branches = branches


class _Rep:
    __slots__ = ("node", "lo", "hi")

    def __init__(self, node, lo: int, hi: int | None) -> None:
        self.node = node
        self.lo = lo
        self.hi = hi  # None means unbounded


# ------------------------------------------------------------------ parser
class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, problem: str, position: int | None = None) -> PatternSyntaxError:
        return PatternSyntaxError(problem, self.pos if position is None else position)

    def peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def next(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def parse(self):
        node = self.parse_alternation()
        if self.pos != len(self.text):
            raise self.error(f"unexpected {self.text[self.pos]!r} (unbalanced parenthesis?)")
        return node

    def parse_alternation(self):
        branches = [self.parse_concat()]
        while self.peek() == "|":
            self.next()
            branches.append(self.parse_concat())
        return branches[0] if len(branches) == 1 else _Alt(branches)

    def parse_concat(self):
        parts = []
        while True:
            ch = self.peek()
            if ch is None or ch in "|)":
                break
            parts.append(self.parse_repeat())
        return _Concat(parts)

    def parse_repeat(self):
        atom = self.parse_atom()
        ch = self.peek()
        if ch == "*":
            self.next()
            atom = _Rep(atom, 0, None)
        elif ch == "+":
            self.next()
            atom = _Rep(atom, 1, None)
        elif ch == "?":
            self.next()
            atom = _Rep(atom, 0, 1)
        elif ch == "{":
            rep = self._try_parse_braced()
            if rep is None:
                return atom  # literal '{' was already consumed as atom text
            atom = _Rep(atom, rep[0], rep[1])
        else:
            return atom
        # A second quantifier directly after the first is an error
        # ("multiple repeat"); lazy/possessive suffixes are unsupported too.
        nxt = self.peek()
        if nxt in ("*", "+", "?"):
            raise self.error(
                "multiple repeat / lazy / possessive quantifiers are not "
                "supported"
            )
        if nxt == "{":
            save = self.pos
            if self._try_parse_braced() is not None:
                raise self.error("multiple repeat quantifiers", save)
            self.pos = save
        return atom

    def _try_parse_braced(self):
        """Parse ``{m}`` / ``{m,}`` / ``{m,n}``; return None for a literal '{'."""
        start = self.pos
        i = start + 1
        n = len(self.text)
        lo_digits = []
        while i < n and self.text[i].isdigit():
            lo_digits.append(self.text[i])
            i += 1
        if not lo_digits:
            return None  # not a quantifier: '{' stays literal
        lo = int("".join(lo_digits))
        hi: int | None
        if i < n and self.text[i] == "}":
            hi = lo
            i += 1
        elif i < n and self.text[i] == ",":
            i += 1
            hi_digits = []
            while i < n and self.text[i].isdigit():
                hi_digits.append(self.text[i])
                i += 1
            if i < n and self.text[i] == "}":
                hi = int("".join(hi_digits)) if hi_digits else None
                i += 1
            else:
                return None
        else:
            return None
        if hi is not None and hi < lo:
            raise self.error(f"quantifier {{{lo},{hi}}} has min > max", start)
        if lo > MAX_QUANTIFIER or (hi is not None and hi > MAX_QUANTIFIER):
            raise self.error(
                f"quantifier bound exceeds MAX_QUANTIFIER={MAX_QUANTIFIER}", start
            )
        self.pos = i
        return (lo, hi)

    def parse_atom(self):
        ch = self.next()
        if ch == "(":
            if self.peek() == "?":
                self.next()
                nxt = self.peek()
                if nxt == ":":
                    self.next()
                else:
                    raise self.error(
                        "group construct '(?...' is not supported "
                        "(only '(?:...)' non-capturing groups are)"
                    )
            node = self.parse_alternation()
            if self.peek() != ")":
                raise self.error("unbalanced '(' -- missing ')'")
            self.next()
            return node
        if ch == ")":
            raise self.error("unbalanced ')'", self.pos - 1)
        if ch == "[":
            return self.parse_class()
        if ch == ".":
            return _Dot()
        if ch == "^":
            return _Anchor("^")
        if ch == "$":
            return _Anchor("$")
        if ch == "\\":
            return self.parse_escape()
        if ch in "*+?":
            raise self.error(f"quantifier {ch!r} has nothing to repeat", self.pos - 1)
        if ch in "{}":
            return _Lit(ch)  # a bare brace is a literal (Python-re compatible)
        return _Lit(ch)

    def parse_escape(self):
        pos = self.pos - 1  # position of the backslash
        ch = self.peek()
        if ch is None:
            raise self.error("trailing backslash", pos)
        self.next()
        if ch in _SIMPLE_ESCAPES:
            return _Lit(_SIMPLE_ESCAPES[ch])
        if ch in _CLASS_ESCAPES:
            return _class_from_escape(ch)
        if ch in _UNSUPPORTED_ESCAPES:
            raise self.error(
                f"escape '\\{ch}' is not supported "
                "(no backreferences, octal/unicode escapes or zero-width assertions)",
                pos,
            )
        if ch == "x":
            hex_digits = self.text[self.pos: self.pos + 2]
            if len(hex_digits) != 2 or any(
                c not in "0123456789abcdefABCDEF" for c in hex_digits
            ):
                raise self.error("malformed \\xHH escape", pos)
            self.pos += 2
            return _Lit(chr(int(hex_digits, 16)))
        if ch in _META or ch in "-/ ":
            return _Lit(ch)
        if ch.isalpha():
            raise self.error(f"unknown escape '\\{ch}'", pos)
        return _Lit(ch)  # escaped punctuation is the literal character

    def parse_class(self):
        start = self.pos - 1
        negated = False
        if self.peek() == "^":
            negated = True
            self.next()
        ranges: list[tuple[int, int]] = []
        first = True
        while True:
            ch = self.peek()
            if ch is None:
                raise self.error("unterminated character class", start)
            if ch == "]" and not first:
                self.next()
                break
            first = False
            if ch == "\\":
                self.next()
                esc = self.parse_escape()
                if isinstance(esc, _Lit):
                    lo = hi = ord(esc.ch)
                    item_ranges = None
                elif isinstance(esc, _Class):
                    item_ranges = esc.ranges
                    if esc.negated:
                        raise self.error(
                            "negated predefined classes cannot be nested inside "
                            "a character class", start
                        )
                    lo = hi = -1
                else:  # pragma: no cover - parse_escape only yields Lit/Class
                    raise self.error("unsupported escape inside character class", start)
            else:
                self.next()
                lo = hi = ord(ch)
                item_ranges = None
            if item_ranges is not None:
                ranges.extend(item_ranges)
                continue
            # Range?  "a-z" -- but a trailing '-' is a literal.
            if self.peek() == "-" and self.pos + 1 < len(self.text) and (
                self.text[self.pos + 1] != "]"
            ):
                self.next()  # consume '-'
                hi_ch = self.peek()
                if hi_ch == "\\":
                    self.next()
                    esc = self.parse_escape()
                    if not isinstance(esc, _Lit):
                        raise self.error(
                            "class range endpoint cannot be a predefined class", start
                        )
                    hi = ord(esc.ch)
                else:
                    hi = ord(self.next())
                if hi < lo:
                    raise self.error("class range is reversed", start)
            ranges.append((lo, hi))
        if not ranges:
            raise self.error("empty character class", start)
        return _Class(negated, _normalize_ranges(ranges))


def _class_from_escape(esc: str) -> _Class:
    base = {"d": _ASCII_DIGIT, "w": _ASCII_WORD, "s": _ASCII_SPACE}[esc.lower()]
    return _Class(esc.isupper(), list(base))


def _normalize_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        mlo, mhi = merged[-1]
        if lo <= mhi + 1:
            merged[-1] = (mlo, max(mhi, hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


# ------------------------------------------------------------------ NFA
# State transitions live in parallel arrays:
#   _char[s]: list of (kind, arg, target)   kind: 0=literal 1=dot 2=class
#   _eps[s]:  list of unguarded epsilon targets
#   _guard[s]: list of (anchor, target)     anchor: "^" or "$"


class _NfaBuilder:
    def __init__(self) -> None:
        self.char: list[list] = []
        self.eps: list[list] = []
        self.guard: list[list] = []

    def new_state(self) -> int:
        self.char.append([])
        self.eps.append([])
        self.guard.append([])
        if len(self.char) > MAX_STATES:
            raise PatternSyntaxError(
                f"pattern expands to more than {MAX_STATES} NFA states "
                "(bounded-execution limit)", 0
            )
        return len(self.char) - 1

    def build(self, node) -> tuple[int, int]:
        """Return (start, accept) states for the fragment."""
        if isinstance(node, _Lit):
            s, a = self.new_state(), self.new_state()
            self.char[s].append((0, node.ch, a))
            return s, a
        if isinstance(node, _Dot):
            s, a = self.new_state(), self.new_state()
            self.char[s].append((1, None, a))
            return s, a
        if isinstance(node, _Class):
            s, a = self.new_state(), self.new_state()
            self.char[s].append((2, (node.negated, node.ranges), a))
            return s, a
        if isinstance(node, _Anchor):
            s, a = self.new_state(), self.new_state()
            self.guard[s].append((node.kind, a))
            return s, a
        if isinstance(node, _Concat):
            if not node.parts:
                s, a = self.new_state(), self.new_state()
                self.eps[s].append(a)
                return s, a
            start, accept = self.build(node.parts[0])
            for part in node.parts[1:]:
                ps, pa = self.build(part)
                self.eps[accept].append(ps)
                accept = pa
            return start, accept
        if isinstance(node, _Alt):
            s, a = self.new_state(), self.new_state()
            for branch in node.branches:
                bs, ba = self.build(branch)
                self.eps[s].append(bs)
                self.eps[ba].append(a)
            return s, a
        if isinstance(node, _Rep):
            return self._build_rep(node)
        raise PatternSyntaxError(f"internal: unknown node {node!r}", 0)  # pragma: no cover

    def _build_rep(self, node: _Rep) -> tuple[int, int]:
        s, a = self.new_state(), self.new_state()
        prev_accept = s
        for _ in range(node.lo):
            cs, ca = self.build(node.node)
            self.eps[prev_accept].append(cs)
            prev_accept = ca
        if node.hi is None:
            # loop: copy may repeat; also allow finishing
            cs, ca = self.build(node.node)
            self.eps[prev_accept].append(cs)
            self.eps[prev_accept].append(a)
            self.eps[ca].append(cs)
            self.eps[ca].append(a)
        else:
            self.eps[prev_accept].append(a)
            for _ in range(node.hi - node.lo):
                cs, ca = self.build(node.node)
                self.eps[prev_accept].append(cs)
                prev_accept = ca
                self.eps[prev_accept].append(a)
        return s, a


class CompiledPattern:
    """An immutable, bounded-execution matcher (search semantics)."""

    __slots__ = ("source", "_char", "_eps", "_guard", "_start", "_accept",
                 "_n_states", "_tables", "_start_closures")

    # Position kinds for guard evaluation: middle / start / end / start+end.
    _MID, _START, _END, _START_END = 0, 1, 2, 3

    def __init__(self, source: str, builder: _NfaBuilder,
                 start: int, accept: int) -> None:
        self.source = source
        self._char = builder.char
        self._eps = builder.eps
        self._guard = builder.guard
        self._start = start
        self._accept = accept
        self._n_states = len(builder.char)
        # Precompute, per position kind, each state's epsilon closure and the
        # per-state char-transition table whose targets are already closed.
        # Matching then only does set unions over frozen sets -- the bounded
        # O(len * states) work is shared across every search() call.
        self._tables: list[list[tuple]] = []
        self._start_closures: list[frozenset] = []
        for pos_kind in (self._MID, self._START, self._END, self._START_END):
            closures = [self._closure(s, pos_kind) for s in range(self._n_states)]
            table: list[tuple] = []
            for s in range(self._n_states):
                transitions = []
                for kind, arg, target in self._char[s]:
                    transitions.append((kind, arg, closures[target]))
                table.append(tuple(transitions))
            self._tables.append(table)
            self._start_closures.append(closures[self._start])

    @property
    def n_states(self) -> int:
        return self._n_states

    def _closure(self, state: int, pos_kind: int) -> frozenset:
        out: set[int] = set()
        stack = [state]
        while stack:
            s = stack.pop()
            if s in out:
                continue
            out.add(s)
            for t in self._eps[s]:
                if t not in out:
                    stack.append(t)
            for anchor, t in self._guard[s]:
                if anchor == "^" and pos_kind in (self._START, self._START_END):
                    if t not in out:
                        stack.append(t)
                elif anchor == "$" and pos_kind in (self._END, self._START_END):
                    if t not in out:
                        stack.append(t)
        return frozenset(out)

    def search(self, text: str) -> bool:
        """Return True when the pattern occurs anywhere in ``text``.

        Cost is O(len(text) * n_states); there is no backtracking.
        """
        n = len(text)
        if n > MAX_SUBJECT_LEN:
            # The rule executor enforces this bound before calling; refuse
            # loudly rather than silently scanning an unbounded subject.
            raise ValueError(
                f"subject of {n} chars exceeds MAX_SUBJECT_LEN={MAX_SUBJECT_LEN}"
            )
        accept = self._accept
        start = self._start
        if n == 0:
            return accept in self._start_closures[self._START_END]
        current = self._start_closures[self._START]
        if accept in current:
            return True
        tables = self._tables
        mid_table = tables[self._MID]
        mid_start = self._start_closures[self._MID]
        end_table = tables[self._END]
        end_start = self._start_closures[self._END]
        last = n - 1
        for i in range(n):
            ch = text[i]
            if i == last:
                table, start_closure = end_table, end_start
            else:
                table, start_closure = mid_table, mid_start
            nxt: set[int] = set()
            for s in current:
                for kind, arg, targets in table[s]:
                    if _predicate_matches(kind, arg, ch):
                        nxt |= targets
            # Unanchored search: a new match may start at the next position.
            nxt |= start_closure
            if accept in nxt:
                return True
            current = nxt
        return False


def _predicate_matches(kind: int, arg, ch: str) -> bool:
    if kind == 0:
        return ch == arg
    if kind == 1:
        return ch != "\n"
    negated, ranges = arg
    o = ord(ch)
    los = [r[0] for r in ranges]
    i = bisect.bisect_right(los, o) - 1
    inside = i >= 0 and ranges[i][1] >= o
    return inside != negated


def compile_pattern(pattern: str) -> CompiledPattern:
    """Validate and compile a contract pattern into a bounded matcher.

    Raises :class:`PatternSyntaxError` for malformed or unsupported patterns;
    callers (the contract loader) translate that into a contract-location
    error before any data is read.
    """
    if not isinstance(pattern, str):
        raise PatternSyntaxError("pattern must be a string", 0)
    if len(pattern) > MAX_PATTERN_LEN:
        raise PatternSyntaxError(
            f"pattern of {len(pattern)} chars exceeds MAX_PATTERN_LEN="
            f"{MAX_PATTERN_LEN}", 0
        )
    ast = _Parser(pattern).parse()
    builder = _NfaBuilder()
    start, accept = builder.build(ast)
    return CompiledPattern(pattern, builder, start, accept)
