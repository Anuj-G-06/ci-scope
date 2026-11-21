"""Cheap proofs that a text change cannot change the program: inert comment lines, same bytecode."""

from __future__ import annotations

import re

_UNSAFE_IN_COMMENT = re.compile("[\x00-\x08\x0b\x0d-\x1f\x7f\udc80-\udcff]")
_CODE_FIELDS = (
    "co_code", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_argcount", "co_posonlyargcount",
    "co_kwonlyargcount", "co_flags", "co_nlocals", "co_stacksize", "co_exceptiontable", "co_name",
)


def comment_or_blank(line: str) -> bool:
    """A comment or blank line Python's tokenizer ignores. A NUL byte, a bare carriage return (a line
    break to Python, so code after it runs), other control characters, or bytes that are not UTF-8
    (decoded here as lone surrogates) can break or change the module, so such lines do not qualify."""
    if _UNSAFE_IN_COMMENT.search(line):
        return False
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def snippet(lines: list[str]) -> str:
    """Source for an indented region, wrapped so it parses on its own."""
    text = "\n".join(lines)
    return "if 1:\n" + text if lines and lines[0][:1] in (" ", "\t") else text


def same_code(a, b) -> bool:
    """Trivial compiler equivalence: equal bytecode, ignoring line numbers and file names."""
    if type(a) is not type(b):
        return False
    if hasattr(a, "co_code"):
        if any(getattr(a, f, None) != getattr(b, f, None) for f in _CODE_FIELDS) or len(a.co_consts) != len(b.co_consts):
            return False
        return all(same_code(x, y) for x, y in zip(a.co_consts, b.co_consts))
    if isinstance(a, (tuple, frozenset)):
        return len(a) == len(b) and all(same_code(x, y) for x, y in zip(sorted(a, key=repr), sorted(b, key=repr)))
    return a == b and repr(a) == repr(b)
