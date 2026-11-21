"""Equivalence modulo recorded states (weak mutation): prove that a changed comparison gives the same
outcome as the original on every operand state a test produced at the pinned commit.

If a test's every evaluation of the changed comparison has the same outcome (same value, or the same
exception) under the old and the new code, its execution cannot diverge there, so the test cannot
observe the change. Anything this module cannot model exactly raises Unknown, and the test stays.

Abstract operands, as recorded by the mapping run:
    ("N",) None   ("b", 0|1)   ("i", int)   ("I", "big int")   ("f", repr)   ("s", str)
    ("S", len, sha1)   ("y", hex)   ("Y", len, sha1)   ("t", [items])   ("c", class name)
    ("e", enum, member)        enum with identity equality
    ("es", enum, member, str)  ("ei", enum, member, int)   enum mixing in str / int equality
    ("ss", class, str)         str subclass with str equality
    ("o", class, "id")         object whose == / != are identity
    ("o", class, "custom")     anything else (never evaluable)
"""

from __future__ import annotations

import ast


class Unknown(Exception):
    pass


_PRIMITIVE = {"N", "b", "i", "f", "s", "y", "t"}
_IDENTITY = {"o_id", "e", "c"}


def _kind(a) -> str:
    tag = a[0]
    if tag == "o":
        return "o_id" if a[2] == "id" else "custom"
    return tag


def _value(a):
    """Exact Python value for primitive abstractions."""
    tag = a[0]
    if tag == "N":
        return None
    if tag == "b":
        return bool(a[1])
    if tag == "i":
        return a[1]
    if tag == "I":
        return int(a[1])
    if tag == "f":
        return float(a[1])
    if tag == "s":
        return a[1]
    if tag == "y":
        return bytes.fromhex(a[1])
    if tag == "t":
        return tuple(_value(x) for x in a[1])
    if tag in ("ss", "es"):
        return a[-1]  # equality and ordering behave as for the str value
    if tag == "ei":
        return a[-1]
    raise Unknown(tag)


def _comparable(a) -> bool:
    return a[0] in _PRIMITIVE or a[0] in ("I", "ss", "es", "ei")


def _equal(a, b, same: bool | None) -> bool:
    ka, kb = _kind(a), _kind(b)
    if ka == "custom" or kb == "custom" or ka == "?" or kb == "?":
        raise Unknown("custom equality")
    if ka in ("S", "Y") or kb in ("S", "Y"):
        if ka == kb:
            return list(a[1:]) == list(b[1:])
        hashed, other = (a, b) if ka in ("S", "Y") else (b, a)
        if hashed[0] == "S" and other[0] in ("s", "ss", "es"):
            text = other[-1]
            if len(text) != hashed[1]:
                return False
            import hashlib  # rare: only long strings are hashed

            return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest() == hashed[2]
        if hashed[0] == "Y" and other[0] == "y":
            import hashlib

            data = bytes.fromhex(other[1])
            return len(data) == hashed[1] and hashlib.sha1(data).hexdigest() == hashed[2]
        raise Unknown("hashed value")
    if ka in _IDENTITY or kb in _IDENTITY:
        # object.__eq__ is identity; a primitive's __eq__ returns NotImplemented for foreign types,
        # so Python falls back to identity.
        if same is None:
            raise Unknown("identity unknown")
        return same
    return _value(a) == _value(b)


def _outcome(op: str, a, b, same: bool | None):
    """('v', result) or ('raise', exception name)."""
    try:
        if op == "is":
            if same is None:
                raise Unknown("identity unknown")
            return ("v", same)
        if op == "is not":
            if same is None:
                raise Unknown("identity unknown")
            return ("v", not same)
        if op == "==":
            return ("v", _equal(a, b, same))
        if op == "!=":
            return ("v", not _equal(a, b, same))
        if not (_comparable(a) and _comparable(b)):
            raise Unknown("ordering or membership on objects")
        left, right = _value(a), _value(b)
        if op == "<":
            return ("v", left < right)
        if op == "<=":
            return ("v", left <= right)
        if op == ">":
            return ("v", left > right)
        if op == ">=":
            return ("v", left >= right)
        if op == "in":
            return ("v", left in right)
        if op == "not in":
            return ("v", left not in right)
    except Unknown:
        raise
    except Exception as error:  # noqa: BLE001 - the original code would raise this too
        return ("raise", type(error).__name__)
    raise Unknown(op)


_OPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}


def _constant_abs(value):
    if value is None:
        return ("N",)
    if type(value) is bool:
        return ("b", int(value))
    if type(value) is int:
        return ("i", value)
    if type(value) is float:
        return ("f", repr(value))
    if type(value) is str:
        return ("s", value)
    if type(value) is bytes:
        return ("y", value.hex())
    raise Unknown("constant")


def same(a, b) -> bool:
    """Structural equality of syntax trees, ignoring positions (what ast.dump compares), with an
    early exit. O(size of the smaller tree), versus ast.dump's string building."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        for field in a._fields:
            if not same(getattr(a, field, None), getattr(b, field, None)):
                return False
        return True
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def core(node: ast.AST) -> ast.Compare | None:
    """The single binary comparison at the heart of a comparison expression (maybe under `not`)."""
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        node = node.operand
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return node
    return None


def _operand(node: ast.AST, site: ast.Compare, left, right):
    """Abstract value of an operand of the new comparison: the site's left or right, or a constant."""
    if same(node, site.left):
        return "L", left
    if same(node, site.comparators[0]):
        return "R", right
    if isinstance(node, ast.Constant):
        return "C", _constant_abs(node.value)
    raise Unknown("operand is a new expression")


def _identity(a_ref, a, b_ref, b, same_lr: bool) -> bool | None:
    if a_ref in ("L", "R") and b_ref in ("L", "R"):
        return True if a_ref == b_ref else same_lr
    constant, other = (a, b) if a_ref == "C" else (b, a)
    if constant[0] == "N":
        return other[0] == "N"
    if constant[0] == "b":
        return other[0] == "b" and other[1] == constant[1]
    return None  # identity of other constants is an implementation detail


def evaluate(expr: ast.AST, site: ast.Compare, left, right, same_lr: bool):
    """Outcome of a comparison expression over one recorded state of `site`."""
    negations = 0
    node = expr
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negations += 1
        node = node.operand
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        raise Unknown("not a binary comparison")
    a_ref, a = _operand(node.left, site, left, right)
    b_ref, b = _operand(node.comparators[0], site, left, right)
    if a_ref == "C" and b_ref == "C":
        raise Unknown("two constants")
    op = _OPS[type(node.ops[0])]
    same = _identity(a_ref, a, b_ref, b, same_lr)
    outcome = _outcome(op, a, b, same)
    if outcome[0] == "v" and negations:
        value = outcome[1]
        if not isinstance(value, bool):
            raise Unknown("negating a non-bool")
        if negations % 2:
            outcome = ("v", not value)
    return outcome


def changed_comparisons(old: ast.AST, new: ast.AST) -> list[tuple[ast.AST, ast.AST]] | None:
    """Pairs (old comparison expression, new comparison expression) where the trees differ, or None
    when any difference is something other than a comparison we can re-evaluate."""
    pairs: list[tuple[ast.AST, ast.AST]] = []

    def walk(a, b) -> bool:
        if isinstance(a, ast.AST) and isinstance(b, ast.AST):
            if core(a) is not None and core(b) is not None:
                if not same(a, b):
                    pairs.append((a, b))
                return True
            if type(a) is not type(b):
                return False
            return all(walk(getattr(a, field, None), getattr(b, field, None)) for field in a._fields)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(walk(x, y) for x, y in zip(a, b))
        return type(a) is type(b) and a == b

    return pairs if walk(old, new) and pairs else None
