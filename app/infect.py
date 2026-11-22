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
import hashlib


class Unknown(Exception):
    pass


_PRIMITIVE = {"N", "b", "i", "f", "s", "y", "t"}
_IDENTITY = {"o_id", "e", "c"}


def _kind(a) -> str:
    tag = a[0]
    if tag == "o":
        return "o_id" if a[2] == "id" else "custom"
    return tag


_DECODE = {  # exact Python value for primitive abstractions
    "N": lambda a: None, "b": lambda a: bool(a[1]), "i": lambda a: a[1], "I": lambda a: int(a[1]),
    "f": lambda a: float(a[1]), "s": lambda a: a[1], "y": lambda a: bytes.fromhex(a[1]),
    "t": lambda a: tuple(_value(x) for x in a[1]),
    "ss": lambda a: a[-1], "es": lambda a: a[-1], "ei": lambda a: a[-1],  # compare as their str/int value
}


def _value(a):
    if a[0] not in _DECODE:
        raise Unknown(a[0])
    return _DECODE[a[0]](a)


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
            return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest() == hashed[2]
        if hashed[0] == "Y" and other[0] == "y":
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


_ORDER = {
    "<": lambda x, y: x < y, "<=": lambda x, y: x <= y, ">": lambda x, y: x > y, ">=": lambda x, y: x >= y,
    "in": lambda x, y: x in y, "not in": lambda x, y: x not in y,
}


def _outcome(op: str, a, b, same: bool | None):
    """('v', result) or ('raise', exception name)."""
    try:
        if op in ("is", "is not"):
            if same is None:
                raise Unknown("identity unknown")
            return ("v", same if op == "is" else not same)
        if op in ("==", "!="):
            equal = _equal(a, b, same)
            return ("v", equal if op == "==" else not equal)
        if op not in _ORDER or not (_comparable(a) and _comparable(b)):
            raise Unknown("ordering or membership on objects")
        return ("v", _ORDER[op](_value(a), _value(b)))
    except Unknown:
        raise
    except Exception as error:  # noqa: BLE001 - the original code would raise this too
        return ("raise", type(error).__name__)


_OPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}


_CONSTANTS = {
    type(None): lambda v: ("N",), bool: lambda v: ("b", int(v)), int: lambda v: ("i", v),
    float: lambda v: ("f", repr(v)), str: lambda v: ("s", v), bytes: lambda v: ("y", v.hex()),
}


def _constant_abs(value):
    if type(value) not in _CONSTANTS:
        raise Unknown("constant")
    return _CONSTANTS[type(value)](value)


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


def proven(data, path: str, pairs, offset: int, notes: list[str]) -> int:
    """Tests whose every recorded state gives the same outcome for every changed comparison."""
    result = None
    for old_expr, new_expr in pairs:
        site = core(old_expr)
        key = f"{site.lineno + offset}:{site.col_offset}:{site.end_lineno + offset}:{site.end_col_offset}"
        entries, overflow = data.values_at(path, key)
        if not entries:
            notes.append(f"{path}:{key} no recorded states")
            return 0
        recorded = infected = 0
        for left, right, same_lr, sid in entries:
            bits = data.bits(sid)
            recorded |= bits
            try:
                if left == "?" or evaluate(old_expr, site, left, right, bool(same_lr)) != evaluate(
                    new_expr, site, left, right, bool(same_lr)
                ):
                    infected |= bits
            except Unknown:
                infected |= bits
        if overflow is not None:
            recorded |= data.bits(overflow)
            infected |= data.bits(overflow)
        clean = recorded & ~infected
        result = clean if result is None else result & clean
        notes.append(f"{path}:{key} states {len(entries)}, clean tests {bin(clean).count('1')}")
    return result or 0
