"""Unified diffs: parse them, apply them exactly, and describe each contiguous change.

Every context and removed line is checked against the pinned file. Anything we cannot apply exactly
raises PatchError, and the caller then runs every test for that commit rather than guessing.
"""

from __future__ import annotations

import re

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


class Hunk:
    __slots__ = ("old_start", "old_count", "new_start", "new_count", "lines")

    def __init__(self, old_start: int, old_count: int, new_start: int, new_count: int):
        self.old_start, self.old_count, self.new_start, self.new_count = old_start, old_count, new_start, new_count
        self.lines: list[str] = []


class FilePatch:
    __slots__ = ("old_path", "new_path", "hunks", "binary", "renamed")

    def __init__(self, old_path: str | None, new_path: str | None):
        self.old_path = old_path  # None for a new file
        self.new_path = new_path  # None for a deleted file
        self.hunks: list[Hunk] = []
        self.binary = False
        self.renamed = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


class Block:
    """One contiguous change. Lines are numbered in the old file. A pure insertion has no removed
    lines and sits after old line `after` (0 means before the first line)."""

    __slots__ = ("first", "removed", "added", "after")

    def __init__(self, first: int, removed: list[str], added: list[str], after: int):
        self.first = first  # first removed old line, or after + 1 for an insertion
        self.removed = removed
        self.added = added
        self.after = after  # the old line the change follows

    @property
    def last(self) -> int:
        return self.first + len(self.removed) - 1

    @property
    def insertion(self) -> bool:
        return not self.removed

    @property
    def lines(self):
        """The old lines this change touches: an insertion touches the lines on both sides of it."""
        return (self.after, self.after + 1) if self.insertion else range(self.first, self.last + 1)


def _strip_prefix(path: str) -> str | None:
    path = path.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def parse(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    remaining_old = remaining_new = 0
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            match = re.match(r"^diff --git a/(.*) b/(.*)$", line)
            current = FilePatch(match.group(1) if match else None, match.group(2) if match else None)
            files.append(current)
            hunk = None
            continue
        if current is None:
            if line.startswith("--- "):  # a diff without the git header
                current = FilePatch(None, None)
                files.append(current)
            else:
                continue
        if hunk is not None and (remaining_old > 0 or remaining_new > 0):
            tag = line[:1]
            if tag == " ":
                remaining_old -= 1
                remaining_new -= 1
            elif tag == "-":
                remaining_old -= 1
            elif tag == "+":
                remaining_new -= 1
            elif tag == "\\":
                hunk.lines.append(line)
                continue
            elif line == "":  # some tools drop the space of an empty context line
                line = " "
                remaining_old -= 1
                remaining_new -= 1
            else:
                raise PatchError(f"unexpected line inside a hunk: {line[:40]!r}")
            hunk.lines.append(line)
            continue
        if line.startswith("\\") and hunk is not None:
            hunk.lines.append(line)
            continue
        match = _HUNK.match(line)
        if match:
            old_start, old_count, new_start, new_count = match.groups()
            hunk = Hunk(
                int(old_start),
                1 if old_count is None else int(old_count),
                int(new_start),
                1 if new_count is None else int(new_count),
            )
            current.hunks.append(hunk)
            remaining_old, remaining_new = hunk.old_count, hunk.new_count
            continue
        if line.startswith("--- "):
            current.old_path = _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            current.new_path = _strip_prefix(line[4:])
        elif line.startswith("new file mode"):
            current.old_path = None
        elif line.startswith("deleted file mode"):
            current.new_path = None
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            current.renamed = True
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            current.binary = True
    for f in files:
        if f.old_path is None and f.new_path is None:
            raise PatchError("a file section names no path")
    return files


def apply(before: str, hunks: list[Hunk]) -> tuple[str, list[Block]]:
    """Applies hunks to the old text. Returns the new text and the change blocks."""
    old = before.split("\n")
    out: list[str] = []
    blocks: list[Block] = []
    pos = 0  # 0-based index of the next old line to consume
    for hunk in hunks:
        start = hunk.old_start - 1 if hunk.old_count > 0 else hunk.old_start
        if start < pos or start > len(old):
            raise PatchError("hunks overlap or point past the end of the file")
        out.extend(old[pos:start])
        pos = start
        removed: list[str] = []
        added: list[str] = []
        block_first = None

        def flush():
            nonlocal removed, added, block_first
            if removed or added:
                after = (block_first - 1) if removed else pos
                first = block_first if removed else pos + 1
                blocks.append(Block(first, removed, added, after))
            removed, added, block_first = [], [], None

        for line in hunk.lines:
            tag, body = line[:1], line[1:]
            if tag == "\\":
                continue
            if tag == " ":
                flush()
                if pos >= len(old) or old[pos] != body:
                    raise PatchError(f"context does not match old line {pos + 1}")
                out.append(body)
                pos += 1
            elif tag == "-":
                if added and not removed:
                    flush()
                if pos >= len(old) or old[pos] != body:
                    raise PatchError(f"removed line does not match old line {pos + 1}")
                if block_first is None:
                    block_first = pos + 1
                removed.append(body)
                pos += 1
            elif tag == "+":
                added.append(body)
                out.append(body)
            else:
                raise PatchError(f"bad hunk line {line[:20]!r}")
        flush()
    out.extend(old[pos:])
    return "\n".join(out), blocks


def splice(before_lines: list[str], first: int, last: int, blocks: list[Block]) -> list[str]:
    """New text of old lines [first, last] (1-based, inclusive) after applying the blocks that lie
    inside that span. A block inserted after line `last` is outside it; one after first-1 is inside."""
    result: list[str] = []
    inside = sorted(
        (
            b
            for b in blocks
            if (b.insertion and first <= b.after < last) or (not b.insertion and first <= b.first and b.last <= last)
        ),
        key=lambda b: (b.first, 0 if b.insertion else 1),
    )
    line = first
    for block in inside:
        if block.insertion:
            while line <= block.after:
                result.append(before_lines[line - 1])
                line += 1
            result.extend(block.added)
        else:
            while line < block.first:
                result.append(before_lines[line - 1])
                line += 1
            result.extend(block.added)
            line = block.last + 1
    while line <= last:
        result.append(before_lines[line - 1])
        line += 1
    return result
