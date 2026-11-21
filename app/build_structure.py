"""Image build step: precompute every source file's structure so the selector never parses whole files.

    python build_structure.py <data_dir>     reads source.pack, writes structure.pack
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pack  # noqa: E402
from structure import analyze  # noqa: E402


def main(data_dir: str) -> None:
    source = pack.Pack(os.path.join(data_dir, "source.pack"))
    target = os.path.join(data_dir, "structure.pack")
    members = {}
    for name in sorted(source.names()):
        if name.endswith(".py"):
            members[name] = json.dumps(analyze(source.read(name).decode("utf-8")), separators=(",", ":")).encode("utf-8")
    pack.write(target, members, level=6)
    count = len(members)
    print(f"structure for {count} files -> {target}")


if __name__ == "__main__":
    main(sys.argv[1])
