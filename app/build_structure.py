"""Image build step: precompute every source file's structure so the selector never parses whole files.

    python build_structure.py <data_dir>     reads source.zip, writes structure.zip
"""

import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structure import analyze  # noqa: E402


def main(data_dir: str) -> None:
    source = zipfile.ZipFile(os.path.join(data_dir, "source.zip"))
    target = os.path.join(data_dir, "structure.zip")
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for name in sorted(source.namelist()):
            if name.endswith(".py"):
                out.writestr(name, json.dumps(analyze(source.read(name).decode("utf-8")), separators=(",", ":")))
                count += 1
    print(f"structure for {count} files -> {target}")


if __name__ == "__main__":
    main(sys.argv[1])
