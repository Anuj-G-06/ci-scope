"""A minimal archive: [8-byte header length][JSON index {name: [offset, length]}][zlib members].

Replaces zipfile at runtime: reading one member is a seek plus one zlib.decompress, and importing
this module costs nothing (zipfile pulls in shutil, pathlib and more, about 30 ms on every run).
"""

import json
import zlib


def write(path: str, members: dict, level: int = 9) -> None:
    blobs, index, offset = [], {}, 0
    for name in sorted(members):
        data = zlib.compress(members[name], level)
        index[name] = [offset, len(data)]
        blobs.append(data)
        offset += len(data)
    header = json.dumps(index, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as file:
        file.write(len(header).to_bytes(8, "little"))
        file.write(header)
        for blob in blobs:
            file.write(blob)


class Pack:
    def __init__(self, path: str):
        self._file = open(path, "rb")
        size = int.from_bytes(self._file.read(8), "little")
        self.index = json.loads(self._file.read(size))
        self._base = 8 + size

    def names(self):
        return self.index.keys()

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def read(self, name: str) -> bytes:
        offset, length = self.index[name]
        self._file.seek(self._base + offset)
        return zlib.decompress(self._file.read(length))
