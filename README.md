# shrapnel

**Every change throws shrapnel. Run only the tests it hits.**

Given patches to sqlglot, it picks the tests each patch can affect and splits them across workers.

- `app/` is the selector. It uses only the Python standard library.
- `data/` holds maps built offline from runs of sqlglot's own test suite at the pinned commit: which tests run each line, which classes and fixture files each test uses, and the operand states each comparison sees. `data/source.zip` is the pinned sqlglot source the selector compares patches against.
- `LICENSES/sqlglot.txt` is sqlglot's MIT license, which covers the sqlglot source in `data/source.zip`.

Build with `docker build .`. The image reads `/data/tests.json` and `/data/commits/*.patch` and prints one JSON line per commit.
