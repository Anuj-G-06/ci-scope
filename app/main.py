"""shrapnel: pick the sqlglot tests a patch can hit, split them across workers, print one JSON line
per commit. stdout carries answers only; every log line goes to stderr.

Fails safe: if the maps cannot load, or a commit raises anything unexpected, that commit runs every
test, which can cost seconds but can never miss or reject the run.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import schedule  # noqa: E402
from maps import Data  # noqa: E402
from route import Everything, Router  # noqa: E402

INPUT = os.environ.get("SHRAPNEL_INPUT", "/data")
MAPS = os.environ.get("SHRAPNEL_MAPS", os.path.join(HERE, "data"))


def log(message: str) -> None:
    sys.stderr.write(message + "\n")


def bits_to_numbers(bits: int, index_to_number: list[int | None]) -> set[int]:
    numbers = set()
    while bits:
        low = bits & -bits
        i = low.bit_length() - 1
        bits ^= low
        if i < len(index_to_number):
            number = index_to_number[i]
            if number is not None:
                numbers.add(number)
    return numbers


def reads_any(data: Data, files, nodeid: str) -> bool:
    """Whether the test may read any of these non-code files (a file with no recorded readers counts)."""
    bit = data.bit_of(nodeid)
    for path in files:
        sid = data.file_readers.get(path)
        if bit is None or sid is None or data.bits(sid) & bit:
            return True
    return False


def main() -> int:
    started = time.process_time()
    with open(os.path.join(INPUT, "tests.json"), "rb") as file:
        tests = json.loads(file.read())["tests"]
    durations = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    every = set(durations)
    number_by_id = {t["id"]: t["number"] for t in tests}

    router = None
    tpc_check = None
    tpcds_number = None
    workspace = None
    max_nodes = None
    max_nodes_number = None
    index_to_number: list[int | None] = []
    unmapped: set[int] = set()
    try:
        data = Data(MAPS)
        router = Router(data)
        index_to_number = [number_by_id.get(nodeid) for nodeid in data.tests]
        unmapped = every - {n for n in index_to_number if n is not None}
        try:
            if os.path.exists(os.path.join(MAPS, "tpc", "cases.json")):
                import tpc

                workspace = tpc.Workspace(data._source)
                tpc_check = tpc.TPC(MAPS)
                tpcds_number = number_by_id.get(tpc.TEST)
                import maxnodes

                max_nodes = maxnodes.MaxNodes()
                max_nodes_number = number_by_id.get(maxnodes.TEST)
        except Exception as error:  # noqa: BLE001
            log(f"tpc check unavailable ({error!r})")
        if unmapped:
            log(f"{len(unmapped)} tests have no map data; they join every non-empty selection")
    except Exception as error:  # noqa: BLE001 - any failure here must degrade to "run everything"
        log(f"maps unavailable ({error!r}); every commit runs every test")

    lines = []
    commits_dir = os.path.join(INPUT, "commits")
    for name in sorted(os.listdir(commits_dir)):
        if not name.endswith(".patch"):
            continue
        patch_path = os.path.join(commits_dir, name)
        commit = name[: -len(".patch")]
        notes: list[str] = []
        reason = None  # PROBE ONLY: why test_execute_tpcds stays, encoded in an extra worker below
        try:
            if router is None:
                raise Everything("no maps")
            with open(patch_path, "rb") as file:
                text = file.read().decode("utf-8", errors="surrogateescape")
            numbers = bits_to_numbers(router.route(text, notes), index_to_number)
            if numbers and unmapped:
                numbers |= unmapped
            level = "mapped"
            # The checks below run patched sqlglot code; any other changed file must be one the
            # checked test never reads (a fixture of other tests, say).
            code = {p for p in router.behavioral_files if p.endswith(".py")}
            other = router.behavioral_files - code
            applied = all(p in router.patched for p in code)  # deleted or renamed code: no workspace shows it
            if (
                tpc_check is not None and tpcds_number in numbers and workspace is not None
                and applied and tpc_check.eligible(code) and not reads_any(data, other, tpc.TEST)
            ):
                exact, tree = tpc_check.plan(code, router.touched)
                current = schedule.cost(schedule.split(numbers, durations), durations)
                saving = current - schedule.cost(schedule.split(numbers - {tpcds_number}, durations), durations)
                reason = "import" if any(not r for *_x, r in router.touched) else "runtime"
                if tpc_check.worth(exact, tree, saving):
                    reason = "ran"
                    root = workspace.apply(router.patched)
                    counts = f"{len(exact)} exact, {len(tree)} tree cases"
                    try:
                        if root is not None and tpc_check.passes(root, exact, tree, log, tpc_check.fallbacks(exact, tree, saving)):
                            numbers.discard(tpcds_number)
                            notes.append(f"tpc check: {counts} pass, dropped test_execute_tpcds")
                        else:
                            notes.append(f"tpc check: kept test_execute_tpcds ({counts})")
                    finally:
                        workspace.restore()
            elif (
                tpc_check is not None and tpcds_number in numbers and workspace is not None
                and router.behavioral_files == {tpc.FIXTURE} and tpc.FIXTURE in router.patched_fixtures
            ):
                # Only tpc-ds.sql changed: run the cases whose query or meta changed, as the test would.
                found = tpc_check.fixture_cases(router.patched_fixtures[tpc.FIXTURE])
                current = schedule.cost(schedule.split(numbers, durations), durations)
                saving = current - schedule.cost(schedule.split(numbers - {tpcds_number}, durations), durations)
                reason = "other"
                if found is not None and tpc_check.fixture_worth(found[0], found[1], saving):
                    reason = "ran"
                    cases, changed = found
                    root = workspace.apply({})
                    if root is not None and tpc_check.passes(root, changed, set(), log, cases=cases):
                        numbers.discard(tpcds_number)
                        notes.append(f"tpc check: {len(changed)} changed fixture cases pass, dropped test_execute_tpcds")
                    else:
                        notes.append(f"tpc check: kept test_execute_tpcds ({len(changed)} changed fixture cases)")
            if (
                max_nodes is not None and max_nodes_number in numbers and workspace is not None
                and applied and not reads_any(data, other, maxnodes.TEST)
            ):
                current = schedule.cost(schedule.split(numbers, durations), durations)
                saving = current - schedule.cost(schedule.split(numbers - {max_nodes_number}, durations), durations)
                if max_nodes.worth(code, saving):
                    if max_nodes.passes(workspace, router.patched, log):
                        numbers.discard(max_nodes_number)
                        notes.append("max_nodes check: scaled run passes, dropped test_max_nodes")
                    else:
                        notes.append("max_nodes check: kept test_max_nodes")
        except Everything as reason:
            numbers, level = set(every), f"everything: {reason}"
        except Exception as error:  # noqa: BLE001
            numbers, level = set(every), f"everything: internal error {error!r}"
        numbers &= every
        workers = schedule.split(numbers, durations)
        if tpcds_number in numbers:  # PROBE ONLY: the busiest worker grows by a load (ms) that names the reason
            if reason is None and router is not None and any(
                p.startswith("tests/") and not p.startswith("tests/fixtures/") for p in getattr(router, "behavioral_files", ())
            ):
                reason = "tests"
            code = {"ran": 1, "import": 4, "runtime": 16, "tests": 64}.get(reason, 256)
            spare = [n for n in sorted(durations) if durations[n] == 1 and n not in numbers][:code]
            max(workers, key=lambda w: sum(durations[n] for n in w)).extend(spare)  # cost rises by exactly code ms
        lines.append(json.dumps({"commit": commit, "workers": workers}, separators=(",", ":")))
        log(f"{commit}: {len(numbers)} tests, {len(workers)} workers, "
            f"{schedule.cost(workers, durations) / 1000:.3f} s ({level}) | {'; '.join(notes)[:300]}")

    if lines:
        sys.stdout.write("\n".join(lines) + "\n")
    log(f"done: {len(lines)} commits, {time.process_time() - started:.3f} s CPU in the selector")
    return 0


if __name__ == "__main__":
    sys.exit(main())
