"""Split selected tests across workers to minimize busiest-worker milliseconds + 2,000 ms per worker.

Minimizing the busiest worker is multiprocessor scheduling (P||Cmax), NP-hard in general. For each
worker count W we run LPT (longest first onto the lightest worker, within 4/3 of optimal), skip any W
whose lower bound max(longest test, ceil(total / W)) + 2000 W cannot beat the best so far, and solve
W = 2 exactly with a subset-sum bitset when that could still win.
"""

from __future__ import annotations

import heapq

WORKER_MS = 2000
MAX_WORKERS = 8


def _lpt(items: list[tuple[int, int]], workers: int) -> tuple[int, list[list[int]]]:
    heap = [(0, i) for i in range(workers)]
    bins: list[list[int]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for ms, number in items:
        load, i = heapq.heappop(heap)
        bins[i].append(number)
        loads[i] = load + ms
        heapq.heappush(heap, (loads[i], i))
    used = [b for b in bins if b]
    return max(loads) + WORKER_MS * len(used), used


def _exact_two(items: list[tuple[int, int]]) -> tuple[int, list[list[int]]] | None:
    """Optimal 2-way split by subset sums. O(K * total / 64) word operations."""
    total = sum(ms for ms, _ in items)
    if total > 2_000_000:  # 2,000 s of tests: never happens with sqlglot's 103 s suite
        return None
    reach = 1  # bit s set when some subset sums to s
    history = []
    for ms, _ in items:
        history.append(reach)
        reach |= reach << ms
    target = total // 2
    mask = (1 << (target + 1)) - 1
    below = reach & mask
    best = below.bit_length() - 1  # largest reachable sum <= total/2
    # Walk back to recover one subset with that sum.
    chosen, remaining = [], best
    for (ms, number), prior in zip(reversed(items), reversed(history)):
        if remaining >= ms and (prior >> (remaining - ms)) & 1:
            chosen.append(number)
            remaining -= ms
    if remaining != 0:
        return None
    picked = set(chosen)
    other = [number for _, number in items if number not in picked]
    bins = [b for b in (chosen, other) if b]
    return max(best, total - best) + WORKER_MS * len(bins), bins


def split(numbers, durations: dict[int, int]) -> list[list[int]]:
    """Workers for the given test numbers. Deterministic for a given input."""
    if not numbers:
        return []
    items = sorted(((durations[n], n) for n in numbers), key=lambda x: (-x[0], x[1]))
    total = sum(ms for ms, _ in items)
    longest = items[0][0]

    def bound(workers: int) -> int:
        return max(longest, -(-total // workers)) + WORKER_MS * workers

    best_cost, best_bins = _lpt(items, 1)
    lpt_two = None
    for workers in range(2, min(MAX_WORKERS, len(items)) + 1):
        if bound(workers) >= best_cost:
            continue
        cost, bins = _lpt(items, workers)
        if workers == 2:
            lpt_two = cost
        if cost < best_cost:
            best_cost, best_bins = cost, bins
    # Exact two-way split, only when it could still beat everything found and LPT left a gap.
    if lpt_two is not None and lpt_two > bound(2) and bound(2) < best_cost:
        exact = _exact_two(items)
        if exact is not None and exact[0] < best_cost:
            best_cost, best_bins = exact
    return best_bins


def cost(bins: list[list[int]], durations: dict[int, int]) -> int:
    used = [b for b in bins if b]
    if not used:
        return 0
    return max(sum(durations[n] for n in b) for b in used) + WORKER_MS * len(used)
