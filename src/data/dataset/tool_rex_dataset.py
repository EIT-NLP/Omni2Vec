from __future__ import annotations
import heapq
import random
from collections import defaultdict
from typing import Any

def _query_disjoint_epoch(
    rows: list[dict[str, Any]], global_batch_size: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    """Build complete global batches containing at most one row per query.

    A max-remaining heap makes high-multiplicity queries drain early while every
    batch draws distinct query groups. The incomplete tail is dropped exactly
    like ``dataloader_drop_last=true``.
    """
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sampling_group_id"]].append(row)
    rng = random.Random(seed)
    heap = []
    for query_id, query_rows in grouped.items():
        rng.shuffle(query_rows)
        heap.append((-len(query_rows), rng.random(), query_id))
    heapq.heapify(heap)

    ordered = []
    while len(heap) >= global_batch_size:
        selected = [heapq.heappop(heap) for _ in range(global_batch_size)]
        for neg_count, _, query_id in selected:
            ordered.append(grouped[query_id].pop())
            remaining = -neg_count - 1
            if remaining:
                heapq.heappush(heap, (-remaining, rng.random(), query_id))
    dropped = sum(-neg_count for neg_count, _, _ in heap)
    if len(ordered) % global_batch_size:
        raise RuntimeError("query-disjoint scheduler emitted an incomplete global batch")
    return ordered, dropped
