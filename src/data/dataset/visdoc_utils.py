from __future__ import annotations
import hashlib
import random
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable
from datasets import Features, Value
from src.data.dataset.base_pair_dataset import MULTIMODAL_FEATURES

VISDOC_FEATURES = Features({
    **dict(MULTIMODAL_FEATURES),
    "positive_group_id": Value("string"),
    "sampling_group_id": Value("string"),
    "pair_id": Value("int64"),
    "task_id": Value("int64"),
    "negative_pair_id": Value("int64"),
})

def stable_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)

def attach_identity(
    example: Dict[str, Any], source: str, page_identity: str, sampling_identity: str | None = None
) -> Dict[str, Any]:
    group = f"{source}:page:{page_identity}"
    example["positive_group_id"] = group
    example["sampling_group_id"] = f"{source}:document:{sampling_identity or page_identity}"
    example["pair_id"] = stable_id(group)
    example["task_id"] = stable_id(f"task:{source}")
    example["global_dataset_name"] = source
    return example

def iter_page_aware(examples: Iterable[Dict[str, Any]], buffer_size: int, seed: int):
    """Spread repeated pages apart using bounded streaming lookahead."""
    if buffer_size <= 0:
        yield from examples
        return
    iterator = iter(examples)
    rng = random.Random(seed)
    groups = OrderedDict()
    buffered = 0
    exhausted = False

    def refill(target: int):
        nonlocal buffered, exhausted
        while not exhausted and buffered < target:
            try:
                example = next(iterator)
            except StopIteration:
                exhausted = True
                break
            groups.setdefault(example.get("sampling_group_id") or example["positive_group_id"], deque()).append(example)
            buffered += 1

    refill(buffer_size)
    while groups:
        order = list(groups)
        rng.shuffle(order)
        for group in order:
            queue = groups.get(group)
            if not queue:
                continue
            yield queue.popleft()
            buffered -= 1
            if not queue:
                groups.pop(group, None)
            refill(buffer_size)
