from __future__ import annotations
from typing import Any
from src.data.dataset.visdoc_utils import stable_id

def _text(value: Any) -> str:
    return " ".join(str(value or "").split())

def _convert_example(row: dict[str, Any]) -> dict[str, Any]:
    query = _text(row.get("qry"))
    memory = _text(row.get("pos_text"))
    if not query or not memory:
        raise ValueError(f"EvoTrain row is missing query/memory: uid={row.get('uid')}")
    positive_group = f"evo_memory:memory:{memory}"
    return {
        "query_text": query,
        "query_image": None,
        "pos_text": memory,
        "pos_image": None,
        "neg_text": "",
        "neg_image": None,
        "positive_group_id": positive_group,
        "sampling_group_id": f"evo_memory:query:{query}",
        "pair_id": stable_id(positive_group),
        "task_id": stable_id("task:memory_retrieval"),
        "global_dataset_name": "memory/EvoTrain-180K",
    }
