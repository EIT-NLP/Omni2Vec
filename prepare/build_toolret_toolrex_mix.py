"""Build a block-homogeneous ToolRet + Tool-REX training epoch.

Each original ToolRet query contributes one deterministically selected positive
and one deterministically selected task-provided negative. Tool-REX rejected
responses are intentionally excluded because they are random collision-avoidance
samples. Every global batch is query-disjoint and uses one negative policy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset.text_retrieval_dataset import TEXT_FEATURES
from src.data.dataset.tool_rex_dataset import _query_disjoint_epoch
from src.data.dataset.visdoc_utils import stable_id


TASK_ID = stable_id("task:tool_retrieval")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _example(query: str, positive: str, source: str, query_id: str, negative: str = ""):
    query = _text(query)
    positive = _text(positive)
    negative = _text(negative)
    if not query or not positive:
        raise ValueError(f"Invalid {source} row: query_id={query_id}")
    positive_id = stable_id(f"tool:{positive}")
    negative_id = stable_id(f"tool:{negative}") if negative else 0
    if negative and negative_id == positive_id:
        raise ValueError(f"Positive equals negative: source={source} query_id={query_id}")
    return {
        "query_text": query,
        "query_image": None,
        "pos_text": positive,
        "pos_image": None,
        "neg_text": negative,
        "neg_image": None,
        "positive_group_id": f"tool:{positive_id}",
        "sampling_group_id": f"{source}:query:{query_id}",
        "pair_id": positive_id,
        "negative_pair_id": negative_id,
        "task_id": TASK_ID,
        "global_dataset_name": source,
    }


def load_toolret(root: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ToolRet parquet files under {root}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                query_id = str(raw["id"])
                query = "\n".join(
                    part for part in (_text(raw.get("prompt")), _text(raw.get("query"))) if part
                )
                positives = [_text(value) for value in raw.get("positive") or []]
                negatives = [_text(value) for value in raw.get("negative") or []]
                positives = [value for value in positives if value]
                positive_set = set(positives)
                negatives = [value for value in negatives if value and value not in positive_set]
                if not positives or not negatives:
                    raise ValueError(f"ToolRet row lacks valid positive/negative: {query_id}")
                positive_index = stable_id(
                    f"toolret-positive:{seed}:{query_id}"
                ) % len(positives)
                negative_index = stable_id(
                    f"toolret-negative:{seed}:{query_id}"
                ) % len(negatives)
                rows.append(
                    _example(
                        query, positives[positive_index],
                        "tool/ToolRet-Training-20w", query_id,
                        negatives[negative_index],
                    )
                )
    return rows


def load_toolrex(path: Path) -> list[dict[str, Any]]:
    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, raw in enumerate(raw_rows):
        query = _text(raw.get("query"))
        query_id = str(raw.get("uid") or stable_id(query))
        rows.append(
            _example(
                query, raw.get("response"), "tool/Tool_REX_train_retriever_50k",
                query_id or str(index),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolret-root", type=Path, required=True)
    parser.add_argument("--toolrex-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    toolret_raw = load_toolret(args.toolret_root, args.seed)
    toolrex_raw = load_toolrex(args.toolrex_json)
    toolret, toolret_dropped = _query_disjoint_epoch(
        toolret_raw, args.global_batch_size, args.seed
    )
    toolrex, toolrex_dropped = _query_disjoint_epoch(
        toolrex_raw, args.global_batch_size, args.seed + 1
    )
    blocks = [
        ("toolret", toolret[i : i + args.global_batch_size])
        for i in range(0, len(toolret), args.global_batch_size)
    ] + [
        ("toolrex", toolrex[i : i + args.global_batch_size])
        for i in range(0, len(toolrex), args.global_batch_size)
    ]
    random.Random(args.seed).shuffle(blocks)

    output = args.output_dir / "train.parquet"
    schema = datasets.Features(TEXT_FEATURES).arrow_schema
    with pq.ParquetWriter(output, schema=schema, compression="zstd") as writer:
        for _, block in blocks:
            writer.write_table(
                pa.Table.from_pylist(block, schema=schema),
                row_group_size=args.global_batch_size,
            )

    source_blocks = Counter(source for source, _ in blocks)
    manifest = {
        "version": 1,
        "seed": args.seed,
        "global_batch_size": args.global_batch_size,
        "optimizer_steps": len(blocks),
        "total_rows": len(blocks) * args.global_batch_size,
        "source_rows_before_query_disjoint": {
            "toolret": len(toolret_raw), "toolrex": len(toolrex_raw),
        },
        "source_rows": {"toolret": len(toolret), "toolrex": len(toolrex)},
        "source_blocks": dict(source_blocks),
        "dropped_rows": {
            "toolret": toolret_dropped, "toolrex": toolrex_dropped,
        },
        "negative_policy": {
            "toolret": "one deterministic task-provided negative per original query",
            "toolrex": "no rejected_response negatives",
        },
        "positive_policy": {
            "toolret": "one deterministic positive per original query",
            "toolrex": "one released response per original row",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
