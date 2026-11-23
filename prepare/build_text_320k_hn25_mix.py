"""Build an exact 320K three-source text mixture with 25% hard negatives.

The output parquet is ordered in 64-row homogeneous global batches.  Each
batch has one source and one negative policy, so every DDP rank receives the
same task shape after ``split_dataset_by_node``.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset.text_retrieval_dataset import (
    TEXT_FEATURES,
    _convert_bge,
    _convert_inf_ir,
    _convert_promptriever,
    _to_example,
)
from src.data.dataset.visdoc_utils import stable_id


COUNTS = {
    "BGE-M3-base": 200_000,
    "Promptriever-base": 40_000,
    "Promptriever-instruction-hard": 41_280,
    "InF-IR-hard": 38_720,
}


def _parquet_rows(
    paths: Iterable[Path], columns: list[str] | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(group, columns=columns).to_pylist():
                yield index, row
                index += 1


def _priority(seed: int, source: str, index: int) -> int:
    return stable_id(f"text320k:{seed}:{source}:{index}")


def _keep_smallest(heap: list[tuple[int, int]], count: int, priority: int, index: int) -> None:
    item = (-priority, index)
    if len(heap) < count:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _selected_indices(heap: list[tuple[int, int]]) -> set[int]:
    return {index for _, index in heap}


def _select_promptriever(paths: list[Path], seed: int) -> tuple[set[int], set[int], dict[str, int]]:
    hard_heap: list[tuple[int, int]] = []
    # Keeping hard_count + base_count base candidates guarantees at least
    # base_count remain after excluding the selected hard rows.
    base_heap: list[tuple[int, int]] = []
    hard_target = COUNTS["Promptriever-instruction-hard"]
    base_reservoir = hard_target + COUNTS["Promptriever-base"]
    total = instruction = hard_eligible = 0
    for index, row in _parquet_rows(
        paths, columns=["has_instruction", "new_negatives", "negative_passages"]
    ):
        total += 1
        has_instruction = bool(row.get("has_instruction"))
        instruction += has_instruction
        has_negative = bool(row.get("new_negatives") or row.get("negative_passages"))
        if has_instruction and has_negative:
            hard_eligible += 1
            _keep_smallest(hard_heap, hard_target, _priority(seed, "prompt-hard", index), index)
        _keep_smallest(base_heap, base_reservoir, _priority(seed, "prompt-base", index), index)
    hard = _selected_indices(hard_heap)
    base_ranked = sorted(
        ((-neg_priority, index) for neg_priority, index in base_heap if index not in hard)
    )
    base = {index for _, index in base_ranked[: COUNTS["Promptriever-base"]]}
    if len(hard) != hard_target or len(base) != COUNTS["Promptriever-base"]:
        raise RuntimeError(f"Promptriever selection failed: hard={len(hard)} base={len(base)}")
    if hard & base:
        raise RuntimeError("Promptriever base and hard selections overlap")
    return base, hard, {
        "source_rows": total,
        "instruction_rows": instruction,
        "hard_eligible_rows": hard_eligible,
    }


def _collect_bge(path: Path, seed: int) -> list[dict[str, Any]]:
    result = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rng = random.Random(_priority(seed, "bge-convert", index))
            example = _to_example(_convert_bge(row, rng, "none"), "BGE-M3", False)
            if example is None:
                raise RuntimeError(f"Invalid BGE-M3 row at index {index}")
            result.append(example)
    if len(result) != COUNTS["BGE-M3-base"]:
        raise RuntimeError(f"Expected 200000 BGE-M3 rows, found {len(result)}")
    return result


def _collect_inf_ir(path: Path, seed: int) -> tuple[list[dict[str, Any]], int]:
    rows = list(_parquet_rows([path]))
    selected = sorted(rows, key=lambda item: _priority(seed, "inf-hard", item[0]))[: COUNTS["InF-IR-hard"]]
    result = []
    for index, row in selected:
        rng = random.Random(_priority(seed, "inf-convert", index))
        example = _to_example(_convert_inf_ir(row, rng, "inf_random"), "InF-IR-hard", True)
        if example is None:
            raise RuntimeError(f"Invalid InF-IR hard-negative row at index {index}")
        result.append(example)
    return result, len(rows)


def _collect_promptriever(
    paths: list[Path], base_indices: set[int], hard_indices: set[int], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base, hard = [], []
    wanted = base_indices | hard_indices
    for index, row in _parquet_rows(paths):
        if index not in wanted:
            continue
        is_hard = index in hard_indices
        source = "Promptriever-instruction-hard" if is_hard else "Promptriever-base"
        policy = "instruction_first" if is_hard else "none"
        rng = random.Random(_priority(seed, f"prompt-convert:{source}", index))
        example = _to_example(_convert_promptriever(row, rng, policy), source, is_hard)
        if example is None:
            raise RuntimeError(f"Invalid Promptriever row at index {index} for {source}")
        (hard if is_hard else base).append(example)
    if len(base) != len(base_indices) or len(hard) != len(hard_indices):
        raise RuntimeError(
            f"Promptriever materialization failed: base={len(base)}/{len(base_indices)} "
            f"hard={len(hard)}/{len(hard_indices)}"
        )
    return base, hard


def _shuffle_rows(rows: list[dict[str, Any]], seed: int, source: str) -> None:
    random.Random(_priority(seed, f"shuffle:{source}", 0)).shuffle(rows)


def _write_batches(
    output: Path, sources: dict[str, list[dict[str, Any]]], batch_size: int, seed: int
) -> dict[str, Any]:
    blocks = []
    for source, rows in sources.items():
        if len(rows) % batch_size:
            raise RuntimeError(f"{source} count {len(rows)} is not divisible by {batch_size}")
        _shuffle_rows(rows, seed, source)
        blocks.extend((source, rows[i : i + batch_size]) for i in range(0, len(rows), batch_size))
    random.Random(seed).shuffle(blocks)

    schema = datasets.Features(TEXT_FEATURES).arrow_schema
    with pq.ParquetWriter(output, schema=schema, compression="zstd") as writer:
        for _, block in blocks:
            table = pa.Table.from_pylist(block, schema=schema)
            writer.write_table(table, row_group_size=batch_size)

    return {
        "total_rows": sum(len(rows) for rows in sources.values()),
        "global_batch_size": batch_size,
        "optimizer_steps": len(blocks),
        "source_counts": {source: len(rows) for source, rows in sources.items()},
        "source_batches": {source: len(rows) // batch_size for source, rows in sources.items()},
        "hard_negative_rows": sum(
            bool(row.get("neg_text")) for rows in sources.values() for row in rows
        ),
        "hard_negative_batches": sum(
            bool(block[0].get("neg_text")) for _, block in blocks
        ),
        "mixed_source_batches": sum(
            len({row["global_dataset_name"] for row in block}) > 1 for _, block in blocks
        ),
        "duplicate_pair_ids": sum(
            count - 1 for count in Counter(
                row["pair_id"] for rows in sources.values() for row in rows
            ).values() if count > 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = sorted((args.text_root / "promptriever-msmarco/data").glob("*.parquet"))
    if len(prompt_paths) != 27:
        raise RuntimeError(f"Expected 27 Promptriever shards, found {len(prompt_paths)}")
    prompt_base_indices, prompt_hard_indices, prompt_stats = _select_promptriever(
        prompt_paths, args.seed
    )
    prompt_base, prompt_hard = _collect_promptriever(
        prompt_paths, prompt_base_indices, prompt_hard_indices, args.seed
    )
    inf_rows, inf_source_rows = _collect_inf_ir(
        args.text_root / "inf-ir-msmarco-only/data/msmarco-00000-of-00001.parquet",
        args.seed,
    )
    sources = {
        "BGE-M3-base": _collect_bge(
            args.text_root / "bge-m3-train-200k/train.jsonl.gz", args.seed
        ),
        "Promptriever-base": prompt_base,
        "Promptriever-instruction-hard": prompt_hard,
        "InF-IR-hard": inf_rows,
    }
    manifest = _write_batches(
        args.output_dir / "train.parquet", sources, args.global_batch_size, args.seed
    )
    manifest.update({
        "physical_source_counts": {
            "BGE-M3": len(sources["BGE-M3-base"]),
            "Promptriever": len(prompt_base_indices | prompt_hard_indices),
            "InF-IR": len(sources["InF-IR-hard"]),
        },
        "promptriever_base_hard_overlap": len(prompt_base_indices & prompt_hard_indices),
        "promptriever_scan": prompt_stats,
        "inf_ir_source_rows": inf_source_rows,
        "hard_negative_fraction": manifest["hard_negative_rows"] / manifest["total_rows"],
        "seed": args.seed,
    })
    expected = {
        "total_rows": 320_000,
        "optimizer_steps": 5_000,
        "hard_negative_rows": 80_000,
        "hard_negative_batches": 1_250,
        "mixed_source_batches": 0,
        "promptriever_base_hard_overlap": 0,
    }
    for key, value in expected.items():
        if manifest[key] != value:
            raise RuntimeError(f"Manifest mismatch for {key}: {manifest[key]} != {value}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
