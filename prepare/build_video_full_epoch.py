"""Build a deterministic one-pass manifest for all six local video sources."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset.text_retrieval_dataset import TEXT_FEATURES
from src.data.dataset.video_local_dataset import (
    _json_retrieval_examples,
    _llava_examples,
    _retrieval_rows,
    _ssv2_examples,
)
from src.data.dataset.visdoc_utils import iter_page_aware


SOURCES = (
    ("MSR-VTT", "retrieval", 9_000),
    ("DiDeMo", "retrieval", 8_395),
    ("MSVD", "retrieval", 1_200),
    ("LLaVA-Hound", "caption", 301_751),
    ("LLaVA-Hound", "qa", 240_000),
    ("SmthSmthV2", "classification", 168_913),
)


def _source_key(source: str, mode: str) -> str:
    return f"{source}-{mode}"


def _source_examples(root: str, source: str, mode: str, seed: int, num_frames: int):
    if source in {"MSR-VTT", "DiDeMo", "MSVD"}:
        annotation = {
            "MSR-VTT": "msrvtt_train_9k.json",
            "DiDeMo": "didemo_train.json",
            "MSVD": "msvd_train.json",
        }[source]
        rows = _retrieval_rows(root, source, annotation, "raw_videos")
        examples = list(_json_retrieval_examples(rows, source, num_frames, seed))
    elif source == "LLaVA-Hound":
        examples = list(_llava_examples(root, mode, num_frames))
    elif source == "SmthSmthV2":
        examples = list(_ssv2_examples(root, num_frames))
    else:
        raise ValueError(f"Unsupported source: {source}/{mode}")
    random.Random(seed).shuffle(examples)
    yield from iter_page_aware(examples, buffer_size=512, seed=seed)


def _normalized(example: dict) -> dict:
    return {key: (0 if key == "negative_pair_id" else example.get(key)) for key in TEXT_FEATURES}


def build_block_schedule(block_counts: dict[str, int], seed: int) -> list[tuple[str, int]]:
    schedule = [(source, index) for source, count in block_counts.items() for index in range(count)]
    random.Random(seed).shuffle(schedule)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema = TEXT_FEATURES.arrow_schema
    temporary_root = Path(tempfile.mkdtemp(prefix="video-full-epoch-", dir="/tmp"))
    source_files = {}
    source_counts = {}
    tails = []
    try:
        for offset, (source, mode, expected_count) in enumerate(SOURCES):
            key = _source_key(source, mode)
            path = temporary_root / f"{key}.parquet"
            writer = pq.ParquetWriter(path, schema, compression="zstd")
            block = []
            count = 0
            for example in _source_examples(
                str(args.dataset_root), source, mode, args.seed + offset, args.num_frames
            ):
                block.append(_normalized(example))
                count += 1
                if len(block) == args.global_batch_size:
                    writer.write_table(pa.Table.from_pylist(block, schema=schema), row_group_size=len(block))
                    block = []
            writer.close()
            if count != expected_count:
                raise RuntimeError(f"{key}: expected {expected_count} examples, got {count}")
            source_files[key] = path
            source_counts[key] = count
            tails.extend((key, row) for row in block)
            print(
                f"Prepared {key}: rows={count}, full_blocks={count // args.global_batch_size}, "
                f"tail={len(block)}", flush=True
            )

        readers = {key: pq.ParquetFile(path) for key, path in source_files.items()}
        block_counts = {key: reader.num_row_groups for key, reader in readers.items()}
        schedule = build_block_schedule(block_counts, args.seed)
        random.Random(args.seed + 10_000).shuffle(tails)
        tail_padding = (-len(tails)) % args.global_batch_size
        selected_tails = list(tails)
        if tail_padding:
            selected_tails.extend(tails[:tail_padding])
        tail_full_rows = len(selected_tails)
        tail_selected_counts = Counter(source for source, _ in selected_tails)

        output_path = args.output_dir / "train.parquet"
        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
        for source, row_group in schedule:
            table = readers[source].read_row_group(row_group)
            if table.num_rows != args.global_batch_size:
                raise RuntimeError(f"Unexpected block size for {source}/{row_group}: {table.num_rows}")
            writer.write_table(table, row_group_size=args.global_batch_size)
        for start in range(0, tail_full_rows, args.global_batch_size):
            rows = [row for _, row in selected_tails[start : start + args.global_batch_size]]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=args.global_batch_size)
        writer.close()

        total_source_rows = sum(source_counts.values())
        selected_rows = len(schedule) * args.global_batch_size + tail_full_rows
        manifest = {
            "source_counts": source_counts,
            "total_source_rows": total_source_rows,
            "global_batch_size": args.global_batch_size,
            "full_source_blocks": block_counts,
            "mixed_tail_blocks": tail_full_rows // args.global_batch_size,
            "tail_selected_by_source": dict(tail_selected_counts),
            "tail_padding_rows": tail_padding,
            "selected_rows": selected_rows,
            "unique_source_rows_covered": total_source_rows,
            "dropped_rows": 0,
            "optimizer_steps": selected_rows // args.global_batch_size,
            "num_frames": args.num_frames,
            "seed": args.seed,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        for path in temporary_root.glob("*.parquet"):
            path.unlink()
        temporary_root.rmdir()


if __name__ == "__main__":
    main()
