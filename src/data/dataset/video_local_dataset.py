from __future__ import annotations
import glob
import io
import json
import os
import random
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterator
import pyarrow.parquet as pq
from PIL import Image
from src.data.dataset.visdoc_utils import stable_id

VIDEO_RETRIEVAL_PROMPT = "Find a video that contains the following visual content:"

VIDEO_CORPUS_PROMPT = "Understand the content of the provided video."

VIDEO_QA_PROMPT = "Answer the question based on the content of the video:"

VIDEO_CAPTION_PROMPT = "Describe the content of the video in detail:"

SSV2_PROMPT = "What action or object interaction is being performed in the video?"

VIDEO_REF_PREFIX = "local-video://"

def _natural_key(path: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]

def _clean_text(value: Any) -> str:
    text = str(value or "")
    for token in ("<video>", "<image>", "<|video_pad|>", "<|image_pad|>"):
        text = text.replace(token, " ")
    return " ".join(text.split())

def _visual(frame_bytes: list[bytes]):
    return {
        "bytes": frame_bytes, "paths": [""] * len(frame_bytes),
        "resolutions": [[224, 224]] * len(frame_bytes),
    }

def _video_reference(path, num_frames=8, row_group=None, row=None, video_id=None):
    reference = json.dumps({"path": path, "num_frames": num_frames,
                            "row_group": row_group, "row": row, "video_id": video_id})
    return {"bytes": [None], "paths": [VIDEO_REF_PREFIX + reference], "resolutions": [[0, 0]]}

def _sample_indices(count, num_frames):
    if count < 1 or num_frames < 1:
        raise ValueError("Video frame count and requested frame count must be positive")
    if count < num_frames:
        return list(range(count)) + [count - 1] * (num_frames - count)
    if num_frames == 1:
        return [0]
    return [round(i * (count - 1) / (num_frames - 1)) for i in range(num_frames)]

def _decode_video(path: str, num_frames: int = 8) -> list[bytes]:
    from decord import VideoReader, cpu

    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    frames = reader.get_batch(_sample_indices(len(reader), num_frames)).asnumpy()
    payloads = []
    for frame in frames:
        output = io.BytesIO()
        Image.fromarray(frame).convert("RGB").save(output, format="JPEG", quality=90)
        payloads.append(output.getvalue())
    return payloads

@lru_cache(maxsize=1)
def _frame_group(path, row_group):
    with pq.ParquetFile(path) as parquet:
        return parquet.read_row_group(row_group, columns=["video_id", "frame_bytes"]).to_pylist()

def resolve_video_reference(visual):
    path = (visual.get("paths") or [None])[0]
    if not isinstance(path, str) or not path.startswith(VIDEO_REF_PREFIX):
        return visual
    reference = json.loads(path[len(VIDEO_REF_PREFIX):])
    if reference["path"].endswith(".lance"):
        import lance
        row = lance.dataset(reference["path"]).take([reference["row"]], columns=["id", "images"]).to_pylist()[0]
        if str(row["id"]) != reference["video_id"]:
            raise ValueError(f"Lance video identity mismatch: {reference}")
        frames = row["images"]
        if not frames or any(not frame for frame in frames):
            raise ValueError(f"Empty Lance video: {reference}")
        frames = [frames[i] for i in _sample_indices(len(frames), reference["num_frames"])]
    elif reference["row_group"] is None:
        frames = _decode_video(reference["path"], reference["num_frames"])
    else:
        row = _frame_group(reference["path"], reference["row_group"])[reference["row"]]
        if str(row["video_id"]) != reference["video_id"]:
            raise ValueError(f"Video metadata/frame identity mismatch: {reference}")
        frames = row["frame_bytes"]
        if not frames or any(not frame for frame in frames):
            raise ValueError(f"Empty video frames: {reference}")
        frames = [frames[i] for i in _sample_indices(len(frames), reference["num_frames"])]
    return _visual(frames)

def _example(query_text, query_frames, positive_text, positive_frames, source, mode, positive_identity):
    group = f"video/{source}/{mode}:{positive_identity}"
    return {
        "query_text": query_text,
        "query_image": query_frames if isinstance(query_frames, dict) else (_visual(query_frames) if query_frames else None),
        "pos_text": positive_text,
        "pos_image": positive_frames if isinstance(positive_frames, dict) else (_visual(positive_frames) if positive_frames else None),
        "neg_text": "", "neg_image": None,
        "positive_group_id": group, "sampling_group_id": f"video/{source}:{positive_identity}",
        "pair_id": stable_id(group), "task_id": stable_id(f"task:video/{source}/{mode}"),
        "global_dataset_name": f"video/{source}/{mode}", "source_mode": mode,
    }

def _retrieval_rows(dataset_root, source, annotation_file, video_subdir):
    with open(os.path.join(dataset_root, source, annotation_file), encoding="utf-8") as handle:
        rows = json.load(handle)
    video_root = os.path.join(dataset_root, source, video_subdir)
    available = set(os.listdir(video_root))
    result = []
    for row in rows:
        video_name = os.path.basename(str(row["video"]))
        video_id = str(row.get("video_id") or os.path.splitext(video_name)[0])
        stem = os.path.splitext(video_name)[0]
        candidates = [video_name] + [stem + ext for ext in (".mp4", ".avi", ".webm", ".mkv")]
        name = next((name for name in candidates if name in available), None)
        if name is None:
            raise FileNotFoundError(f"Missing {source} video {video_name} under {video_root}")
        captions = row.get("caption")
        captions = captions if isinstance(captions, list) else [captions]
        captions = [_clean_text(caption) for caption in captions if _clean_text(caption)]
        if not captions:
            raise ValueError(f"Missing caption for {source}/{video_id}")
        result.append((video_id, os.path.join(video_root, name), captions))
    return result

def _json_retrieval_examples(rows, source, num_frames, seed):
    rng = random.Random(seed)
    for video_id, video_path, captions in rows:
        caption = rng.choice(captions)
        yield _example(f"{VIDEO_RETRIEVAL_PROMPT} {caption}", [], VIDEO_CORPUS_PROMPT,
                       _video_reference(video_path, num_frames), source, "retrieval", video_id)

def _iter_parquet_rows(files: list[str], num_frames=8, with_text=False) -> Iterator[dict[str, Any]]:
    if not files:
        raise FileNotFoundError("No video parquet files found")
    columns = ["video_id", "pos_text"] if with_text else ["video_id"]
    for path in sorted(files, key=_natural_key):
        with pq.ParquetFile(path) as parquet:
            for group in range(parquet.num_row_groups):
                for index, row in enumerate(parquet.read_row_group(group, columns=columns).to_pylist()):
                    row["visual"] = _video_reference(path, num_frames, group, index, str(row["video_id"]))
                    yield row

@lru_cache(maxsize=1)
def _load_qa_map(path: str):
    result = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            conversations = row["conversations"]
            question, answer = _clean_text(conversations[0]["value"]), _clean_text(conversations[1]["value"])
            if not question or not answer:
                raise ValueError(f"Empty QA annotation for {row['video']}")
            result[str(row["video"])].append((question, answer))
    return result

def _llava_examples(dataset_root: str, mode: str, num_frames=8):
    root = os.path.join(dataset_root, "llava_hound")
    files = glob.glob(os.path.join(root, "parquet_8frames", "*.parquet"))
    annotation_root = os.path.join(root, "video_instruction", "train", "sft")
    if mode == "caption":
        frame_index = {}
        for frame_row in _iter_parquet_rows(files, num_frames):
            video_id = str(frame_row["video_id"])
            if video_id in frame_index:
                raise ValueError(f"Duplicate LLaVA video: {video_id}")
            frame_index[video_id] = frame_row["visual"]
        with open(os.path.join(annotation_root, "video_caption_300k.jsonl"), encoding="utf-8") as annotations:
            for line in annotations:
                annotation = json.loads(line)
                video_id = str(annotation["video"])
                if video_id not in frame_index:
                    raise ValueError(f"Missing or duplicate LLaVA caption video: {video_id}")
                answer = _clean_text(annotation["conversations"][1]["value"])
                if not answer:
                    raise ValueError(f"Empty caption for {video_id}")
                yield _example(VIDEO_CAPTION_PROMPT, frame_index.pop(video_id), answer, [], "LLaVA-Hound", mode, answer)
        if frame_index:
            raise ValueError("LLaVA caption/frame counts differ")
    elif mode == "qa":
        qa_map = _load_qa_map(os.path.join(annotation_root, "video_qa_240k.jsonl"))
        missing = set(qa_map)
        for frame_row in _iter_parquet_rows(files, num_frames):
            video_id = str(frame_row["video_id"])
            missing.discard(video_id)
            for question, answer in qa_map.get(video_id, []):
                yield _example(f"{VIDEO_QA_PROMPT} {question}", frame_row["visual"], answer, [], "LLaVA-Hound", mode, answer)
        if missing:
            raise ValueError(f"Missing frames for {len(missing)} LLaVA QA videos")
    else:
        raise ValueError(f"Unknown LLaVA mode: {mode}")

def _ssv2_examples(dataset_root: str, num_frames=8):
    files = glob.glob(os.path.join(dataset_root, "SmthSmthV2", "parquet_8frames", "*.parquet"))
    for row in _iter_parquet_rows(files, num_frames, with_text=True):
        label = _clean_text(row["pos_text"])
        if not label:
            raise ValueError(f"Empty SSv2 target for {row['video_id']}")
        yield _example(SSV2_PROMPT, row["visual"], label, [], "SmthSmthV2", "classification", label)
