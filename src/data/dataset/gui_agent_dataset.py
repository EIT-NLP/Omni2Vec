from __future__ import annotations
import ast
import os
import random
from typing import Any
from src.data.dataset.visdoc_utils import stable_id
from src.model.processor import VLM_IMAGE_TOKENS

def _paths(value: Any, image_root: str, max_images: int) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = [value]
    paths = [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]
    if max_images > 0 and len(paths) > max_images:
        if max_images == 1:
            # The last screenshot is the state closest to the requested action.
            paths = [paths[-1]]
        else:
            indices = [
                round(i * (len(paths) - 1) / (max_images - 1))
                for i in range(max_images)
            ]
            paths = [paths[index] for index in indices]
    return [os.path.join(image_root, path) for path in paths]

def _clean_text(value: Any, model_backbone: str) -> str:
    text = str(value or "")
    for token in ("<|image_1|>", "<|image_pad|>", VLM_IMAGE_TOKENS.get(model_backbone, "")):
        if token:
            text = text.replace(token, " " )
    return " ".join(text.split())

def _visual(paths: list[str]):
    return None if not paths else {
        "bytes": [None] * len(paths), "paths": paths, "resolutions": [[128, 128]] * len(paths)
    }

def _convert_example(row, source: str, image_root: str, model_backbone: str, max_images: int):
    retrieval_type = str(row.get("retrieval_type") or "unknown")
    query = _clean_text(row.get("qry_text"), model_backbone)
    positive = _clean_text(row.get("pos_text"), model_backbone)
    query_paths = _paths(row.get("qry_image_path"), image_root, max_images)
    positive_paths = _paths(row.get("pos_image_path"), image_root, max_images)
    if not query or (not positive and not positive_paths):
        raise ValueError(f"Invalid GUI row: source={source}, pos_id={row.get('pos_id')}")
    positive_identity = str(row.get("pos_id") or row.get("pos_image_path") or positive)
    group = f"gui/{source}:{retrieval_type}:{positive_identity}"
    return {
        "query_text": query,
        "query_image": _visual(query_paths),
        "pos_text": positive,
        "pos_image": _visual(positive_paths),
        "neg_text": "", "neg_image": None,
        "positive_group_id": group,
        "sampling_group_id": f"gui/{source}:{retrieval_type}:{positive_identity.split('_traj-')[0]}",
        "pair_id": stable_id(group),
        "task_id": stable_id(f"task:gui/{retrieval_type}"),
        "global_dataset_name": f"gui/{source}/{retrieval_type}",
        "multi_image_as_images": True,
    }

def _buffer_shuffle(rows, buffer_size: int, seed: int):
    if buffer_size <= 0:
        yield from rows
        return
    rng = random.Random(seed)
    buffer = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= buffer_size:
            yield buffer.pop(rng.randrange(len(buffer)))
    rng.shuffle(buffer)
    yield from buffer
