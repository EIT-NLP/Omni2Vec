from __future__ import annotations
import zipfile
from typing import Any
from src.data.dataset.visdoc_utils import stable_id
from src.model.processor import VLM_IMAGE_TOKENS

def _clean_text(value: Any, model_backbone: str) -> str:
    text = str(value or "")
    for token in ("<|image_1|>", "<|image_pad|>", VLM_IMAGE_TOKENS.get(model_backbone, "")):
        if token:
            text = text.replace(token, " " )
    return " ".join(text.split())

def _clean_path(value: Any) -> str:
    value = str(value or "").strip()
    return "" if value.lower() == "null" else value

def _zip_member(image_path: str) -> str:
    return image_path[len("images/"):] if image_path.startswith("images/") else image_path

def _read_image(archive: zipfile.ZipFile, image_path: str, resolution: list[int]):
    if not image_path:
        return None
    member = _zip_member(image_path)
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise KeyError(f"Missing MMEB image in {archive.filename}: {member}") from exc
    return {"bytes": [payload], "paths": [""], "resolutions": [resolution]}

def _positive_identity(task_name: str, row: dict[str, Any], model_backbone: str) -> str:
    positive_image = _clean_path(row.get("pos_image_path"))
    if positive_image:
        return f"image:{positive_image}"
    return f"text:{_clean_text(row.get('pos_text'), model_backbone)}"

def _negative_identity(task_name: str, row: dict[str, Any], model_backbone: str) -> str | None:
    negative_image = _clean_path(row.get("neg_image_path"))
    if negative_image:
        return f"image/{task_name}:image:{negative_image}"
    negative_text = _clean_text(row.get("neg_text"), model_backbone)
    if negative_text:
        return f"image/{task_name}:text:{negative_text}"
    return None

def _convert_example(
    row, archive, task_name, model_backbone, resolution,
):
    query_path = _clean_path(row.get("qry_image_path"))
    positive_path = _clean_path(row.get("pos_image_path"))
    negative_path = _clean_path(row.get("neg_image_path"))
    group = f"image/{task_name}:{_positive_identity(task_name, row, model_backbone)}"
    example = {
        "query_text": _clean_text(row.get("qry"), model_backbone),
        "query_image": _read_image(archive, query_path, resolution),
        "pos_text": _clean_text(row.get("pos_text"), model_backbone),
        "pos_image": _read_image(archive, positive_path, resolution),
        "neg_text": _clean_text(row.get("neg_text"), model_backbone),
        "neg_image": _read_image(archive, negative_path, resolution),
        "positive_group_id": group,
        "sampling_group_id": group,
        "pair_id": stable_id(group),
        "task_id": stable_id(f"task:image/{task_name}"),
        "global_dataset_name": f"image/{task_name}",
    }
    negative_group = _negative_identity(task_name, row, model_backbone)
    if negative_group is not None:
        example["negative_pair_id"] = stable_id(negative_group)
    return example
