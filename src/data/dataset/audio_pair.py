from __future__ import annotations
import hashlib
import os
from typing import Any, Dict, List, Optional
import datasets
from src.data.dataset.base_pair_dataset import RESOLUTION_MAPPING
from src.data.eval_dataset.audio_instruction_utils import AUDIO_CORPUS_PROMPT, build_query_text
from src.model.processor import process_input_text

_DATASET_NAMES = {
    "audiocaps": "AudioCaps",
    "ave": "AVE",
    "clotho": "Clotho",
    "speechcoco": "SpeechCOCO",
    "fsd50k": "ESC-50",
    "nsynth": "NSynth",
    "speechcommand": "SpeechCommands",
    "urbansound8k": "UrbanSound8K",
}

OUTPUT_FEATURES = datasets.Features({
    "query_text": datasets.Sequence(datasets.Value("string")),
    "query_image": {
        "paths": datasets.Sequence(datasets.Value("string")),
        "bytes": datasets.Sequence(datasets.Value("binary")),
        "resolutions": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
    },
    "query_audio": {
        "path": datasets.Value("string"), "bytes": datasets.Value("binary"),
        "start": datasets.Value("float32"), "end": datasets.Value("float32"),
    },
    "pos_text": datasets.Sequence(datasets.Value("string")),
    "pos_image": {
        "paths": datasets.Sequence(datasets.Value("string")),
        "bytes": datasets.Sequence(datasets.Value("binary")),
        "resolutions": datasets.Sequence(datasets.Sequence(datasets.Value("int32"), length=2)),
    },
    "pos_audio": {
        "path": datasets.Value("string"), "bytes": datasets.Value("binary"),
        "start": datasets.Value("float32"), "end": datasets.Value("float32"),
    },
    "dataset_infos": {
        "uid": datasets.Value("string"),
        "task_type": datasets.Value("string"),
        "positive_group_id": datasets.Value("string"),
    },
    "positive_group_id": datasets.Value("string"),
    "pair_id": datasets.Value("int64"),
    "task_id": datasets.Value("int64"),
    "global_dataset_name": datasets.Value("string"),
})

def stable_pair_id(value: str) -> int:
    """Return a deterministic non-negative int64 identity."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)

def _first(values, default=None):
    if isinstance(values, list) and values:
        return values[0]
    return default

def _assets_by_id(row: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    return {str(x.get("asset_id")): x for x in (row.get(key) or []) if isinstance(x, dict)}

def _audio_value(asset: Optional[Dict[str, Any]], ref: Optional[Dict[str, Any]], data_path: str):
    if not asset:
        return None
    path = asset.get("path")
    payload = asset.get("bytes")
    if path and not os.path.isabs(path):
        path = os.path.join(data_path, path)
    if payload is not None:
        path = None
    return {
        "path": path,
        "bytes": payload,
        "start": (ref or {}).get("start"),
        "end": (ref or {}).get("end"),
    }

def _image_value(assets: List[Dict[str, Any]], data_path: str, resolution):
    if not assets:
        return None
    paths, payloads, resolutions = [], [], []
    for asset in assets:
        path = asset.get("path")
        payload = asset.get("bytes")
        if path and not os.path.isabs(path):
            path = os.path.join(data_path, path)
        if payload is not None:
            path = None
        paths.append(path)
        payloads.append(payload)
        width, height = asset.get("width"), asset.get("height")
        resolutions.append([int(width), int(height)] if width and height else list(resolution))
    return {"paths": paths, "bytes": payloads, "resolutions": resolutions}

def _positive_group(row: Dict[str, Any], dataname: str, task_type: str) -> str:
    if task_type == "audio_text_classification":
        label = str(_first(row.get("pos_text_candidates") or [], ""))
        return f"{dataname}:label:{label}"
    if task_type in {"audio_image_retrieval", "audio_video_retrieval"}:
        ids = row.get("pos_image_asset_ids") or []
        if ids:
            # AVE frame ids share the video id before the final frame suffix.
            value = str(ids[0])
            if dataname.lower() == "ave":
                value = value.rsplit(":", 1)[0]
            return f"{dataname}:image:{value}"
    pos_audio = row.get("pos_audio") or {}
    if pos_audio.get("asset_id"):
        return f"{dataname}:audio:{pos_audio['asset_id']}"
    return f"{dataname}:row:{row.get('uid', row.get('metadata_json', 'unknown'))}"

def _convert_row(row: Dict[str, Any], data_path: str, model_backbone: str, image_resolution: str):
    dataname = str(row.get("dataname") or "audio")
    canonical_name = _DATASET_NAMES.get(dataname.lower(), dataname)
    task_type = str(row.get("task_type") or "")
    audio_assets = _assets_by_id(row, "audio_assets")
    image_assets = _assets_by_id(row, "image_assets")
    query_audio_ref = row.get("query_audio") or {}
    pos_audio_ref = row.get("pos_audio") or {}
    query_audio = _audio_value(audio_assets.get(str(query_audio_ref.get("asset_id"))), query_audio_ref, data_path)
    pos_audio = _audio_value(audio_assets.get(str(pos_audio_ref.get("asset_id"))), pos_audio_ref, data_path)

    query_candidates = [str(x) for x in (row.get("query_text_candidates") or []) if str(x).strip()]
    pos_candidates = [str(x) for x in (row.get("pos_text_candidates") or []) if str(x).strip()]
    query_text = [
        build_query_text(canonical_name, candidate)[0]
        for candidate in query_candidates
    ] or build_query_text(canonical_name)
    pos_text: Any = pos_candidates or [" " ]
    query_image = None
    pos_image = None

    if task_type == "text_audio_retrieval":
        pos_text = AUDIO_CORPUS_PROMPT
    elif task_type in {"audio_image_retrieval", "audio_video_retrieval"}:
        query_text = build_query_text(canonical_name)
        refs = [image_assets.get(str(x)) for x in (row.get("pos_image_asset_ids") or [])]
        refs = [x for x in refs if x]
        resolution = RESOLUTION_MAPPING.get(image_resolution, RESOLUTION_MAPPING["low"])
        pos_image = _image_value(refs, data_path, resolution)
        pos_text = process_input_text(
            "Understand the content of the provided video." if canonical_name == "AVE" else "Understand the content of the provided image.",
            model_backbone,
            add_video_token=(canonical_name == "AVE"),
            add_image_token=(canonical_name != "AVE"),
        )
    elif task_type == "audio_text_classification":
        query_text = build_query_text(canonical_name)
    else:
        raise ValueError(f"Unsupported audio task_type={task_type!r}")

    group = _positive_group(row, dataname, task_type)
    return {
        "query_text": query_text,
        "query_image": query_image,
        "query_audio": query_audio,
        "pos_text": [pos_text] if isinstance(pos_text, str) else pos_text,
        "pos_image": pos_image,
        "pos_audio": pos_audio,
        "dataset_infos": {"uid": str(row.get("uid") or ""), "task_type": task_type, "positive_group_id": group},
        "positive_group_id": group,
        "pair_id": stable_pair_id(group),
        "task_id": stable_pair_id(f"task:{dataname.lower()}"),
        "global_dataset_name": dataname.lower(),
    }
