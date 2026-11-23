import re
from PIL import Image
from datasets.features.image import image_to_bytes
from src.data.dataset.base_pair_dataset import RESOLUTION_MAPPING
from src.data.dataset.visdoc_utils import attach_identity
from src.model.processor import VLM_IMAGE_TOKENS

QUERY_INSTRUCTION = "Retrieve documents that best answer the following query: "

TARGET_INSTRUCTION = "Generate an embedding for the following document, in the format of images: "

def _image_parts(value):
    if isinstance(value, Image.Image):
        return image_to_bytes(value), ""
    if isinstance(value, dict):
        return value.get("bytes"), value.get("path") or ""
    return None, ""

def _document_identity(page_identity):
    value = str(page_identity)
    document = re.sub(r"[/\\]page[_-]?\d+\.(?:jpg|jpeg|png)$", "", value, flags=re.I)
    if document == value:
        document = re.sub(r"[_-]\d+\.(?:jpg|jpeg|png)$", "", value, flags=re.I)
    return document

def _examples(rows, model_backbone, image_resolution, source):
    resolution = list(RESOLUTION_MAPPING.get(image_resolution, RESOLUTION_MAPPING["low"]))
    for row in rows:
        payload, path = _image_parts(row.get("image"))
        identity = str(row.get("image_filename") or path or "")
        if not identity or (payload is None and not path):
            continue
        example = {
            "query_text": QUERY_INSTRUCTION + str(row.get("query") or ""),
            "query_image": None,
            "pos_text": TARGET_INSTRUCTION + VLM_IMAGE_TOKENS[model_backbone],
            "pos_image": {"bytes": [payload], "paths": [path], "resolutions": [resolution]},
            "neg_text": "", "neg_image": None,
        }
        yield attach_identity(example, source, identity, _document_identity(identity))
