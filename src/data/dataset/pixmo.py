import random
from collections import deque
from PIL import Image
from datasets.features.image import image_to_bytes
from src.data.dataset.base_pair_dataset import RESOLUTION_MAPPING
from src.data.dataset.visdoc_utils import attach_identity
from src.model.processor import VLM_IMAGE_TOKENS

QUERY_INSTRUCTION = "Retrieve documents that best answer the following query: "

TARGET_INSTRUCTION = "Generate an embedding for the following document, in the format of images: "

def _examples(rows, model_backbone, image_resolution, source, page_buffer_size=128, seed=42):
    resolution = list(RESOLUTION_MAPPING.get(image_resolution, RESOLUTION_MAPPING["low"]))
    iterator = iter(rows)
    rng = random.Random(seed)
    while True:
        pages = []
        for _ in range(page_buffer_size):
            try:
                row = next(iterator)
            except StopIteration:
                break
            image = row.get("image")
            if isinstance(image, Image.Image):
                payload, path = image_to_bytes(image), ""
            elif isinstance(image, dict):
                payload, path = image.get("bytes"), image.get("path") or ""
            else:
                payload, path = None, ""
            image_id = str(row.get("image_id") or path or "")
            questions = deque((row.get("questions") or {}).get("question") or [])
            if image_id and questions and (payload is not None or path):
                pages.append((image_id, payload, path, questions))
        if not pages:
            return
        rng.shuffle(pages)
        # Emit one question per page per round. This keeps the first 64 examples
        # page-unique while buffering only a small number of image payloads.
        while pages:
            remaining = []
            for image_id, payload, path, questions in pages:
                question = questions.popleft()
                example = {
                    "query_text": QUERY_INSTRUCTION + str(question),
                    "query_image": None,
                    "pos_text": TARGET_INSTRUCTION + VLM_IMAGE_TOKENS[model_backbone],
                    "pos_image": {"bytes": [payload], "paths": [path], "resolutions": [resolution]},
                    "neg_text": "", "neg_image": None,
                }
                yield attach_identity(example, source, image_id)
                if questions:
                    remaining.append((image_id, payload, path, questions))
            pages = remaining
