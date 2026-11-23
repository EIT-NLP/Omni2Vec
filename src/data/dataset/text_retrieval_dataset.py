from __future__ import annotations
import random
from typing import Any
from datasets import Features, Value
from src.data.dataset.visdoc_utils import VISDOC_FEATURES, stable_id

TEXT_FEATURES = Features({**dict(VISDOC_FEATURES), "negative_pair_id": Value("int64")})

def _text(value: Any) -> str:
    return " ".join(str(value or "").split())

def _passage(value: Any) -> str:
    if isinstance(value, dict):
        title = _text(value.get("title"))
        body = _text(value.get("text"))
        return "\n".join(part for part in (title, body) if part)
    return _text(value)

def _choose(values: Any, rng: random.Random) -> str:
    if not isinstance(values, list):
        values = [values] if values else []
    choices = [_passage(value) for value in values]
    choices = [value for value in choices if value]
    return rng.choice(choices) if choices else ""

def _query(query: Any, instruction: Any = None) -> str:
    query = _text(query)
    instruction = _text(instruction)
    if instruction:
        return f"{instruction}\n{query}"
    return query

def _convert_bge(row: dict[str, Any], rng: random.Random, hard_negative_source: str) -> dict[str, str]:
    positive = _choose(row.get("pos"), rng)
    negative = _choose(row.get("neg"), rng) if hard_negative_source != "none" else ""
    return {"query": _query(row.get("query")), "positive": positive, "negative": negative}

def _convert_inf_ir(row: dict[str, Any], rng: random.Random, hard_negative_source: str) -> dict[str, str]:
    negatives = [row.get("hard_negative_document_1"), row.get("hard_negative_document_2")]
    if hard_negative_source == "inf_1":
        negative = _passage(negatives[0])
    elif hard_negative_source == "inf_2":
        negative = _passage(negatives[1])
    elif hard_negative_source in {"inf_random", "any"}:
        negative = _choose(negatives, rng)
    else:
        negative = ""
    return {
        "query": _query(row.get("query_positive"), row.get("instruction_positive")),
        "positive": _passage(row.get("document_positive")),
        "negative": negative,
    }

def _convert_promptriever(
    row: dict[str, Any], rng: random.Random, hard_negative_source: str
) -> dict[str, str]:
    instruction = row.get("only_instruction") if row.get("has_instruction") else ""
    query = row.get("only_query") or row.get("query")
    if hard_negative_source == "instruction":
        negative = _choose(row.get("new_negatives"), rng)
    elif hard_negative_source == "standard":
        negative = _choose(row.get("negative_passages"), rng)
    elif hard_negative_source in {"instruction_first", "any"}:
        negative = _choose(row.get("new_negatives"), rng) or _choose(row.get("negative_passages"), rng)
    else:
        negative = ""
    return {
        "query": _query(query, instruction),
        "positive": _choose(row.get("positive_passages"), rng),
        "negative": negative,
    }

def _to_example(converted: dict[str, str], source: str, require_hard_negative: bool):
    query, positive, negative = (converted[key] for key in ("query", "positive", "negative"))
    if not query or not positive or (require_hard_negative and not negative):
        return None
    positive_key = stable_id(positive)
    negative_key = stable_id(negative) if negative else 0
    if negative and negative_key == positive_key:
        return None
    return {
        "query_text": query,
        "query_image": None,
        "pos_text": positive,
        "pos_image": None,
        "neg_text": negative,
        "neg_image": None,
        "positive_group_id": f"text:passage:{positive_key}",
        "sampling_group_id": f"text:{source}:query:{stable_id(query)}",
        "pair_id": positive_key,
        "negative_pair_id": negative_key,
        "task_id": stable_id("task:text_retrieval"),
        "global_dataset_name": f"text/{source}",
    }
