"""Load MRMR Knowledge in the same way as the official task implementation."""

from __future__ import annotations

import io
import logging
from typing import Dict, Iterator, List, Optional

from datasets import load_dataset
from PIL import Image

from config import cfg

logger = logging.getLogger(__name__)

HF_DATASET_ID = cfg.data.hf_dataset_id
DEFAULT_QUERY_INSTRUCTION = cfg.data.query_instruction

DOMAIN_PREFIXES = {
    "Science": ["agriculture", "geography", "chemistry", "biology"],
    "Medicine": ["diagnostics", "clinical_medicine", "basic_medical", "pharmacy"],
    "Art": ["music", "design", "art_theory", "art"],
    "Humanities": ["history", "sociology", "psychology", "literature"],
}

CATEGORY_MAP = {
    "Art": "Art",
    "Art_Theory": "Art",
    "Design": "Art",
    "Music": "Art",
    "Sociology": "Humanities",
    "Literature": "Humanities",
    "History": "Humanities",
    "Psychology": "Humanities",
    "Clinical_Medicine": "Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Medicine",
    "Basic_Medical_Science": "Medicine",
    "Pharmacy": "Medicine",
    "Biology": "Science",
    "Chemistry": "Science",
    "Geography": "Science",
    "Agriculture": "Science",
}


def resize_image(image: Image.Image | None, max_size: int = 2000) -> Image.Image | None:
    """Official MRMR resize rule."""
    if image is None:
        return None
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    scale_factor = max_size / width if width > height else max_size / height
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _ensure_pil_image(value) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    return None


def coarse_domain_from_category(category: str | None) -> str:
    if not category:
        return "Unknown"
    return CATEGORY_MAP.get(category, "Unknown")


def available_domains() -> List[str]:
    return list(DOMAIN_PREFIXES.keys())


def _query_to_item(row: dict, split: str) -> dict:
    return {
        "id": f"query-{split}-{row['id']}",
        "source_id": str(row["id"]),
        "text": row.get("text"),
        "image": resize_image(_ensure_pil_image(row.get("image"))),
        "modality": row.get("modality", "image,text"),
        "category": row.get("category"),
        "coarse_domain": coarse_domain_from_category(row.get("category")),
        "instruction": DEFAULT_QUERY_INSTRUCTION,
    }


def _corpus_to_item(row: dict, split: str) -> dict:
    return {
        "id": f"corpus-{split}-{row['id']}",
        "source_id": str(row["id"]),
        "text": row.get("text"),
        "image": resize_image(_ensure_pil_image(row.get("image"))),
        "modality": row.get("modality", "image,text"),
    }


def load_queries(
    domains: Optional[List[str]] = None,
    split: str = "test",
    max_queries: int | None = None,
) -> List[dict]:
    """Load query items exactly like the official MRMR task mapping."""
    logger.info("Loading queries from %s (%s)...", HF_DATASET_ID, split)
    ds = load_dataset(HF_DATASET_ID, "query", split=split)
    wanted = set(domains or available_domains())

    queries: List[dict] = []
    for row in ds:
        item = _query_to_item(row, split)
        if item["coarse_domain"] not in wanted:
            continue
        queries.append(item)
        if max_queries is not None and len(queries) >= max_queries:
            break

    logger.info("Loaded %d queries", len(queries))
    return queries


def load_qrels(query_ids: set[str], split: str = "test") -> Dict[str, Dict[str, int]]:
    """Load qrels and prefix ids exactly like the official task."""
    logger.info("Loading qrels from %s (%s)...", HF_DATASET_ID, split)
    ds = load_dataset(HF_DATASET_ID, "qrels", split=split)
    qrels: Dict[str, Dict[str, int]] = {}

    for row in ds:
        qid = f"query-{split}-{row['query_id']}"
        if qid not in query_ids:
            continue
        did = f"corpus-{split}-{row['corpus_id']}"
        qrels.setdefault(qid, {})[did] = int(row.get("score", 1))

    logger.info("Loaded qrels for %d queries", len(qrels))
    return qrels


def iter_corpus(split: str = "test", max_docs: int | None = None) -> Iterator[dict]:
    """Yield corpus items one by one to avoid keeping all PIL images in memory."""
    logger.info("Streaming corpus from %s (%s)...", HF_DATASET_ID, split)
    ds = load_dataset(HF_DATASET_ID, "corpus", split=split)
    total = len(ds) if max_docs is None else min(len(ds), max_docs)
    for idx in range(total):
        yield _corpus_to_item(ds[idx], split)


def count_available_corpus(split: str = "test") -> int:
    ds = load_dataset(HF_DATASET_ID, "corpus", split=split)
    return len(ds)
