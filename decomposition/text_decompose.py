"""
Text decomposition using the propositionizer model from Dense X Retrieval (Chen et al., 2023).

Model: chentong00/propositionizer-wiki-flan-t5-large
  - Flan-T5-Large (~3.1 GB) distilled from GPT-4
  - 96.3% accuracy on query decomposition (MixGR Table 1)
  - 94.7% accuracy on document decomposition (MixGR Table 1)

Input format: "Title: {title}. Section: {section}. Content: {content}"
Output format: JSON list of propositions

Both queries and documents use the same model, following MixGR §3.1.
All results are cached to disk to avoid redundant inference.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TextDecomposer:
    """
    Decompose text into atomic propositions using the propositionizer.

    Usage:
        decomposer = TextDecomposer()
        subqueries = decomposer.decompose_query("What is X and how does it affect Y?")
        propositions = decomposer.decompose_document("Long document text...")
    """

    MODEL_NAME = "chentong00/propositionizer-wiki-flan-t5-large"

    def __init__(
        self,
        model_name: str = None,
        cache_dir: Optional[Path] = None,
        device: str = "cuda",
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name or self.MODEL_NAME
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/decompositions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._tokenizer = None

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    def load_model(self):
        """Load the propositionizer model. Call explicitly or let it lazy-load."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info(f"Loading propositionizer: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")
            logger.info("Model loaded on CUDA")
        else:
            self.device = "cpu"
            logger.info("Model loaded on CPU")

    # --------------------------------------------------------
    # Core inference
    # --------------------------------------------------------

    def _propositionize(self, title: str, section: str, content: str) -> List[str]:
        """
        Run the propositionizer on a single text.
        Input format follows the official usage:
          "Title: {title}. Section: {section}. Content: {content}"
        """
        import torch

        self.load_model()

        input_text = f"Title: {title}. Section: {section}. Content: {content}"

        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse the JSON list output
        propositions = self._parse_proposition_output(output_text)
        if propositions:
            return propositions

        # Fallback: return the raw output as a single proposition
        logger.warning(f"Failed to parse propositionizer output: {output_text[:100]}")
        return [output_text.strip()] if output_text.strip() else [content.strip()]

    def _parse_proposition_output(self, text: str) -> List[str]:
        """Parse propositionizer output, handling truncated JSON."""
        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(p).strip() for p in result if str(p).strip()]
        except json.JSONDecodeError:
            pass

        # Try repairing truncated JSON: output cut off mid-array
        # e.g. '["prop1", "prop2", "prop3' → close the string and array
        if text.startswith("["):
            repaired = text.rstrip()
            # Remove trailing incomplete string
            if repaired.endswith(","):
                repaired = repaired[:-1]
            if not repaired.endswith("]"):
                # Find last complete string (ends with ")
                last_quote = repaired.rfind('"')
                if last_quote > 0:
                    repaired = repaired[:last_quote + 1] + "]"
                else:
                    repaired = repaired + '"]'
            try:
                result = json.loads(repaired)
                if isinstance(result, list):
                    parsed = [str(p).strip() for p in result if str(p).strip()]
                    if parsed:
                        logger.info(f"  Repaired truncated JSON: {len(parsed)} propositions")
                        return parsed
            except json.JSONDecodeError:
                pass

        return []

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def _cache_key(self, text: str, task: str) -> str:
        h = hashlib.md5(f"{task}:{text}".encode()).hexdigest()
        return h

    def _load_cache(self, key: str) -> Optional[List[str]]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, result: List[str]):
        path = self.cache_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def decompose_query(self, text: str, use_cache: bool = True) -> List[str]:
        """
        Decompose a query into subqueries.
        Uses title="" and section="" following MixGR's approach.

        Args:
            text: The query text.

        Returns:
            List of subquery strings.
        """
        key = self._cache_key(text, "subquery")
        if use_cache:
            cached = self._load_cache(key)
            if cached is not None:
                return cached

        subqueries = self._propositionize(title="", section="", content=text)

        if not subqueries:
            subqueries = [text]

        if use_cache:
            self._save_cache(key, subqueries)
        return subqueries

    def decompose_document(
        self, text: str, title: str = "", section: str = "",
        use_cache: bool = True,
    ) -> List[str]:
        """
        Decompose a document into propositions.

        Args:
            text: The document text.
            title: Document title (if available).
            section: Section name (if available).

        Returns:
            List of proposition strings.
        """
        key = self._cache_key(text, "proposition")
        if use_cache:
            cached = self._load_cache(key)
            if cached is not None:
                return cached

        propositions = self._propositionize(title=title, section=section, content=text)

        if not propositions:
            propositions = [text]

        if use_cache:
            self._save_cache(key, propositions)
        return propositions

    def decompose_queries_batch(
        self,
        queries: Dict[str, "Query"],
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Decompose all queries into subqueries.

        Args:
            queries: Dict[qid, Query] from the data loader.

        Returns:
            Dict[qid, List[str]]
        """
        self.load_model()  # Load once before batch
        results = {}
        items = list(queries.items())

        if show_progress:
            try:
                from tqdm import tqdm
                items = tqdm(items, desc="Decomposing queries")
            except ImportError:
                pass

        for qid, query in items:
            results[qid] = self.decompose_query(query.text)

        n_total = sum(len(v) for v in results.values())
        avg = n_total / max(len(results), 1)
        logger.info(f"Decomposed {len(results)} queries → {n_total} subqueries (avg {avg:.1f})")
        return results

    def decompose_corpus_batch(
        self,
        corpus: Dict[str, "Document"],
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Decompose all corpus documents into propositions.

        Args:
            corpus: Dict[did, Document] from the data loader.

        Returns:
            Dict[did, List[str]]
        """
        self.load_model()  # Load once before batch
        results = {}
        items = list(corpus.items())

        if show_progress:
            try:
                from tqdm import tqdm
                items = tqdm(items, desc="Decomposing corpus")
            except ImportError:
                pass

        for did, doc in items:
            results[did] = self.decompose_document(doc.text)

        n_total = sum(len(v) for v in results.values())
        avg = n_total / max(len(results), 1)
        logger.info(f"Decomposed {len(results)} docs → {n_total} propositions (avg {avg:.1f})")
        return results