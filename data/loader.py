"""
Data loader for MRMR Knowledge dataset.

Dataset: MRMRbenchmark/knowledge (HuggingFace)
3 subsets, each with 'test' split:
  - corpus (26,223 rows): id, modality, text, image, image 1..4, vision
  - query  (555 rows):    id, modality, text, image, image 1..2, domain, answer, instruction
  - qrels  (1,010 rows):  query_id, corpus_id, score

Usage:
    loader = MRMRKnowledgeLoader(domains=["Science", "Medicine"])
    queries, corpus, qrels = loader.load()
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """A single multimodal query."""
    qid: str
    text: str
    images: List[Image.Image] = field(default_factory=list)
    domain: str = ""
    answer: Optional[str] = None
    instruction: str = "Retrieve relevant documents that help answer the question."


@dataclass
class Document:
    """A single corpus document (may contain images)."""
    did: str
    text: str
    images: List[Image.Image] = field(default_factory=list)
    modality: str = "text"  # "text", "image", "image,text"


class MRMRKnowledgeLoader:
    """
    Load MRMR Knowledge dataset.

    HuggingFace repo: MRMRbenchmark/knowledge
    Subsets: "corpus", "query", "qrels"
    """

    # Map domain names used in MRMR query IDs to our filter names
    DOMAIN_PREFIXES = {
        "Science": [
            "Agriculture", "Geography", "Chemistry", "Biology",
        ],
        "Medicine": [
            "Diagnostics", "Clinical_Medicine", "Basic_Medical",
            "Pharmacy", "Diagnostics_and_Laboratory",
        ],
        "Art": [
            "Music", "Design", "Art_Theory", "Art",
        ],
        "Humanities": [
            "History", "Sociology", "Psychology", "Literature",
        ],
    }

    # def __init__(
    #     self,
    #     domains: List[str] = None,
    #     hf_dataset_id: str = "MRMRbenchmark/knowledge",
    #     cache_dir: Optional[Path] = None,
    # ):
    #     self.domains = domains or ["Science", "Medicine"]
    #     self.hf_dataset_id = hf_dataset_id
    #     self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/mrmr_raw")
    #     self.cache_dir.mkdir(parents=True, exist_ok=True)

    #     # Build set of domain prefixes to filter on
    #     self._domain_prefixes = set()
    #     for d in self.domains:
    #         if d in self.DOMAIN_PREFIXES:
    #             self._domain_prefixes.update(self.DOMAIN_PREFIXES[d])
    #         else:
    #             self._domain_prefixes.add(d)
    def __init__(
        self,
        domains: List[str] = None,
        hf_dataset_id: str = "MRMRbenchmark/knowledge",
        cache_dir: Optional[Path] = None,
        local_dataset_path: Optional[Path] = None,
        force_local: bool = False,
    ):
        self.domains = domains or ["Science", "Medicine"]
        self.hf_dataset_id = hf_dataset_id
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/mrmr_raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.local_dataset_path = Path(local_dataset_path) if local_dataset_path else None
        self.force_local = force_local

        # Build set of domain prefixes to filter on
        self._domain_prefixes = set()
        for d in self.domains:
            if d in self.DOMAIN_PREFIXES:
                self._domain_prefixes.update(self.DOMAIN_PREFIXES[d])
            else:
                self._domain_prefixes.add(d)

    # def load(self):
    #     """
    #     Returns:
    #         queries: Dict[str, Query]
    #         corpus:  Dict[str, Document]
    #         qrels:   Dict[str, Dict[str, int]]
    #     """
    #     logger.info(f"Loading MRMR Knowledge for domains: {self.domains}")

    #     try:
    #         return self._load_from_huggingface()
    #     except Exception as e:
    #         logger.warning(f"HuggingFace loading failed: {e}")
    #         logger.info("Trying local cache...")
    #         return self._load_from_local()
    def load(self):
        logger.info(f"Loading MRMR Knowledge for domains: {self.domains}")

        # 优先走本地 HuggingFace save_to_disk / load_from_disk 数据
        if self.local_dataset_path is not None:
            logger.info(f"Trying local dataset path: {self.local_dataset_path}")
            return self._load_from_disk_dataset()

        # 如果明确要求只走本地 json fallback
        if self.force_local:
            logger.info("force_local=True, trying local json cache only...")
            return self._load_from_local()

        # 否则再尝试联网
        try:
            return self._load_from_huggingface()
        except Exception as e:
            logger.warning(f"HuggingFace loading failed: {e}")
            logger.info("Trying local json cache...")
            return self._load_from_local()
    def _load_from_disk_dataset(self):
        """
        从本地离线 HuggingFace 数据集目录加载。
        要求 local_dataset_path 指向一个已经 save_to_disk() 的目录，
        或者一个包含 corpus/query/qrels 三个子目录的目录。
        """
        from datasets import load_from_disk

        root = self.local_dataset_path
        if root is None:
            raise ValueError("local_dataset_path is None")

        logger.info(f"Loading local dataset from disk: {root}")

        # 兼容两种目录组织：
        # 1. root/corpus, root/query, root/qrels 分别是 save_to_disk 结果
        # 2. root 就是一个总目录（这种情况通常不适用于你当前三子集结构）
        corpus_path = root / "corpus"
        query_path = root / "query"
        qrels_path = root / "qrels"

        if corpus_path.exists() and query_path.exists() and qrels_path.exists():
            ds_qrels = load_from_disk(str(qrels_path))
            ds_queries = load_from_disk(str(query_path))
            ds_corpus = load_from_disk(str(corpus_path))
            from datasets import DatasetDict
            def unwrap_split(ds, name):
                if isinstance(ds, DatasetDict):
                    if "test" in ds:
                        return ds["test"]
                    first_key = list(ds.keys())[0]
                    logger.warning(f"{name} is DatasetDict, using split: {first_key}")
                    return ds[first_key]
                return ds

            ds_qrels = unwrap_split(ds_qrels, "qrels")
            ds_queries = unwrap_split(ds_queries, "query")
            ds_corpus = unwrap_split(ds_corpus, "corpus")
        else:
            raise FileNotFoundError(
                f"Expected local dataset structure like:\n"
                f"{root}/corpus\n{root}/query\n{root}/qrels\n"
                f"but not found."
            )

        # --- Load qrels first ---
        logger.info("Loading local qrels...")
        qrels = {}
        for row in ds_qrels:
            qid = str(row["query_id"])
            did = str(row["corpus_id"])
            score = int(row.get("score", 1))
            qrels.setdefault(qid, {})[did] = score
        logger.info(f"  Loaded {len(qrels)} query-doc relations")

        # --- Load queries ---
        logger.info("Loading local queries...")
        queries = {}
        for row in ds_queries:
            qid = str(row["id"])

            if not self._matches_domain(qid, row):
                continue

            text = row.get("text", "")
            images = self._extract_images_from_row(row)
            domain = self._infer_domain(qid, row)
            instruction = row.get(
                "instruction",
                "Retrieve relevant documents that help answer the question."
            )

            queries[qid] = Query(
                qid=qid,
                text=text,
                images=images,
                domain=domain,
                answer=row.get("answer", None),
                instruction=instruction,
            )

        qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}
        logger.info(f"  Loaded {len(queries)} queries for domains {self.domains}")

        # --- Load corpus ---
        logger.info("Loading local corpus...")
        corpus = {}
        for row in ds_corpus:
            did = str(row["id"])
            text = row.get("text", "")
            images = self._extract_images_from_row(row)
            modality = row.get("modality", "text")

            corpus[did] = Document(
                did=did,
                text=text,
                images=images,
                modality=modality
            )

        logger.info(f"  Loaded {len(corpus)} corpus documents")
        self._print_stats(queries, corpus, qrels)
        return queries, corpus, qrels

    def _load_from_huggingface(self):
        from datasets import load_dataset

        # --- Load qrels first (to know which queries we need) ---
        logger.info("Loading qrels...")
        ds_qrels = load_dataset(
            self.hf_dataset_id, "qrels", split="test",
            cache_dir=str(self.cache_dir)
        )
        qrels = {}
        for row in ds_qrels:
            qid = str(row["query_id"])
            did = str(row["corpus_id"])
            score = int(row.get("score", 1))
            qrels.setdefault(qid, {})[did] = score
        logger.info(f"  Loaded {len(qrels)} query-doc relations")

        # --- Load queries ---
        logger.info("Loading queries...")
        ds_queries = load_dataset(
            self.hf_dataset_id, "query", split="test",
            cache_dir=str(self.cache_dir)
        )

        queries = {}
        for row in ds_queries:
            qid = str(row["id"])

            # Filter by domain using query ID prefix
            if not self._matches_domain(qid, row):
                continue

            text = row.get("text", "")
            images = self._extract_images_from_row(row)
            domain = self._infer_domain(qid, row)
            instruction = row.get("instruction",
                "Retrieve relevant documents that help answer the question.")

            queries[qid] = Query(
                qid=qid,
                text=text,
                images=images,
                domain=domain,
                answer=row.get("answer", None),
                instruction=instruction,
            )

        # Also filter qrels to only keep queries we loaded
        qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}
        logger.info(f"  Loaded {len(queries)} queries for domains {self.domains}")

        # --- Load corpus (shared, 26k docs) ---
        logger.info("Loading corpus (26k docs, this may take a moment)...")
        ds_corpus = load_dataset(
            self.hf_dataset_id, "corpus", split="test",
            cache_dir=str(self.cache_dir)
        )

        corpus = {}
        for row in ds_corpus:
            did = str(row["id"])
            text = row.get("text", "")
            images = self._extract_images_from_row(row)
            modality = row.get("modality", "text")

            corpus[did] = Document(
                did=did, text=text, images=images, modality=modality
            )

        logger.info(f"  Loaded {len(corpus)} corpus documents")

        self._print_stats(queries, corpus, qrels)
        return queries, corpus, qrels

    def _extract_images_from_row(self, row: dict) -> List[Image.Image]:
        """Extract PIL images from MRMR row fields: image, image 1..4, vision."""
        images = []
        # MRMR uses these column names for images
        for key in ["image", "image 1", "image 2", "image 3", "image 4", "vision"]:
            val = row.get(key)
            if val is None:
                continue
            if isinstance(val, Image.Image):
                images.append(val.convert("RGB"))
            elif isinstance(val, dict) and "bytes" in val:
                from io import BytesIO
                images.append(Image.open(BytesIO(val["bytes"])).convert("RGB"))
        return images

    def _matches_domain(self, qid: str, row: dict) -> bool:
        """Check if a query belongs to one of our target domains."""
        domain_field = row.get("domain", "")
        if domain_field:
            for d in self.domains:
                if d.lower() in domain_field.lower():
                    return True

        for prefix in self._domain_prefixes:
            if prefix.lower() in qid.lower():
                return True

        return len(self._domain_prefixes) == 0

    def _infer_domain(self, qid: str, row: dict) -> str:
        """Infer which of our target domains this query belongs to."""
        domain_field = row.get("domain", "")
        if domain_field:
            return domain_field

        for domain, prefixes in self.DOMAIN_PREFIXES.items():
            for prefix in prefixes:
                if prefix.lower() in qid.lower():
                    return domain
        return "Unknown"

    def _load_from_local(self):
        """Fallback: load from local JSON files."""
        queries, corpus, qrels = {}, {}, {}

        for domain in self.domains:
            dk = domain.lower()
            for name, target in [
                (f"queries_{dk}.json", queries),
                (f"qrels_{dk}.json", qrels),
            ]:
                path = self.cache_dir / name
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    if target is queries:
                        for qid, item in data.items():
                            queries[qid] = Query(
                                qid=qid, text=item["text"],
                                domain=domain,
                            )
                    else:
                        qrels.update(data)

        corpus_file = self.cache_dir / "corpus.json"
        if corpus_file.exists():
            with open(corpus_file) as f:
                data = json.load(f)
            for did, item in data.items():
                corpus[did] = Document(did=did, text=item["text"])

        logger.info(f"Local load: {len(queries)} queries, {len(corpus)} docs")
        return queries, corpus, qrels

    def _print_stats(self, queries, corpus, qrels):
        domains = {}
        for q in queries.values():
            domains.setdefault(q.domain, []).append(q)

        print(f"\n{'='*55}")
        print(f"  MRMR Knowledge Dataset")
        print(f"{'='*55}")
        print(f"  Queries:  {len(queries)}")
        print(f"  Corpus:   {len(corpus)}")
        print(f"  Qrels:    {sum(len(v) for v in qrels.values())}")

        for domain, qs in sorted(domains.items()):
            n_img = sum(1 for q in qs if q.images)
            print(f"  {domain:12s}: {len(qs):4d} queries, {n_img} with images")

        n_doc_img = sum(1 for d in corpus.values() if d.images)
        print(f"  Corpus images: {n_doc_img}/{len(corpus)} docs have images")
        print(f"{'='*55}\n")


def get_dataset_stats(queries, corpus, qrels):
    """Standalone stats printer (for test compatibility)."""
    domains = {}
    for q in queries.values():
        domains.setdefault(q.domain, []).append(q)

    print(f"\n{'='*50}")
    print(f"MRMR Knowledge Dataset Stats")
    print(f"{'='*50}")
    print(f"Total queries:  {len(queries)}")
    print(f"Total corpus:   {len(corpus)}")
    print(f"Total qrels:    {sum(len(v) for v in qrels.values())}")

    for domain, qs in domains.items():
        n_with_img = sum(1 for q in qs if len(q.images) > 0)
        avg_imgs = sum(len(q.images) for q in qs) / max(len(qs), 1)
        print(f"  {domain}: {len(qs)} queries, "
              f"{n_with_img} with images, avg {avg_imgs:.1f} img/query")

    n_docs_with_img = sum(1 for d in corpus.values() if len(d.images) > 0)
    print(f"\n  Corpus: {n_docs_with_img}/{len(corpus)} docs have images")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config import cfg

    loader = MRMRKnowledgeLoader(
        domains=cfg.data.knowledge_domains,
        hf_dataset_id=cfg.data.hf_dataset_id,
        cache_dir=cfg.paths.data_dir,
    )
    queries, corpus, qrels = loader.load()