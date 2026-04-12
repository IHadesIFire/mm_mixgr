"""Configuration for the GME-only MRMR retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent
    cache_dir: Path = field(init=False)
    embedding_cache_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.cache_dir = self.project_root / "cache"
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.results_dir = self.project_root / "results"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in [self.cache_dir, self.embedding_cache_dir, self.results_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    hf_dataset_id: str = "MRMRbenchmark/knowledge"
    split: str = "test"
    domains: List[str] = field(
        default_factory=lambda: ["Science", "Medicine", "Art", "Humanities"]
    )
    query_instruction: str = "Retrieve relevant documents that help answer the question."


@dataclass
class ModelConfig:
    model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
    device: str = "cuda"
    batch_size: int = 4
    max_length: int = 4096
    max_image_tokens: int = 4096


@dataclass
class RetrievalConfig:
    top_k: int = 100
    corpus_chunk_size: int = 4096
    checkpoint_every: int = 1000


@dataclass
class EvalConfig:
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 100])
    primary_metric: str = "nDCG@10"


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


cfg = Config()
