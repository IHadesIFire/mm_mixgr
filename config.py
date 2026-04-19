"""
MM-MixGR: Multi-Modal Mixed-Granularity Retrieval
Configuration for MRMR Knowledge subset (Science + Medicine pilot)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PathConfig:
    """All paths."""
    project_root: Path = Path(__file__).parent
    data_dir: Path = Path("./data/mrmr_raw")
    cache_dir: Path = Path("./cache")
    embedding_cache_dir: Path = Path("./cache/embeddings")
    decomposition_cache_dir: Path = Path("./cache/decompositions")
    results_dir: Path = Path("./results")

    def __post_init__(self):
        for d in [self.data_dir, self.cache_dir, self.embedding_cache_dir,
                  self.decomposition_cache_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """MRMR dataset configuration."""
    # HuggingFace dataset IDs (MRMR splits into separate repos per task)
    hf_dataset_id: str = "MRMRbenchmark/knowledge"
    # Other tasks: MRMRbenchmark/theorem, MRMRbenchmark/traffic,
    #              MRMRbenchmark/design, MRMRbenchmark/negation

    # Which Knowledge domains to use (pilot: Science + Medicine)
    knowledge_domains: List[str] = field(
        default_factory=lambda: ["Science", "Medicine"]
    )

    # Corpus is shared across all Knowledge tasks
    # 26,223 docs, avg 421.6 text tokens, avg 0.72 images per doc
    corpus_name: str = "knowledge_corpus"

    # Query stats from paper Table 2:
    # Science: 137 queries, avg 32.1 text tokens, avg 1.2 images
    # Medicine: 167 queries, avg 32.0 text tokens, avg 1.1 images


@dataclass
class ModelConfig:
    """Model configuration."""
    # --- Multimodal Embedding Model ---
    # Qwen3-VL-Embedding-2B: new, small, not contaminated on MRMR
    mm_encoder_name: str = "Qwen/Qwen3-VL-Embedding-2B"
    mm_encoder_dim: int = 2048  # output embedding dimension
    mm_encoder_max_tokens: int = 8192

    # --- Text Decomposition (for subqueries and propositions) ---
    # Propositionizer from Dense X Retrieval (Chen et al., 2023)
    # Used in MixGR: 96.3% query accuracy, 94.7% document accuracy
    decompose_model_name: str = "chentong00/propositionizer-wiki-flan-t5-large"

    # --- Visual Decomposition ---
    # For image region detection guided by subquery text
    visual_decompose_method: str = "vlm"  # "vlm" or "grounding_dino"

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"  # bfloat16 for 5070 Ti / H20
    batch_size: int = 8


@dataclass
class RetrievalConfig:
    """Retrieval and fusion configuration."""
    # Top-k candidates per granularity before RRF fusion
    top_k_per_granularity: int = 200

    # Final top-k to return after fusion
    top_k_final: int = 100

    # RRF constant (standard value from Cormack et al., 2009)
    rrf_k: int = 60

    # Granularities to enable
    # Text-side (from original MixGR):
    #   sq_d: full query text → full doc
    #   sq_p: full query text → doc proposition
    #   ss_p: subquery text → doc proposition
    # Multimodal (our extension):
    #   mm_d:  (full query text + image) → full doc
    #   mm_p:  (full query text + image) → doc proposition
    #   ms_p:  (subquery + relevant image region) → doc proposition
    enabled_granularities: List[str] = field(
        default_factory=lambda: [
            "sq_d",   # text: query → doc
            "sq_p",   # text: query → proposition
            "ss_p",   # text: subquery → proposition
            "mm_d",   # multimodal: (query+img) → doc
            "mm_p",   # multimodal: (query+img) → proposition
            "ms_p",   # multimodal: (subquery+region) → proposition
        ]
    )


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Metrics to compute
    metrics: List[str] = field(
        default_factory=lambda: ["nDCG@5", "nDCG@10", "nDCG@20", "Recall@10", "Recall@100"]
    )


@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment name for logging
    experiment_name: str = "mm_mixgr_knowledge_pilot"
    seed: int = 42


# Singleton for easy import
cfg = Config()