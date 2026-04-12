from config import cfg
from data.loader import available_domains
from embeddings.visual_encoder import create_encoder
from evaluation.metrics import evaluate_mrmr
from retrieval.granular import retrieve


def test_imports():
    assert cfg.data.hf_dataset_id == "MRMRbenchmark/knowledge"
    assert "Science" in available_domains()
    assert create_encoder("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct") is not None
    assert callable(evaluate_mrmr)
    assert callable(retrieve)
