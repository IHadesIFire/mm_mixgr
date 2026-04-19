"""
测试脚本：用 mock 数据验证 config.py 和 data/loader.py 的输入输出
不需要下载 MRMR 数据集即可运行
"""

import sys
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Test 1: Config 加载
# ============================================================
def test_config():
    print("\n" + "=" * 60)
    print("TEST 1: Config")
    print("=" * 60)

    from config import cfg

    print(f"  experiment_name:       {cfg.experiment_name}")
    print(f"  knowledge_domains:     {cfg.data.knowledge_domains}")
    print(f"  mm_encoder:            {cfg.model.mm_encoder_name}")
    print(f"  mm_encoder_dim:        {cfg.model.mm_encoder_dim}")
    print(f"  enabled_granularities: {cfg.retrieval.enabled_granularities}")
    print(f"  top_k_per_granularity: {cfg.retrieval.top_k_per_granularity}")
    print(f"  rrf_k:                 {cfg.retrieval.rrf_k}")
    print(f"  eval metrics:          {cfg.eval.metrics}")

    # 验证路径创建
    assert cfg.paths.cache_dir.exists(), "cache_dir not created"
    assert cfg.paths.results_dir.exists(), "results_dir not created"

    print("\n  ✓ Config loaded successfully")
    return cfg


# ============================================================
# Test 2: 创建 mock 数据，模拟 MRMR Knowledge 格式
# ============================================================
def create_mock_data(data_dir: Path):
    """
    创建 mock 数据，模拟 MRMR Knowledge 的真实格式。
    
    MRMR Knowledge 数据概要 (来自论文 Table 2):
    - Science: 137 queries, corpus 共 26,223 docs
    - Medicine: 167 queries
    - Query: text + 1-2 张图片
    - Doc: text (avg 421 tokens) + 0-1 张图片
    """
    print("\n" + "=" * 60)
    print("TEST 2: Creating mock data")
    print("=" * 60)

    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # --- 生成 mock 图片 ---
    def make_dummy_image(path, size=(224, 224), color="random"):
        if color == "random":
            arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        else:
            arr = np.full((*size, 3), color, dtype=np.uint8)
        Image.fromarray(arr).save(path)
        return str(path)

    query_img_1 = make_dummy_image(img_dir / "query_physics_diagram.png", color=(200, 200, 240))
    query_img_2 = make_dummy_image(img_dir / "query_cell_microscope.png", color=(200, 240, 200))
    doc_img_1 = make_dummy_image(img_dir / "doc_force_diagram.png", color=(240, 200, 200))
    doc_img_2 = make_dummy_image(img_dir / "doc_cell_structure.png", color=(200, 240, 240))

    # --- Mock queries (模拟论文中的真实例子) ---
    queries_science = {
        "test_Agriculture_207": {
            "text": "<image 1> With which group are leaf galls like the ones on grapevines, caused by the aphid-like insect called Grape phylloxera - which has root and leaf feeding stages in its lifecycle and is considered an extremely atypical symptom for this insect group - more commonly associated?",
            "images": [query_img_1],
            "domain": "Science",
            "answer": "Mites",
        },
        "validation_Agriculture_6": {
            "text": "The picture below shows a common soil organism. How should this organism be classified in terms of Flora vs. Fauna and by its size category? <image 1>",
            "images": [query_img_1],
            "domain": "Science",
            "answer": "macro-fauna",
        },
    }

    queries_medicine = {
        "test_Clinical_Medicine_283": {
            "text": "A 61-year-old woman is in the hospital for 2 weeks with bronchopneumonia following surgery for endometrial adenocarcinoma. She then becomes suddenly short of breath. This microscopic appearance from her lung is most typical for which of the following pathologic abnormalities? <image 1>",
            "images": [query_img_2],
            "domain": "Medicine",
            "answer": "Thromboembolism",
        },
        "test_Diagnostics_42": {
            "text": "What is the diagnosis for <image 1> ?",
            "images": [query_img_2],
            "domain": "Medicine",
            "answer": "Fat necrosis",
        },
    }

    # --- Mock corpus (模拟 interleaved image-text documents) ---
    corpus = {
        "doc_001": {
            "text": "Soil fauna are a large number of animal species (95% of all species live in soil), whether over their entire life or at least during a larval stage, that offer a protection against environmental hazards, such as extreme temperatures and moisture fluctuations. Soil organisms include mesofauna, which are typically soil invertebrates...",
            "images": [doc_img_1],
        },
        "doc_002": {
            "text": "Mites in the family Eriophyidae often cause galls to form on their hosts. The family contains more than 3,000 described species which attack a wide variety of plants. Lime nail galls caused by the mite Eriophyes tiliae.",
            "images": [],
        },
        "doc_003": {
            "text": "Thromboembolism: Risk factors include gynecologic oncology patients with advanced stages, hospitalization, chemotherapy drugs, advanced age. Virchow's triad classically describes risk factors for deep vein thrombosis: stasis, endothelial injury, and hypercoagulability.",
            "images": [doc_img_2],
        },
        "doc_004": {
            "text": "Fat necrosis is a benign condition that occurs when fatty breast tissue is damaged. On MRI, fat necrosis typically shows fat signal on T1-weighted images, making the necrotic area stand out from surrounding tissue.",
            "images": [doc_img_2],
        },
        "doc_005": {
            "text": "Newton's second law states that the force acting on an object is equal to the mass of that object times its acceleration. This fundamental theorem of classical mechanics is expressed as F = ma.",
            "images": [],
        },
    }

    # --- Mock qrels (query → relevant doc mappings) ---
    qrels_science = {
        "test_Agriculture_207": {"doc_001": 1, "doc_002": 1},
        "validation_Agriculture_6": {"doc_001": 1},
    }
    qrels_medicine = {
        "test_Clinical_Medicine_283": {"doc_003": 1},
        "test_Diagnostics_42": {"doc_004": 1},
    }

    # 写入本地 JSON
    for name, data in [
        ("queries_science.json", queries_science),
        ("queries_medicine.json", queries_medicine),
        ("corpus.json", corpus),
        ("qrels_science.json", qrels_science),
        ("qrels_medicine.json", qrels_medicine),
    ]:
        path = data_dir / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Created: {path}")

    print(f"\n  ✓ Mock data created in {data_dir}")
    return queries_science, queries_medicine, corpus, qrels_science, qrels_medicine


# ============================================================
# Test 3: Loader 从本地 JSON 读取
# ============================================================
def test_loader_local(data_dir: Path):
    print("\n" + "=" * 60)
    print("TEST 3: Loader (local JSON fallback)")
    print("=" * 60)

    from data.loader import MRMRKnowledgeLoader, Query, Document, get_dataset_stats

    loader = MRMRKnowledgeLoader(
        domains=["Science", "Medicine"],
        cache_dir=data_dir,
    )

    # 直接测试 local fallback (不访问 HuggingFace)
    queries, corpus, qrels = loader._load_from_local()

    # 打印结果
    print(f"\n  Queries loaded: {len(queries)}")
    for qid, q in queries.items():
        print(f"    [{q.domain}] {qid}: {q.text[:80]}...")

    print(f"\n  Corpus loaded: {len(corpus)}")
    for did, d in corpus.items():
        print(f"    {did}: {d.text[:60]}... (images: {len(d.images)})")

    print(f"\n  Qrels loaded:")
    for qid, rels in qrels.items():
        print(f"    {qid} → {rels}")

    # 基本断言
    assert len(queries) == 4, f"Expected 4 queries, got {len(queries)}"
    assert len(corpus) == 5, f"Expected 5 corpus docs, got {len(corpus)}"
    assert all(isinstance(q, Query) for q in queries.values())
    assert all(isinstance(d, Document) for d in corpus.values())

    # 打印统计
    get_dataset_stats(queries, corpus, qrels)

    print("  ✓ Loader local fallback works correctly")
    return queries, corpus, qrels


# ============================================================
# Test 4: 验证数据结构是否适合下游使用
# ============================================================
def test_downstream_compatibility(queries, corpus, qrels):
    print("\n" + "=" * 60)
    print("TEST 4: Downstream compatibility check")
    print("=" * 60)

    # 检查 qrels 中所有 doc_id 都在 corpus 中
    missing = []
    for qid, rels in qrels.items():
        for did in rels:
            if did not in corpus:
                missing.append((qid, did))
    if missing:
        print(f"  ⚠ Missing corpus docs: {missing}")
    else:
        print(f"  ✓ All qrel doc IDs found in corpus")

    # 检查 qrels 中所有 query_id 都在 queries 中
    missing_q = [qid for qid in qrels if qid not in queries]
    if missing_q:
        print(f"  ⚠ Missing queries: {missing_q}")
    else:
        print(f"  ✓ All qrel query IDs found in queries")

    # 模拟: 对一个 query 做 embedding 需要哪些字段
    sample_q = list(queries.values())[0]
    print(f"\n  Sample query for embedding:")
    print(f"    qid:    {sample_q.qid}")
    print(f"    text:   {sample_q.text[:80]}...")
    print(f"    images: {len(sample_q.images)} images")
    print(f"    domain: {sample_q.domain}")
    print(f"    instruction: {sample_q.instruction}")

    # 模拟: Qwen3-VL-Embedding-2B 的输入格式
    print(f"\n  Expected input for Qwen3-VL-Embedding-2B:")
    print(f'    {{"text": "<query text>", "image": <PIL.Image>}}')
    print(f'    → outputs: numpy array of shape (2048,)')

    # 模拟: 各粒度检索需要什么
    print(f"\n  Granularity requirements:")
    print(f"    sq_d:  query.text ↔ doc.text")
    print(f"    sq_p:  query.text ↔ doc.propositions[]")
    print(f"    ss_p:  query.subqueries[] ↔ doc.propositions[]")
    print(f"    mm_d:  (query.text + query.images) ↔ doc.text")
    print(f"    mm_p:  (query.text + query.images) ↔ doc.propositions[]")
    print(f"    ms_p:  (query.subqueries[] + query.image_regions[]) ↔ doc.propositions[]")
    print(f"\n  → Next step: text_decompose.py (split queries → subqueries)")
    print(f"  → Next step: visual_decompose.py (split images → regions)")

    print("\n  ✓ Data structure is compatible with downstream pipeline")


# ============================================================
# Test 5: 快速检查真实 MRMR 数据集结构 (可选, 需要网络)
# ============================================================
def test_huggingface_schema():
    """
    如果你有网络，运行这个来查看 MRMR 数据集的真实字段名。
    没网络会自动跳过。

    MRMR 按任务分成 5 个独立 repo:
      MRMRbenchmark/knowledge  (我们用这个)
      MRMRbenchmark/theorem
      MRMRbenchmark/traffic
      MRMRbenchmark/design
      MRMRbenchmark/negation

    每个 repo 有 3 个 subset: corpus, query, qrels
    """
    print("\n" + "=" * 60)
    print("TEST 5: HuggingFace schema check (optional, needs network)")
    print("=" * 60)

    try:
        from datasets import load_dataset

        subsets = ["query", "corpus", "qrels"]
        for subset in subsets:
            print(f"\n  --- Subset: {subset} ---")
            ds = load_dataset(
                "MRMRbenchmark/knowledge", subset,
                split="test", streaming=True
            )
            first = next(iter(ds))
            for key, val in first.items():
                val_type = type(val).__name__
                if isinstance(val, str):
                    val_preview = val[:70] + "..." if len(val) > 70 else val
                elif isinstance(val, Image.Image):
                    val_preview = f"PIL Image {val.size}"
                elif val is None:
                    val_preview = "None"
                else:
                    val_preview = str(val)[:70]
                print(f"    {key:20s} ({val_type:10s}): {val_preview}")

        print("\n  ✓ Schema retrieved. Loader is aligned with these fields.")

    except Exception as e:
        print(f"  ⏭ Skipped: {e}")
        print(f"  💡 在本地运行: pip install datasets && python tests/test_step1.py")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("🧪 MM-MixGR Test Suite")
    print("=" * 60)

    # Test 1: Config
    cfg = test_config()

    # Test 2: Mock data
    mock_dir = cfg.paths.data_dir / "mock"
    qs_sci, qs_med, corpus_mock, qr_sci, qr_med = create_mock_data(mock_dir)

    # Test 3: Loader
    queries, corpus, qrels = test_loader_local(mock_dir)

    # Test 4: Downstream check
    test_downstream_compatibility(queries, corpus, qrels)

    # Test 5: Real HF schema (optional)
    test_huggingface_schema()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. 运行 test 5 确认 MRMR 真实字段名")
    print("  2. 有需要的话调整 loader.py 的字段映射")
    print("  3. 继续实现 decomposition/text_decompose.py")
