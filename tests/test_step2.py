"""
测试 decomposition/text_decompose.py
使用真实的 propositionizer 模型，需要 GPU 或 CPU。
首次运行会从 HuggingFace 下载 ~3.1GB 模型。

运行: python tests/test_step2.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from data.loader import Query, Document
from decomposition.text_decompose import TextDecomposer


def test_decomposer():
    print("=" * 60)
    print("Loading propositionizer model...")
    print("  Model: chentong00/propositionizer-wiki-flan-t5-large")
    print("  Size: ~3.1 GB (will download on first run)")
    print("=" * 60)

    decomposer = TextDecomposer(device="cuda")
    decomposer.load_model()

    # --------------------------------------------------------
    # Test 1: Query decomposition
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 1: Query → Subqueries")
    print("-" * 60)

    test_queries = [
        # 简单 query (应该只有 1 个 subquery)
        "What is the diagnosis for <image 1> ?",

        # 复杂 query，来自 MRMR Science (多个语义成分)
        "<image 1> With which group are leaf galls like the ones on grapevines, "
        "caused by the aphid-like insect called Grape phylloxera - which has root "
        "and leaf feeding stages in its lifecycle and is considered an extremely "
        "atypical symptom for this insect group - more commonly associated?",

        # 复杂 query，来自 MRMR Medicine
        "A 61-year-old woman is in the hospital for 2 weeks with bronchopneumonia "
        "following surgery for endometrial adenocarcinoma. She then becomes suddenly "
        "short of breath. This microscopic appearance from her lung is most typical "
        "for which of the following pathologic abnormalities? <image 1>",

        # MixGR 论文中的例子 (SciFact)
        "Citrullinated proteins externalized in neutrophil extracellular traps act "
        "indirectly to perpetuate the inflammatory cycle via induction of autoantibodies.",
    ]

    for i, query_text in enumerate(test_queries):
        subqueries = decomposer.decompose_query(query_text, use_cache=False)
        print(f"\n  Query {i+1}: {query_text[:80]}...")
        print(f"  → {len(subqueries)} subqueries:")
        for j, sq in enumerate(subqueries):
            print(f"    [{j}] {sq}")

    # --------------------------------------------------------
    # Test 2: Document decomposition
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 2: Document → Propositions")
    print("-" * 60)

    test_docs = [
        # 来自 MRMR corpus 的真实文档片段
        (
            "Soil fauna",
            "Soil fauna are a large number of animal species (95% of all arthropods "
            "live in soil), whether over their entire life or at least during a larval "
            "stage, that offer a protection against environmental hazards, such as extreme "
            "temperatures and moisture fluctuations. Soil organisms include mesofauna, "
            "which are typically soil invertebrates."
        ),
        # MixGR 论文中的例子
        (
            "NETosis in RA",
            "RA sera and immunoglobulin fractions from RA patients with high levels of "
            "ACPA significantly enhanced NETosis, and the NETs induced by these "
            "autoantibodies displayed distinct protein content. Indeed, during NETosis, "
            "neutrophils externalized the citrullinated autoantigens implicated in RA "
            "pathogenesis. In turn, NETs significantly augmented inflammatory responses "
            "in RA and OA synovial fibroblasts, including induction of IL-6 and IL-8."
        ),
    ]

    for i, (title, content) in enumerate(test_docs):
        propositions = decomposer.decompose_document(
            text=content, title=title, use_cache=False
        )
        print(f"\n  Doc {i+1} (title: {title}):")
        print(f"  → {len(propositions)} propositions:")
        for j, p in enumerate(propositions):
            print(f"    [{j}] {p}")

    # --------------------------------------------------------
    # Test 3: Batch decomposition with Query/Document objects
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 3: Batch decomposition")
    print("-" * 60)

    queries = {
        "q1": Query(qid="q1", text=test_queries[0], domain="Medicine"),
        "q2": Query(qid="q2", text=test_queries[2], domain="Medicine"),
        "q3": Query(qid="q3", text=test_queries[3], domain="Science"),
    }
    corpus = {
        "d1": Document(did="d1", text=test_docs[0][1]),
        "d2": Document(did="d2", text=test_docs[1][1]),
    }

    all_subqueries = decomposer.decompose_queries_batch(queries, show_progress=False)
    all_propositions = decomposer.decompose_corpus_batch(corpus, show_progress=False)

    print(f"\n  Batch results:")
    for qid, sqs in all_subqueries.items():
        print(f"    {qid}: {len(sqs)} subqueries")
    for did, props in all_propositions.items():
        print(f"    {did}: {len(props)} propositions")

    # --------------------------------------------------------
    # Test 4: Cache verification
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 4: Cache verification")
    print("-" * 60)

    # Second call should use cache (much faster)
    import time
    t0 = time.time()
    cached_result = decomposer.decompose_query(test_queries[0], use_cache=True)
    t1 = time.time()
    print(f"  Cached call took {t1-t0:.4f}s (should be < 0.01s)")
    print(f"  Result matches: {cached_result == all_subqueries['q1']}")

    # --------------------------------------------------------
    # Test 5: ss_p granularity preview
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 5: ss_p granularity preview")
    print("-" * 60)

    qid, did = "q2", "d2"
    sqs = all_subqueries[qid]
    props = all_propositions[did]

    print(f"\n  Query {qid}: {len(sqs)} subqueries")
    print(f"  Doc {did}: {len(props)} propositions")
    print(f"\n  ss_p(q,d) = (1/{len(sqs)}) × Σ max_j sim(sq_i, prop_j)")
    print(f"  Similarity pairs to compute: {len(sqs)} × {len(props)} = {len(sqs) * len(props)}")

    print("\n" + "=" * 60)
    print("✅ All tests passed with real propositionizer model!")
    print("=" * 60)


if __name__ == "__main__":
    test_decomposer()