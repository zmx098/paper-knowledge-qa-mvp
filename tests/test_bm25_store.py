from paper_qa.store.bm25_store import BM25Store


def _chunks(entries):
    return [
        {"text": t, "source": s, "page": None, "heading": "", "chunk_index": i}
        for i, (t, s) in enumerate(entries)
    ]


def test_bm25_relevant_doc_ranked_first(tmp_path):
    store = BM25Store(tmp_path / "db")
    store.add(
        _chunks(
            [
                ("检索增强生成结合了检索模块与生成模型，用于提升知识问答的准确性", "a.md"),
                ("向量数据库用于存储论文的嵌入向量并支持相似度检索", "b.md"),
            ]
        )
    )
    hits = store.query("检索增强生成", top_k=1)
    assert hits
    assert hits[0].source == "a.md"


def test_bm25_persistence(tmp_path):
    store = BM25Store(tmp_path / "db")
    store.add(_chunks([("深度学习模型训练需要大量数据", "a.md")]))
    store2 = BM25Store(tmp_path / "db")
    assert store2.count == 1
    assert store2.query("深度学习", top_k=1)[0].source == "a.md"
