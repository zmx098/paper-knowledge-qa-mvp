import numpy as np

from paper_qa.store.vector_store import VectorStore


def _sample_metas(texts, source="a.md"):
    return [
        {"text": t, "source": source, "page": None, "heading": "", "chunk_index": i}
        for i, t in enumerate(texts)
    ]


def test_add_and_query_returns_most_similar(tmp_path):
    vs = VectorStore(tmp_path / "db")
    texts = ["我喜欢猫", "我喜欢狗"]
    vs.add(texts, [[1.0, 0.0], [0.0, 1.0]], _sample_metas(texts))
    hits = vs.query([1.0, 0.0], top_k=1)
    assert hits[0].text == "我喜欢猫"
    assert hits[0].score > 0.99


def test_persistence_across_reload(tmp_path):
    vs = VectorStore(tmp_path / "db")
    vs.add(["内容A"], [[1.0, 0.0, 0.0]], _sample_metas(["内容A"]))
    vs2 = VectorStore(tmp_path / "db")
    assert vs2.count == 1
    hits = vs2.query([1.0, 0.0, 0.0], top_k=1)
    assert hits[0].text == "内容A"


def test_clear_removes_all(tmp_path):
    vs = VectorStore(tmp_path / "db")
    vs.add(["内容A"], [[1.0, 0.0]], _sample_metas(["内容A"]))
    vs.clear()
    assert vs.count == 0
    assert vs.query([1.0, 0.0], top_k=1) == []
    vs2 = VectorStore(tmp_path / "db")
    assert vs2.count == 0


def test_add_dimension_mismatch_rejected(tmp_path):
    vs = VectorStore(tmp_path / "db")
    vs.add(["内容A"], [[1.0, 0.0]], _sample_metas(["内容A"]))
    try:
        vs.add(["内容B"], [[1.0]], _sample_metas(["内容B"]))
    except Exception:
        pass
    else:
        raise AssertionError("维度不一致应当报错")
