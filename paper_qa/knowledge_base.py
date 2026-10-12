"""知识库门面：负责入库（解析→分块→向量化→存储）与检索。"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .ingestion.chunker import chunk_document
from .ingestion.loader import SUPPORTED_EXTS, load_document
from .llm.client import LLMClient
from .store.bm25_store import BM25Store
from .store.vector_store import SearchHit, VectorStore


class KnowledgeBase:
    def __init__(self, cfg: Config, llm: LLMClient | None = None):
        self.cfg = cfg
        self.llm = llm or LLMClient(cfg)
        self.vector_store = VectorStore(cfg.db_dir)
        self.bm25_store = BM25Store(cfg.db_dir)

    # ---------------- 入库 ----------------
    def ingest_path(self, path: Path, clear: bool = False) -> int:
        """导入单个文件或目录（递归查找 pdf / md / txt）。"""
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
            )
        elif path.is_file():
            files = [path]
        else:
            raise FileNotFoundError(f"路径不存在：{path}")

        if not files:
            print(f"在 {path} 下没有找到支持的文档（pdf/md/txt）。")
            return 0

        if clear:
            self.vector_store.clear()
            self.bm25_store.clear()

        total = 0
        for file_path in files:
            doc = load_document(file_path)
            chunks = chunk_document(doc, self.cfg.chunk_size, self.cfg.chunk_overlap)
            metas = [
                {
                    "text": c.text,
                    "source": c.source,
                    "page": c.page,
                    "heading": c.heading,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ]
            if not metas:
                print(f"  ⚠ {doc.name}: 未能提取到有效文本，已跳过。")
                continue
            if self.cfg.retrieval_mode in {"vector", "hybrid"}:
                embeddings = self.llm.embed_texts([m["text"] for m in metas])
                self.vector_store.add(
                    [m["text"] for m in metas], embeddings, metas
                )
            if self.cfg.retrieval_mode in {"bm25", "hybrid"}:
                self.bm25_store.add(metas)
            total += len(chunks)
            print(f"  ✓ {doc.name}: {len(chunks)} 个片段")
        return total

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        top_k = top_k or self.cfg.top_k
        mode = self.cfg.retrieval_mode

        if mode == "bm25":
            return self.bm25_store.query(query, top_k)

        query_emb = self.llm.embed_texts([query])[0]
        if mode == "vector":
            return self.vector_store.query(query_emb, top_k)

        # hybrid：向量与 BM25 各自归一化后加权融合
        vec_hits = self.vector_store.query(query_emb, top_k * 2)
        bm_hits = self.bm25_store.query(query, top_k * 2)
        return _fusion(vec_hits, bm_hits, top_k, alpha=0.6)


def _minmax(v: float, arr: list[float]) -> float:
    lo, hi = min(arr), max(arr)
    if hi - lo < 1e-9:
        return 1.0
    return (v - lo) / (hi - lo)


def _fusion(
    vec_hits: list[SearchHit],
    bm_hits: list[SearchHit],
    top_k: int,
    alpha: float = 0.6,
) -> list[SearchHit]:
    """按 (source, chunk_index) 对齐两路结果，加权求和后取 top_k。"""
    combined: dict[tuple[str, int | None], dict] = {}
    for h in vec_hits:
        combined[(h.source, h.chunk_index)] = {
            "hit": h,
            "score": alpha * _minmax(h.score, [x.score for x in vec_hits]),
        }
    for h in bm_hits:
        key = (h.source, h.chunk_index)
        extra = (1 - alpha) * _minmax(h.score, [x.score for x in bm_hits])
        if key in combined:
            combined[key]["score"] += extra
        else:
            combined[key] = {"hit": h, "score": extra}
    ranked = sorted(combined.values(), key=lambda d: d["score"], reverse=True)[:top_k]
    return [d["hit"] for d in ranked]
