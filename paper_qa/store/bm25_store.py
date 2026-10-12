"""BM25 + jieba 的本地关键词检索。

不依赖 embedding 接口，适用于只有对话模型、没有向量化能力的 LLM 服务
（例如 DeepSeek），也适合完全离线的检索场景。
"""
from __future__ import annotations

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Plus

from .vector_store import SearchHit


class BM25Store:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.path = db_dir / "bm25.json"
        self._docs: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None
        self._load()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w.strip() for w in jieba.lcut(text) if w.strip()]

    def _load(self) -> None:
        if self.path.exists():
            self._docs = json.loads(self.path.read_text(encoding="utf-8"))
            self._corpus = [self._tokenize(d["text"]) for d in self._docs]
            if self._docs:
                self._bm25 = BM25Plus(self._corpus)

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @property
    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs = []
        self._corpus = []
        self._bm25 = None
        if self.path.exists():
            self.path.unlink()

    def add(self, chunks: list[dict]) -> None:
        self._docs.extend(chunks)
        self._corpus.extend(self._tokenize(c["text"]) for c in chunks)
        self._bm25 = BM25Plus(self._corpus)
        self._save()

    def query(self, text: str, top_k: int) -> list[SearchHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(text))
        # BM25Plus 带 delta 平滑，适合短文档/小语料，按分数取 top_k 即可
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for i in idx:
            d = self._docs[i]
            hits.append(
                SearchHit(
                    text=d["text"],
                    source=d["source"],
                    page=d.get("page"),
                    heading=d.get("heading", ""),
                    chunk_index=d.get("chunk_index"),
                    score=float(scores[i]),
                )
            )
        return hits
