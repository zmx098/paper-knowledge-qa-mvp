"""基于 NumPy 的轻量向量库（余弦相似度），持久化到本地磁盘。

MVP 阶段不引入 Chroma/FAISS 等重依赖，方便开箱即用；
后续可平滑替换为正式向量数据库（见 README 路线图）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    text: str
    source: str
    page: int | None
    heading: str
    chunk_index: int | None
    score: float


class VectorStore:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = db_dir / "vectors.npy"
        self.meta_path = db_dir / "metadata.json"
        self._vectors: np.ndarray | None = None
        self._metas: list[dict] = []
        self._load()

    # ---- 持久化 ----
    def _load(self) -> None:
        if self.vectors_path.exists() and self.meta_path.exists():
            self._vectors = np.load(self.vectors_path)
            self._metas = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self._vectors is not None:
            np.save(self.vectors_path, self._vectors)
        self.meta_path.write_text(
            json.dumps(self._metas, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @property
    def count(self) -> int:
        return len(self._metas)

    def clear(self) -> None:
        self._vectors = None
        self._metas = []
        for p in (self.vectors_path, self.meta_path):
            if p.exists():
                p.unlink()

    # ---- 写入 ----
    def add(
        self, texts: list[str], embeddings: list[list[float]], metas: list[dict]
    ) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("embeddings 必须是二维数组")
        if len(arr) != len(metas):
            raise ValueError("embeddings 与 metas 长度不一致")
        self._vectors = arr if self._vectors is None else np.concatenate([self._vectors, arr], axis=0)
        self._metas.extend(metas)
        self._save()

    # ---- 查询 ----
    def query(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        if self._vectors is None or not self._metas:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        v_norm = np.linalg.norm(self._vectors, axis=1)
        v_norm[v_norm == 0] = 1e-9
        sims = ((q @ self._vectors.T) / (v_norm * np.linalg.norm(q) + 1e-12)).reshape(-1)
        idx = np.argsort(-sims)[:top_k]
        hits = []
        for i in idx:
            meta = self._metas[int(i)]
            hits.append(
                SearchHit(
                    text=meta["text"],
                    source=meta["source"],
                    page=meta.get("page"),
                    heading=meta.get("heading", ""),
                    chunk_index=meta.get("chunk_index"),
                    score=float(sims[int(i)]),
                )
            )
        return hits
