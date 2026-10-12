"""应用配置：从环境变量 / .env 文件读取。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    # ---- LLM（OpenAI 兼容接口）----
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    chat_model: str = field(
        default_factory=lambda: os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("LLM_EMBED_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = field(default_factory=lambda: _get_int("EMBEDDING_DIM", 1536))

    # ---- 检索与分块 ----
    retrieval_mode: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    )
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 100))
    db_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DB_DIR", str(PROJECT_ROOT / "data" / "db")))
    )

    @property
    def papers_dir(self) -> Path:
        return Path(os.getenv("PAPERS_DIR", str(PROJECT_ROOT / "data" / "papers")))

    def validate(self) -> None:
        if not self.llm_api_key:
            raise RuntimeError(
                "未配置 LLM_API_KEY。请先执行：cp .env.example .env，然后填入你的 API Key。"
            )
        if self.retrieval_mode not in {"vector", "bm25", "hybrid"}:
            raise RuntimeError(
                f"RETRIEVAL_MODE 仅支持 vector / bm25 / hybrid，当前为：{self.retrieval_mode}"
            )
