"""论文/文档加载器：支持 PDF、Markdown、纯文本。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class RawDocument:
    """一份原始文档：path / name / pages（PDF 每页一段，md/txt 仅一页）。"""

    path: Path
    name: str
    pages: list[str]


def load_document(path: Path) -> RawDocument:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return RawDocument(path=path, name=path.name, pages=[text])
    raise ValueError(f"不支持的格式：{ext}（支持 {sorted(SUPPORTED_EXTS)}）")


def _load_pdf(path: Path) -> RawDocument:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return RawDocument(path=path, name=path.name, pages=pages)
