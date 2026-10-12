"""文本分块：按段落切分、超长段落滑动切分，支持重叠与标题上下文。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import RawDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    heading: str = ""
    chunk_index: int = 0


def _split_paragraphs(text: str) -> list[str]:
    """按空行分段，保留 markdown 标题行。"""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_long(paragraph: str, size: int, overlap: int) -> list[str]:
    """超长段落按固定窗口滑动切分。"""
    if len(paragraph) <= size:
        return [paragraph]
    step = size - overlap
    pieces = []
    for i in range(0, len(paragraph), step):
        piece = paragraph[i : i + size].strip()
        if piece:
            pieces.append(piece)
    return pieces


def chunk_document(
    doc: RawDocument, chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Chunk]:
    """将一篇文档切成片段列表，记录来源、页码与所属小节标题。"""
    chunks: list[Chunk] = []
    for page_idx, page_text in enumerate(doc.pages, start=1):
        heading = ""
        for para in _split_paragraphs(page_text):
            m = _HEADING_RE.match(para)
            if m:
                heading = m.group(2)
                chunks.append(
                    Chunk(
                        text=para,
                        source=doc.name,
                        page=page_idx,
                        heading=heading,
                        chunk_index=len(chunks),
                    )
                )
                continue
            for piece in _split_long(para, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.name,
                        page=page_idx,
                        heading=heading,
                        chunk_index=len(chunks),
                    )
                )
    return chunks
