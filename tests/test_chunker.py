from pathlib import Path

from paper_qa.ingestion.chunker import _split_long, chunk_document
from paper_qa.ingestion.loader import RawDocument


def test_split_long_respects_size_and_overlap():
    text = "字" * 2000
    pieces = _split_long(text, size=800, overlap=100)
    assert len(pieces) >= 3
    assert all(len(p) <= 800 for p in pieces)


def test_split_long_short_text_kept_whole():
    pieces = _split_long("短文本", size=800, overlap=100)
    assert pieces == ["短文本"]


def test_chunk_markdown_with_headings():
    doc = RawDocument(
        path=Path("a.md"),
        name="a.md",
        pages=["# 引言\n\n第一段内容。\n\n## 方法\n\n第二段内容。"],
    )
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 4  # 两个标题 + 两个正文段落
    assert chunks[0].heading == "引言"
    assert chunks[0].text == "# 引言"
    assert chunks[2].heading == "方法"


def test_chunk_page_number_attribution():
    doc = RawDocument(
        path=Path("b.pdf"),
        name="b.pdf",
        pages=["第一页正文。", "# 结论\n\n第二页结论。"],
    )
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=100)
    assert chunks[0].page == 1
    assert chunks[1].page == 2
    assert chunks[1].heading == "结论"
