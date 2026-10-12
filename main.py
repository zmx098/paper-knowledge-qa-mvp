#!/usr/bin/env python3
"""个人论文知识库问答系统（MVP）— 命令行入口。

用法示例：
  python main.py ingest --dir data/papers            # 导入论文目录
  python main.py ingest --dir examples/sample_paper.md --clear
  python main.py ask "什么是检索增强生成？"          # 单次提问
  python main.py ask "论文结论是什么？" --top-k 8    # 指定检索片段数
  python main.py chat                                # 交互式问答
  python main.py list                                # 查看知识库规模
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_qa.config import Config
from paper_qa.knowledge_base import KnowledgeBase
from paper_qa.qa.pipeline import QAPipeline


def _make_kb() -> KnowledgeBase:
    cfg = Config()
    cfg.validate()
    return KnowledgeBase(cfg)


def _source_line(h) -> str:
    loc = f"{h.source}" + (f"（第 {h.page} 页）" if h.page else "")
    if h.heading:
        loc += f" · {h.heading}"
    return loc


def cmd_ingest(args) -> int:
    kb = _make_kb()
    target = Path(args.dir)
    print(f"开始入库：{target}")
    total = kb.ingest_path(target, clear=args.clear)
    print(f"完成，共入库 {total} 个片段。")
    return 0


def cmd_ask(args) -> int:
    kb = _make_kb()
    pipe = QAPipeline(kb, Config())
    answer = pipe.answer(args.question, top_k=args.top_k)
    print("\n" + answer.text + "\n")
    if answer.sources:
        print("— 参考来源 —")
        for i, h in enumerate(answer.sources, start=1):
            print(f"[{i}] {_source_line(h)}")
    return 0


def cmd_chat(args) -> int:
    kb = _make_kb()
    cfg = Config()
    pipe = QAPipeline(kb, cfg)
    history: list[dict] = []
    print("进入问答模式（输入 exit / quit 退出）\n")
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "退出"}:
            break
        answer = pipe.answer(q, top_k=args.top_k, history=history)
        print(f"\nAI：{answer.text}\n")
        if answer.sources:
            print("参考来源：" + "；".join(f"[{i}]{h.source}" for i, h in enumerate(answer.sources, start=1)))
            print()
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer.text})
        history = history[-10:]
    return 0


def cmd_list(args) -> int:
    cfg = Config()
    kb = KnowledgeBase(cfg)  # 仅读本地索引，无需 API Key
    print(f"向量库片段数：{kb.vector_store.count}；BM25 片段数：{kb.bm25_store.count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-qa", description="个人论文知识库问答系统（MVP）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="导入论文到知识库")
    p_ingest.add_argument("--dir", default="data/papers", help="论文目录或单个文件路径")
    p_ingest.add_argument("--clear", action="store_true", help="清空现有索引后重新导入")

    p_ask = sub.add_parser("ask", help="针对知识库提问")
    p_ask.add_argument("question", help="问题内容")
    p_ask.add_argument("--top-k", type=int, default=None, help="检索片段数")

    p_chat = sub.add_parser("chat", help="交互式问答")
    p_chat.add_argument("--top-k", type=int, default=None, help="检索片段数")

    sub.add_parser("list", help="查看知识库规模")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handlers = {"ingest": cmd_ingest, "ask": cmd_ask, "chat": cmd_chat, "list": cmd_list}
    try:
        return handlers[args.command](args)
    except RuntimeError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - CLI 兜底
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
