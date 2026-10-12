"""问答流水线：检索 → 组装上下文 → 生成带引用的回答。"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..knowledge_base import KnowledgeBase
from ..store.vector_store import SearchHit

SYSTEM_PROMPT = (
    "你是一个严谨的科研助手，只能基于【论文片段】中提供的内容回答用户问题。\n"
    "规则：\n"
    "1. 在引用到具体片段内容时，于相关句末用 [编号] 标注来源，编号对应片段列表。\n"
    "2. 如果片段不足以回答，直接说明“知识库中没有找到相关依据”，不要编造或外推。\n"
    "3. 用中文回答，语言简洁、有逻辑，必要时分点说明。"
)


@dataclass
class Answer:
    text: str
    sources: list[SearchHit]


class QAPipeline:
    def __init__(self, kb: KnowledgeBase, cfg: Config):
        self.kb = kb
        self.cfg = cfg

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        history: list[dict] | None = None,
    ) -> Answer:
        hits = self.kb.retrieve(query, top_k)
        if not hits:
            return Answer(
                text="知识库为空或没有检索到相关内容，请先执行 ingest 导入论文。",
                sources=[],
            )

        context_parts = []
        for i, h in enumerate(hits, start=1):
            loc = f"{h.source}" + (f"（第 {h.page} 页）" if h.page else "")
            if h.heading:
                loc += f" · 小节：{h.heading}"
            context_parts.append(f"[{i}] 来源：{loc}\n{h.text}\n")
        context = "\n".join(context_parts)

        history_text = ""
        if history:
            lines = [
                f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
                for m in history[-6:]
            ]
            history_text = "\n【对话历史】\n" + "\n".join(lines) + "\n"

        user_prompt = (
            "请基于以下论文片段回答用户问题。\n\n"
            "【论文片段】\n"
            f"{context}\n"
            f"{history_text}"
            "【用户问题】\n"
            f"{query}\n"
        )

        text = self.kb.llm.chat(SYSTEM_PROMPT, user_prompt)
        return Answer(text=text, sources=hits)
