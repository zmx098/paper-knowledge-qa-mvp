"""OpenAI 兼容的 LLM 客户端：对话生成 + 文本向量化。"""
from __future__ import annotations

from openai import OpenAI

from ..config import Config


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    def embed_texts(self, texts: list[str], batch: int = 32) -> list[list[float]]:
        """批量文本向量化，按请求顺序返回。"""
        results: list[list[float]] = []
        for i in range(0, len(texts), batch):
            resp = self.client.embeddings.create(
                model=self.cfg.embed_model, input=texts[i : i + batch]
            )
            ordered = sorted(resp.data, key=lambda item: item.index)
            results.extend(item.embedding for item in ordered)
        return results

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
