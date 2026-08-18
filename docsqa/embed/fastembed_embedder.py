"""Local, free embeddings via fastembed (ONNX, CPU).

Default model is ``BAAI/bge-small-en-v1.5`` (384-dim). BGE models benefit from a
query-side instruction, so ``embed_query`` prepends it. The model is loaded
lazily on first use (it downloads a small ONNX file once, then runs offline).
"""

from __future__ import annotations

from typing import Any

from ..config import EmbedSettings

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbedder:
    def __init__(self, settings: EmbedSettings) -> None:
        self._settings = settings
        self._model: Any = None

    @property
    def dim(self) -> int:
        return self._settings.dim

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._settings.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        return [vec.tolist() for vec in model.embed(texts, batch_size=self._settings.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        query = text
        if "bge" in self._settings.model_name.lower():
            query = _BGE_QUERY_INSTRUCTION + text
        return next(iter(model.embed([query]))).tolist()
