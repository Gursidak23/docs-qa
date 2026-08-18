"""Optional Gemini embeddings adapter (free tier).

Note: ``text-embedding-004`` returns 768-dim vectors, so using this provider
requires setting ``DOCSQA_EMBED__DIM=768`` and a migration that widens the
``chunk.embedding`` column. The default fastembed model (384-dim) matches the
shipped schema out of the box.
"""

from __future__ import annotations

from typing import Any

from ..config import EmbedSettings


class GeminiEmbedder:
    def __init__(self, settings: EmbedSettings, api_key: str | None) -> None:
        self._settings = settings
        self._api_key = api_key
        self._client: Any = None

    @property
    def dim(self) -> int:
        return self._settings.dim

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        resp = client.models.embed_content(model=self._settings.gemini_model, contents=texts)
        return [list(e.values) for e in resp.embeddings]

    def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.models.embed_content(model=self._settings.gemini_model, contents=text)
        return list(resp.embeddings[0].values)
