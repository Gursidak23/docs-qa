"""Construction helpers that wire collaborators together for the CLI and API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Settings
from .embed.base import Embedder
from .llm.base import LlmClient
from .retrieve.hybrid import HybridRetriever
from .retrieve.rerank import Reranker

if TYPE_CHECKING:
    from .cache import Cache
    from .eval.judge import Judge
    from .rag.answer import AnswerService


def build_embedder(settings: Settings) -> Embedder:
    if settings.embed.provider == "gemini":
        from .embed.gemini_embedder import GeminiEmbedder

        base: Embedder = GeminiEmbedder(settings.embed, settings.llm.gemini_api_key)
    else:
        from .embed.fastembed_embedder import FastEmbedEmbedder

        base = FastEmbedEmbedder(settings.embed)

    if settings.cache.enabled:
        from .embed.cache import CachingEmbedder

        return CachingEmbedder(base)
    return base


def build_cache(settings: Settings) -> Cache:
    from .cache import MemoryCache, NoopCache, RedisCache

    if not settings.cache.enabled:
        return NoopCache()
    if settings.cache.backend == "redis":
        return RedisCache(settings.redis.url)
    return MemoryCache()


def build_retriever(settings: Settings, embedder: Embedder | None = None) -> HybridRetriever:
    return HybridRetriever(embedder or build_embedder(settings), settings)


def build_reranker(settings: Settings) -> Reranker:
    if not settings.rerank.enabled:
        from .retrieve.rerank import NoopReranker

        return NoopReranker()

    from .retrieve.rerank import CrossEncoderReranker

    return CrossEncoderReranker(settings.rerank)


def _build_single_llm(settings: Settings, provider: str) -> LlmClient:
    if provider == "gemini":
        from .llm.gemini import GeminiClient

        return GeminiClient(settings.llm)
    if provider == "groq":
        from .llm.groq import GroqClient

        return GroqClient(settings.llm)
    if provider == "ollama":
        from .llm.ollama import OllamaClient

        return OllamaClient(settings.llm)
    if provider == "openrouter":
        from .llm.openrouter import OpenRouterClient

        return OpenRouterClient(settings.llm)
    raise ValueError(f"unknown LLM provider: {provider}")


def build_llm(settings: Settings) -> LlmClient:
    """Primary provider with retry, optional fallback, and token-bucket limiting."""
    from .llm.fallback import FallbackLlmClient

    primary = _build_single_llm(settings, settings.llm.provider)
    secondary = None
    fallback = settings.llm.fallback_provider
    if fallback != "none" and fallback != settings.llm.provider:
        secondary = _build_single_llm(settings, fallback)
    client: LlmClient = FallbackLlmClient(
        primary, secondary, max_retries=settings.llm.max_retries
    )

    rate = settings.cache.llm_rate_per_minute
    if rate and rate > 0:
        from .llm.ratelimit import AsyncTokenBucket, RateLimitedLlmClient

        client = RateLimitedLlmClient(client, AsyncTokenBucket(rate))
    return client


def build_answer_service(settings: Settings) -> AnswerService:
    """Assemble the full RAG pipeline (retriever + reranker + LLM + answer cache)."""
    from .rag.answer import AnswerService

    embedder = build_embedder(settings)
    return AnswerService(
        retriever=build_retriever(settings, embedder),
        reranker=build_reranker(settings),
        llm=build_llm(settings),
        settings=settings,
        cache=build_cache(settings),
    )


def _llm_has_credentials(settings: Settings) -> bool:
    provider = settings.llm.provider
    if provider == "gemini":
        return bool(settings.llm.gemini_api_key)
    if provider == "groq":
        return bool(settings.llm.groq_api_key)
    if provider == "openrouter":
        return bool(settings.llm.openrouter_api_key)
    return provider == "ollama"


def build_judge(settings: Settings, llm: LlmClient | None = None) -> Judge:
    """LLM-as-judge when enabled and credentialed; otherwise the lexical judge."""
    from .eval.judge import LexicalJudge, LlmJudge

    if settings.eval.use_llm_judge and _llm_has_credentials(settings):
        return LlmJudge(llm or build_llm(settings))
    return LexicalJudge()
