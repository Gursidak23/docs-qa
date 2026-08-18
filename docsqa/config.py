"""Strongly-typed, environment-overridable configuration.

All settings live under the ``DOCSQA_`` prefix and nested values use a double
underscore delimiter, e.g. ``DOCSQA_LLM__PROVIDER=groq`` or
``DOCSQA_POSTGRES__DSN=postgresql+asyncpg://...``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseModel):
    dsn: str = "postgresql+asyncpg://docsqa:docsqa@localhost:5432/docsqa"
    echo: bool = False
    pool_size: int = 10


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"


class EmbedSettings(BaseModel):
    provider: Literal["fastembed", "gemini"] = "fastembed"
    model_name: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    batch_size: int = 64
    # Used only when provider == "gemini".
    gemini_model: str = "text-embedding-004"


class RerankSettings(BaseModel):
    enabled: bool = True
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Keep this many candidates after reranking.
    top_n: int = 6


class RetrieveSettings(BaseModel):
    # Candidate counts from each arm before fusion.
    vector_top_k: int = 30
    lexical_top_k: int = 30
    # Reciprocal Rank Fusion constant (higher = flatter weighting).
    rrf_k: int = 60
    # How many fused candidates to hand to the reranker / LLM when rerank is off.
    fused_top_k: int = 20
    use_reranker: bool = True


class ChunkSettings(BaseModel):
    max_tokens: int = 400
    overlap_tokens: int = 64
    # Drop trailing fragments smaller than this (merged into the previous chunk).
    min_tokens: int = 64
    tokenizer_encoding: str = "cl100k_base"


class LlmSettings(BaseModel):
    provider: Literal["gemini", "groq", "ollama", "openrouter"] = "gemini"
    # Optional secondary provider tried when the primary errors/rate-limits.
    fallback_provider: Literal["gemini", "groq", "ollama", "openrouter", "none"] = "groq"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-oss-20b:free"

    temperature: float = 0.1
    max_output_tokens: int = 1024
    request_timeout_seconds: float = 60.0
    max_retries: int = 3


class CacheSettings(BaseModel):
    enabled: bool = True
    backend: Literal["memory", "redis"] = "memory"
    answer_ttl_seconds: int = 3600
    embed_ttl_seconds: int = 86400
    # Token-bucket guard so we stay inside free LLM quotas (requests/minute).
    llm_rate_per_minute: int = 25


class EvalSettings(BaseModel):
    dataset_path: str = "docsqa/eval/gold/support.jsonl"
    retrieval_k: int = 10
    # CI gates: a run "fails" if any metric drops below its floor.
    min_recall_at_k: float = 0.7
    min_mrr: float = 0.5
    min_groundedness: float = 0.8
    max_hallucination_rate: float = 0.15
    # Judge provider for groundedness/answer relevance (LLM-as-judge).
    use_llm_judge: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCSQA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Moonshot Docs Q&A"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    retrieve: RetrieveSettings = Field(default_factory=RetrieveSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
