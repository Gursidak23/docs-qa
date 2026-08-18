"""Retrieval: dense (pgvector), lexical (Postgres FTS), and RRF hybrid fusion."""

from .hybrid import HybridRetriever, reciprocal_rank_fusion

__all__ = ["HybridRetriever", "reciprocal_rank_fusion"]
