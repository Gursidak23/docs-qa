from __future__ import annotations

from docsqa.models import RetrievedChunk
from docsqa.retrieve.hybrid import reciprocal_rank_fusion


def _rc(chunk_id: int, vrank: int | None = None, lrank: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        text=f"text-{chunk_id}",
        heading_path=None,
        page_no=None,
        uri="u",
        doc_title=None,
        vector_rank=vrank,
        lexical_rank=lrank,
    )


def test_rrf_rewards_items_present_in_both_lists() -> None:
    vector = [_rc(1, vrank=0), _rc(2, vrank=1), _rc(3, vrank=2)]
    lexical = [_rc(2, lrank=0), _rc(4, lrank=1), _rc(1, lrank=2)]

    fused = reciprocal_rank_fusion([vector, lexical], k=60)

    ids = [r.chunk_id for r in fused]
    assert ids[0] == 2  # appears near the top of both arms
    assert set(ids) == {1, 2, 3, 4}

    merged_two = next(r for r in fused if r.chunk_id == 2)
    assert merged_two.vector_rank == 1
    assert merged_two.lexical_rank == 0


def test_rrf_scores_are_descending() -> None:
    fused = reciprocal_rank_fusion([[_rc(1, vrank=0), _rc(2, vrank=1)]], k=60)
    assert fused[0].score >= fused[1].score


def test_rrf_handles_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
