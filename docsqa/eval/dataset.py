"""Gold dataset format + loader for the evaluation harness.

The gold file is JSONL (one JSON object per line; blank lines and ``#`` comment
lines are ignored). Each case records a question, an optional reference answer,
and which documents/chunks are considered relevant. Relevance can be annotated
by stable source URI/path (recommended, portable across re-ingests) or by
database ids:

    {"id": "q1", "question": "How do I reset my password?",
     "reference_answer": "Open Settings > Security and click Reset.",
     "relevant_uris": ["docs/security.md"]}

Use either ``relevant_chunk_ids`` (chunk-level eval) or ``relevant_uris`` /
``relevant_doc_ids`` (document-level eval) per case, not a mix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class GoldCase:
    id: str
    question: str
    reference_answer: str | None = None
    relevant_doc_ids: list[int] = field(default_factory=list)
    relevant_chunk_ids: list[int] = field(default_factory=list)
    relevant_uris: list[str] = field(default_factory=list)

    @property
    def chunk_level(self) -> bool:
        """Whether retrieval should be scored per chunk rather than per document."""
        return bool(self.relevant_chunk_ids)

    @property
    def num_relevant(self) -> int:
        """Size of the gold relevant set used as the recall/nDCG denominator."""
        if self.relevant_chunk_ids:
            return len(set(self.relevant_chunk_ids))
        return len(set(self.relevant_uris)) + len(set(self.relevant_doc_ids))

    def is_relevant(
        self, *, chunk_id: int, document_id: int, uri: str
    ) -> bool:
        """True when a retrieved chunk matches this case's gold annotations."""
        if self.relevant_chunk_ids:
            return chunk_id in self.relevant_chunk_ids
        return document_id in self.relevant_doc_ids or uri in self.relevant_uris

    @classmethod
    def from_dict(cls, raw: dict, *, index: int = 0) -> GoldCase:
        question = str(raw.get("question") or raw.get("query") or "").strip()
        if not question:
            raise ValueError(f"gold case #{index} is missing a 'question'")
        reference = raw.get("reference_answer") or raw.get("answer")
        return cls(
            id=str(raw.get("id") or f"case-{index}"),
            question=question,
            reference_answer=str(reference) if reference is not None else None,
            relevant_doc_ids=[int(x) for x in raw.get("relevant_doc_ids", [])],
            relevant_chunk_ids=[int(x) for x in raw.get("relevant_chunk_ids", [])],
            relevant_uris=[
                str(x) for x in (raw.get("relevant_uris") or raw.get("relevant_ids", []))
            ],
        )


def parse_gold(lines: Iterable[str]) -> list[GoldCase]:
    """Parse JSONL lines into gold cases (ignoring blanks and ``#`` comments)."""
    cases: list[GoldCase] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cases.append(GoldCase.from_dict(json.loads(stripped), index=index))
    return cases


def load_gold(path: str | Path) -> list[GoldCase]:
    """Load a gold dataset from a JSONL file."""
    text = Path(path).read_text(encoding="utf-8")
    cases = parse_gold(text.splitlines())
    if not cases:
        raise ValueError(f"no gold cases found in {path}")
    return cases
