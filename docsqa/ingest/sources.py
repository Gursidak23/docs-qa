"""Expand user-supplied paths into a list of ingestible files."""

from __future__ import annotations

from pathlib import Path

from .parsers.registry import SUPPORTED_EXTENSIONS


def expand_paths(paths: list[str]) -> list[Path]:
    """Return supported files from the given files/directories (recursive)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    resolved = f.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        out.append(f)
        elif path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
    return out
