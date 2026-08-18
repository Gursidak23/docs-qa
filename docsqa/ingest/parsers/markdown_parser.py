"""Markdown parser: preserve heading structure for better chunk provenance."""

from __future__ import annotations

from markdown_it import MarkdownIt

from ...models import Block, ParsedDocument

_md = MarkdownIt("commonmark")


def parse_markdown(data: bytes, *, uri: str) -> ParsedDocument:
    text = data.decode("utf-8", "replace")
    tokens = _md.parse(text)

    blocks: list[Block] = []
    title: str | None = None
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1]) if len(tok.tag) > 1 and tok.tag[1].isdigit() else 1
            inline = tokens[i + 1] if i + 1 < n else None
            heading = inline.content.strip() if inline is not None else ""
            if heading:
                blocks.append(Block(text=heading, kind="heading", level=level))
                if title is None and level <= 2:
                    title = heading
            i += 2  # skip the inline + heading_close handled by loop
            continue
        if tok.type in ("inline", "fence", "code_block"):
            content = tok.content.strip()
            if content:
                blocks.append(Block(text=content, kind="text"))
        i += 1

    if title is None and blocks:
        title = blocks[0].text[:200]
    return ParsedDocument(source_type="md", title=title, blocks=blocks, byte_size=len(data))
