"""HTML parser built on selectolax: keep heading structure, drop boilerplate."""

from __future__ import annotations

from selectolax.parser import HTMLParser

from ...models import Block, ParsedDocument

_DROP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside")
_CONTENT_SELECTOR = "h1,h2,h3,h4,h5,h6,p,li,pre"


def _is_heading(tag: str | None) -> bool:
    return tag is not None and len(tag) == 2 and tag[0] == "h" and tag[1].isdigit()


def parse_html(data: bytes | str, *, uri: str) -> ParsedDocument:
    html = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    tree = HTMLParser(html)

    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()

    title: str | None = None
    title_node = tree.css_first("title")
    if title_node is not None:
        title = title_node.text(strip=True) or None

    root = tree.body or tree.root
    blocks: list[Block] = []
    if root is not None:
        for node in root.css(_CONTENT_SELECTOR):
            txt = node.text(separator=" ", strip=True)
            if not txt:
                continue
            if _is_heading(node.tag):
                level = int(node.tag[1])
                blocks.append(Block(text=txt, kind="heading", level=level))
                if title is None:
                    title = txt
            else:
                blocks.append(Block(text=txt, kind="text"))

    if title is None and blocks:
        title = blocks[0].text[:200]
    byte_size = len(data) if isinstance(data, bytes) else len(data.encode("utf-8"))
    return ParsedDocument(source_type="html", title=title, blocks=blocks, byte_size=byte_size)
