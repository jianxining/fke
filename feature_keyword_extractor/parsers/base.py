from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from feature_keyword_extractor.schemas import BodyBlock, HeadingNode, ParsedDocument, make_doc_id


@dataclass
class _MutableHeading:
    heading_id: str
    title: str
    parent_id: Optional[str]
    ancestor_titles: List[str]
    level: int
    child_ids: List[str] = field(default_factory=list)
    body_ids: List[str] = field(default_factory=list)


class SectionGraphBuilder:
    """Builds a heading dependency graph while hiding heading level from output."""

    def __init__(self, path: Path):
        self.path = path
        self._heading_seq = 0
        self._body_seq = 0
        self._stack: List[_MutableHeading] = []
        self._headings: List[_MutableHeading] = []
        self._body_blocks: List[BodyBlock] = []

    def add_heading(self, title: str, level: int) -> None:
        clean_title = title.strip()
        if not clean_title:
            return

        effective_level = min(max(level, 1), 3)
        while self._stack and self._stack[-1].level >= effective_level:
            self._stack.pop()

        parent = self._stack[-1] if self._stack else None
        self._heading_seq += 1
        heading_id = f"H{self._heading_seq:03d}"
        node = _MutableHeading(
            heading_id=heading_id,
            title=clean_title,
            parent_id=parent.heading_id if parent else None,
            ancestor_titles=[*parent.ancestor_titles, parent.title] if parent else [],
            level=effective_level,
        )

        if parent:
            parent.child_ids.append(heading_id)
        self._headings.append(node)
        self._stack.append(node)

    def add_body(self, text: str) -> None:
        clean_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if not clean_text:
            return

        if not self._stack:
            self.add_heading("未命名内容", 1)

        self._body_seq += 1
        body_id = f"B{self._body_seq:03d}"
        self._body_blocks.append(BodyBlock(body_id=body_id, text=clean_text))
        self._stack[-1].body_ids.append(body_id)

    def to_document(self) -> ParsedDocument:
        return ParsedDocument(
            doc_id=make_doc_id(self.path),
            heading_nodes=[
                HeadingNode(
                    heading_id=node.heading_id,
                    title=node.title,
                    parent_id=node.parent_id,
                    ancestor_titles=node.ancestor_titles,
                    child_ids=node.child_ids,
                    body_ids=node.body_ids,
                )
                for node in self._headings
            ],
            body_blocks=self._body_blocks,
        )


HEADING_STYLE_LEVELS: Dict[str, int] = {
    "heading 1": 1,
    "标题 1": 1,
    "heading 2": 2,
    "标题 2": 2,
    "heading 3": 3,
    "标题 3": 3,
}
