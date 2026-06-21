from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BodyBlock(BaseModel):
    body_id: str
    text: str


class HeadingNode(BaseModel):
    heading_id: str
    title: str
    parent_id: Optional[str]
    ancestor_titles: List[str] = Field(default_factory=list)
    child_ids: List[str] = Field(default_factory=list)
    body_ids: List[str] = Field(default_factory=list)

    def context_text(self) -> str:
        ancestors = " > ".join(self.ancestor_titles)
        return f"{ancestors} > {self.title}" if ancestors else self.title


class ParsedDocument(BaseModel):
    doc_id: str
    heading_nodes: List[HeadingNode] = Field(default_factory=list)
    body_blocks: List[BodyBlock] = Field(default_factory=list)

    @property
    def body_by_id(self) -> Dict[str, BodyBlock]:
        return {block.body_id: block for block in self.body_blocks}

    @property
    def heading_by_id(self) -> Dict[str, HeadingNode]:
        return {node.heading_id: node for node in self.heading_nodes}


def make_doc_id(path: Path) -> str:
    return path.name
