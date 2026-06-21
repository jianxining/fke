from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from feature_keyword_extractor.parsers.base import HEADING_STYLE_LEVELS, SectionGraphBuilder
from feature_keyword_extractor.schemas import ParsedDocument


Block = Union[Paragraph, Table]


class DocxParser:
    def parse(self, path: Path) -> ParsedDocument:
        document = Document(path)
        builder = SectionGraphBuilder(path)

        for block in _iter_block_items(document):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue
                level = _heading_level(block)
                if level is not None:
                    builder.add_heading(text, level)
                else:
                    builder.add_body(text)
            elif isinstance(block, Table):
                text = _table_text(block)
                builder.add_body(text)

        return builder.to_document()


def _iter_block_items(parent: DocumentType) -> Iterator[Block]:
    for child in parent.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _heading_level(paragraph: Paragraph) -> int | None:
    style_name = (paragraph.style.name or "").strip().lower()
    return HEADING_STYLE_LEVELS.get(style_name)


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        rows.append("\t".join(cell.text.strip() for cell in row.cells if cell.text.strip()))
    return "\n".join(row for row in rows if row)
