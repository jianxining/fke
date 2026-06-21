from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from feature_keyword_extractor.parsers.base import SectionGraphBuilder
from feature_keyword_extractor.schemas import ParsedDocument


class MarkdownParser:
    def parse(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        tokens = MarkdownIt().parse(text)
        builder = SectionGraphBuilder(path)
        index = 0

        while index < len(tokens):
            token = tokens[index]
            if token.type == "heading_open":
                level = int(token.tag[1:])
                content = _inline_content(tokens, index + 1)
                if level <= 3:
                    builder.add_heading(content, level)
                    index = _skip_until(tokens, index, "heading_close")
                else:
                    body_parts = [content]
                    heading_close = _skip_until(tokens, index, "heading_close")
                    next_index = heading_close + 1
                    if next_index < len(tokens) and tokens[next_index].type == "paragraph_open":
                        following = _following_inline_content(tokens, next_index)
                        if following:
                            body_parts.append(following)
                        index = _skip_block(tokens, next_index)
                    else:
                        index = heading_close
                    builder.add_body("\n".join(part for part in body_parts if part))
            elif token.type in {"paragraph_open", "bullet_list_open", "ordered_list_open"}:
                content = _following_inline_content(tokens, index)
                if content:
                    builder.add_body(content)
                index = _skip_block(tokens, index)
            elif token.type == "fence":
                builder.add_body(token.content)
            elif token.type == "inline" and token.content.strip():
                builder.add_body(token.content)
            index += 1

        return builder.to_document()


def _inline_content(tokens, index: int) -> str:
    if index < len(tokens) and tokens[index].type == "inline":
        return tokens[index].content.strip()
    return ""


def _following_inline_content(tokens, index: int) -> str:
    parts = []
    cursor = index
    while cursor < len(tokens):
        token = tokens[cursor]
        if token.type == "inline" and token.content.strip():
            parts.append(token.content.strip())
        elif token.type in {"heading_close", "paragraph_close", "bullet_list_close", "ordered_list_close"}:
            break
        cursor += 1
    return "\n".join(parts)


def _skip_until(tokens, index: int, token_type: str) -> int:
    cursor = index
    while cursor < len(tokens) and tokens[cursor].type != token_type:
        cursor += 1
    return cursor


def _skip_block(tokens, index: int) -> int:
    close_map = {
        "paragraph_open": "paragraph_close",
        "bullet_list_open": "bullet_list_close",
        "ordered_list_open": "ordered_list_close",
    }
    close_type = close_map.get(tokens[index].type)
    return _skip_until(tokens, index, close_type) if close_type else index
