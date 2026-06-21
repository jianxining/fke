from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from feature_keyword_extractor.agents.schemas import BusinessCapability
from feature_keyword_extractor.schemas import ParsedDocument


class ResultAggregator:
    def aggregate(
        self,
        records: Iterable[Tuple[ParsedDocument, BusinessCapability]],
        keywords_by_feature: Dict[str, List[str]],
    ) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        for document, capability in records:
            feature = capability.feature_name
            item = result.setdefault(
                feature,
                {
                    "出现文档": [],
                    "合并关键词": [],
                    "相关标题": [],
                    "关联正文记号": [],
                },
            )
            _append_unique(item["出现文档"], document.doc_id)
            for keyword in keywords_by_feature.get(feature, []):
                _append_unique(item["合并关键词"], keyword)
            for path in _heading_paths(document, capability.related_heading_ids):
                _append_unique(item["相关标题"], path)
            for body_id in capability.related_body_ids:
                _append_unique(item["关联正文记号"], body_id)
        return result

    @staticmethod
    def write_json(result: Dict[str, dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def write_csv(result: Dict[str, dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["功能点", "关键词组", "来源文档", "相关标题", "关联正文记号"])
            writer.writeheader()
            for feature, item in result.items():
                writer.writerow(
                    {
                        "功能点": feature,
                        "关键词组": "|".join(item["合并关键词"]),
                        "来源文档": "|".join(item["出现文档"]),
                        "相关标题": "|".join(item["相关标题"]),
                        "关联正文记号": "|".join(item["关联正文记号"]),
                    }
                )


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _heading_paths(document: ParsedDocument, heading_ids: List[str]) -> List[str]:
    heading_by_id = document.heading_by_id
    paths = []
    for heading_id in heading_ids:
        node = heading_by_id.get(heading_id)
        if not node:
            continue
        paths.append(" > ".join([*node.ancestor_titles, node.title]))
    return paths
