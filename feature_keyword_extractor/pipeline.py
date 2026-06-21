from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from feature_keyword_extractor.agents.feature_agent import FeatureExtractionAgent
from feature_keyword_extractor.agents.schemas import BusinessCapability
from feature_keyword_extractor.cluster import Clusterer
from feature_keyword_extractor.exporter import ResultAggregator
from feature_keyword_extractor.merge import CapabilityMerger
from feature_keyword_extractor.nlp.keyword_extractor import KeywordExtractor
from feature_keyword_extractor.parsers import DocxParser, MarkdownParser
from feature_keyword_extractor.schemas import ParsedDocument


class ExtractionPipeline:
    def __init__(
        self,
        feature_agent: FeatureExtractionAgent,
        clusterer: Clusterer | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        merge_agent=None,
    ):
        self.feature_agent = feature_agent
        self.clusterer = clusterer or Clusterer()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.merge_agent = merge_agent

    def run(self, input_dir: Path, output_dir: Path) -> Dict[str, dict]:
        documents = parse_documents(input_dir)
        records: List[Tuple[ParsedDocument, BusinessCapability]] = []
        keywords_by_feature: Dict[str, List[str]] = {}
        errors = []

        for document in documents:
            body_by_id = document.body_by_id
            for cluster in self.clusterer.cluster(document.heading_nodes):
                try:
                    response = self.feature_agent.extract(document, cluster)
                except Exception as exc:  # noqa: BLE001 - audit file should capture bad LLM batches.
                    errors.append({"doc_id": document.doc_id, "cluster_id": cluster.cluster_id, "error": str(exc)})
                    continue
                for capability in response.business_capabilities:
                    records.append((document, capability))
                    texts = [body_by_id[body_id].text for body_id in capability.related_body_ids if body_id in body_by_id]
                    keywords = [item.term for item in self.keyword_extractor.extract(texts)]
                    keywords_by_feature.setdefault(capability.feature_name, [])
                    for keyword in keywords:
                        if keyword not in keywords_by_feature[capability.feature_name]:
                            keywords_by_feature[capability.feature_name].append(keyword)

        merge_decisions = []
        if self.merge_agent:
            merge_result = CapabilityMerger(self.merge_agent).merge([capability for _, capability in records])
            merge_decisions = merge_result.decisions
            records = [
                (document, _with_feature_name(capability, merge_result.canonical_by_index[index]))
                for index, (document, capability) in enumerate(records)
            ]
            keywords_by_feature = self._keywords_for_merged_capabilities(records)

        aggregator = ResultAggregator()
        result = aggregator.aggregate(records, keywords_by_feature)
        output_dir.mkdir(parents=True, exist_ok=True)
        aggregator.write_csv(result, output_dir / "feature_keywords.csv")
        aggregator.write_json(result, output_dir / "feature_keywords.json")
        _write_audit_files(output_dir, documents, errors, merge_decisions, result)
        return result

    def _keywords_for_merged_capabilities(
        self,
        records: List[Tuple[ParsedDocument, BusinessCapability]],
    ) -> Dict[str, List[str]]:
        keywords_by_feature: Dict[str, List[str]] = {}
        for document, capability in records:
            body_by_id = document.body_by_id
            texts = [body_by_id[body_id].text for body_id in capability.related_body_ids if body_id in body_by_id]
            for keyword_score in self.keyword_extractor.extract(texts):
                keywords_by_feature.setdefault(capability.feature_name, [])
                if keyword_score.term not in keywords_by_feature[capability.feature_name]:
                    keywords_by_feature[capability.feature_name].append(keyword_score.term)
        return keywords_by_feature


def parse_documents(input_dir: Path) -> List[ParsedDocument]:
    documents = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() == ".docx":
            documents.append(DocxParser().parse(path))
        elif path.suffix.lower() in {".md", ".markdown"}:
            documents.append(MarkdownParser().parse(path))
    return documents


def _with_feature_name(capability: BusinessCapability, feature_name: str) -> BusinessCapability:
    if capability.feature_name == feature_name:
        return capability
    return BusinessCapability(
        feature_name=feature_name,
        description=capability.description,
        related_heading_ids=capability.related_heading_ids,
        related_body_ids=capability.related_body_ids,
    )


def _write_audit_files(
    output_dir: Path,
    documents: List[ParsedDocument],
    errors: list[dict],
    merge_decisions: list[dict],
    result: Dict[str, dict],
) -> None:
    sections = [document.model_dump() for document in documents]
    (output_dir / "sections.json").write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "merge_decisions.json").write_text(
        json.dumps(merge_decisions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "capability_index.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
