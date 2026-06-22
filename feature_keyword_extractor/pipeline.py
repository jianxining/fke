from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from feature_keyword_extractor.agents.feature_agent import FeatureExtractionAgent
from feature_keyword_extractor.agents.schemas import BusinessCapability
from feature_keyword_extractor.exporter import ResultAggregator
from feature_keyword_extractor.merge import CapabilityMerger
from feature_keyword_extractor.nlp.keyword_extractor import KeywordExtractor
from feature_keyword_extractor.parsers import DocxParser, MarkdownParser
from feature_keyword_extractor.schemas import ParsedDocument


class ExtractionPipeline:
    def __init__(
        self,
        feature_agent: FeatureExtractionAgent,
        keyword_extractor: KeywordExtractor | None = None,
        merge_agent=None,
        merge_name_similarity_threshold: float = 0.4,
        merge_clusterer=None,
    ):
        self.feature_agent = feature_agent
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.merge_agent = merge_agent
        self.merge_name_similarity_threshold = merge_name_similarity_threshold
        self.merge_clusterer = merge_clusterer

    def run(self, input_dir: Path, output_dir: Path) -> Dict[str, dict]:
        documents = parse_documents(input_dir)
        records: List[Tuple[ParsedDocument, BusinessCapability]] = []
        keywords_by_feature: Dict[str, List[str]] = {}
        errors = []
        extraction_timing: List[dict] = []

        for document in documents:
            body_by_id = document.body_by_id
            t0 = time.perf_counter()
            try:
                response = self.feature_agent.extract(document)
            except Exception as exc:  # noqa: BLE001 - audit file should capture bad LLM batches.
                elapsed = time.perf_counter() - t0
                extraction_timing.append({"doc_id": document.doc_id, "elapsed_seconds": round(elapsed, 3), "success": False, "error": str(exc)})
                errors.append({"doc_id": document.doc_id, "error": str(exc)})
                continue
            elapsed = time.perf_counter() - t0
            extraction_timing.append({
                "doc_id": document.doc_id,
                "elapsed_seconds": round(elapsed, 3),
                "success": True,
                "capability_count": len(response.business_capabilities),
            })
            for capability in response.business_capabilities:
                records.append((document, capability))
                texts = [body_by_id[body_id].text for body_id in capability.related_body_ids if body_id in body_by_id]
                keywords = [item.term for item in self.keyword_extractor.extract(texts)]
                keywords_by_feature.setdefault(capability.feature_name, [])
                for keyword in keywords:
                    if keyword not in keywords_by_feature[capability.feature_name]:
                        keywords_by_feature[capability.feature_name].append(keyword)

        merge_decisions = []
        merge_timing: List[dict] = []
        if self.merge_agent:
            merge_result = CapabilityMerger(
                self.merge_agent,
                name_similarity_threshold=self.merge_name_similarity_threshold,
                clusterer=self.merge_clusterer,
            ).merge([capability for _, capability in records])
            merge_decisions = merge_result.decisions
            merge_timing = merge_result.timing
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

        timing = _build_timing(extraction_timing, merge_timing)
        (output_dir / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_timing(timing)
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


def _build_timing(extraction_timing: List[dict], merge_timing: List[dict]) -> dict:
    extraction_total = sum(item["elapsed_seconds"] for item in extraction_timing)
    merge_total = sum(item["elapsed_seconds"] for item in merge_timing)
    return {
        "extraction": {
            "calls": extraction_timing,
            "total_seconds": round(extraction_total, 3),
            "call_count": len(extraction_timing),
        },
        "merge": {
            "calls": merge_timing,
            "total_seconds": round(merge_total, 3),
            "call_count": len(merge_timing),
        },
        "llm_total_seconds": round(extraction_total + merge_total, 3),
    }


def _print_timing(timing: dict) -> None:
    ext = timing["extraction"]
    merge = timing["merge"]
    print(f"\n{'='*50}")
    print(f"LLM 调用计时统计")
    print(f"{'='*50}")
    print(f"功能点提取: {ext['call_count']} 次调用, 总耗时 {ext['total_seconds']:.3f}s")
    for item in ext["calls"]:
        status = "[OK]" if item["success"] else "[FAIL]"
        caps = f", {item['capability_count']}个功能点" if item["success"] else ""
        print(f"  {status} {item['doc_id']}: {item['elapsed_seconds']:.3f}s{caps}")
    print(f"功能点合并: {merge['call_count']} 次调用, 总耗时 {merge['total_seconds']:.3f}s")
    for item in merge["calls"]:
        merged = "合并" if item["should_merge"] else "跳过"
        print(f"  [{merged}] {item['pair'][0]} vs {item['pair'][1]}: {item['elapsed_seconds']:.3f}s (相似度={item['name_similarity']})")
    print(f"{'='*50}")
    print(f"LLM 总耗时: {timing['llm_total_seconds']:.3f}s")
    print(f"{'='*50}\n")
