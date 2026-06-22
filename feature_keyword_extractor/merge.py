from __future__ import annotations

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from feature_keyword_extractor.agents.schemas import BusinessCapability


@dataclass
class MergeResult:
    capabilities: List[BusinessCapability]
    decisions: List[dict]
    canonical_by_index: Dict[int, str]
    timing: List[dict] = field(default_factory=list)


class CapabilityMerger:
    def __init__(
        self,
        merge_agent,
        confidence_threshold: float = 0.8,
        name_similarity_threshold: float = 0.3,
        clusterer=None,
    ):
        self.merge_agent = merge_agent
        self.confidence_threshold = confidence_threshold
        self.name_similarity_threshold = name_similarity_threshold
        self.clusterer = clusterer

    def merge(self, capabilities: List[BusinessCapability]) -> MergeResult:
        if len(capabilities) <= 1:
            return MergeResult(
                capabilities=capabilities,
                decisions=[],
                canonical_by_index={index: capability.feature_name for index, capability in enumerate(capabilities)},
            )

        candidate_pairs = self._candidate_pairs(capabilities)
        parent = list(range(len(capabilities)))
        canonical_names: Dict[int, str] = {}
        decisions: List[dict] = []
        timing: List[dict] = []

        for left, right in candidate_pairs:
            hint = {
                "name_similarity": round(
                    SequenceMatcher(
                        None,
                        capabilities[left].feature_name,
                        capabilities[right].feature_name,
                    ).ratio(),
                    4,
                )
            }
            t0 = time.perf_counter()
            decision = self.merge_agent.judge([capabilities[left], capabilities[right]], similarity_hint=hint)
            elapsed = time.perf_counter() - t0
            timing.append({
                "pair": [capabilities[left].feature_name, capabilities[right].feature_name],
                "name_similarity": hint["name_similarity"],
                "elapsed_seconds": round(elapsed, 3),
                "should_merge": decision.should_merge,
            })
            decisions.append(decision.model_dump(by_alias=True))
            if decision.should_merge and decision.confidence >= self.confidence_threshold:
                left_root = self._find(parent, left)
                right_root = self._find(parent, right)
                parent[right_root] = left_root
                canonical_names[left_root] = decision.canonical_name

        groups: Dict[int, List[BusinessCapability]] = {}
        for index, capability in enumerate(capabilities):
            root = self._find(parent, index)
            groups.setdefault(root, []).append(capability)

        merged = [self._merge_group(root, items, canonical_names) for root, items in groups.items()]
        canonical_by_index = {}
        for index in range(len(capabilities)):
            root = self._find(parent, index)
            canonical_by_index[index] = canonical_names.get(root, capabilities[root].feature_name)
        return MergeResult(capabilities=merged, decisions=decisions, canonical_by_index=canonical_by_index, timing=timing)

    def _candidate_pairs(self, capabilities: List[BusinessCapability]) -> List[Tuple[int, int]]:
        if self.clusterer is not None:
            return self._cluster_candidate_pairs(capabilities)
        return self._similarity_candidate_pairs(capabilities)

    def _cluster_candidate_pairs(self, capabilities: List[BusinessCapability]) -> List[Tuple[int, int]]:
        names = [cap.feature_name for cap in capabilities]
        groups = self.clusterer.cluster_texts(names)
        pairs = []
        for group in groups:
            for i, left in enumerate(group):
                for right in group[i + 1:]:
                    pairs.append((left, right))
        return pairs

    def _similarity_candidate_pairs(self, capabilities: List[BusinessCapability]) -> List[Tuple[int, int]]:
        pairs = []
        for left in range(len(capabilities)):
            for right in range(left + 1, len(capabilities)):
                ratio = SequenceMatcher(
                    None,
                    capabilities[left].feature_name,
                    capabilities[right].feature_name,
                ).ratio()
                if ratio >= self.name_similarity_threshold:
                    pairs.append((left, right))
        return pairs

    @classmethod
    def _find(cls, parent: List[int], index: int) -> int:
        if parent[index] != index:
            parent[index] = cls._find(parent, parent[index])
        return parent[index]

    @staticmethod
    def _merge_group(
        root: int,
        capabilities: List[BusinessCapability],
        canonical_names: Dict[int, str],
    ) -> BusinessCapability:
        if len(capabilities) == 1:
            return capabilities[0]
        heading_ids: List[str] = []
        body_ids: List[str] = []
        descriptions = []
        for capability in capabilities:
            descriptions.append(capability.description)
            for heading_id in capability.related_heading_ids:
                if heading_id not in heading_ids:
                    heading_ids.append(heading_id)
            for body_id in capability.related_body_ids:
                if body_id not in body_ids:
                    body_ids.append(body_id)
        return BusinessCapability(
            feature_name=canonical_names.get(root, capabilities[0].feature_name),
            description="；".join(descriptions),
            related_heading_ids=heading_ids,
            related_body_ids=body_ids,
        )
