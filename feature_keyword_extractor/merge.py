from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from feature_keyword_extractor.agents.schemas import BusinessCapability


@dataclass
class MergeResult:
    capabilities: List[BusinessCapability]
    decisions: List[dict]
    canonical_by_index: Dict[int, str]


class CapabilityMerger:
    def __init__(self, merge_agent, confidence_threshold: float = 0.8):
        self.merge_agent = merge_agent
        self.confidence_threshold = confidence_threshold

    def merge(self, capabilities: List[BusinessCapability]) -> MergeResult:
        if len(capabilities) <= 1:
            return MergeResult(
                capabilities=capabilities,
                decisions=[],
                canonical_by_index={index: capability.feature_name for index, capability in enumerate(capabilities)},
            )

        parent = list(range(len(capabilities)))
        canonical_names: Dict[int, str] = {}
        decisions: List[dict] = []

        for left, right in self._candidate_pairs(capabilities):
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
            decision = self.merge_agent.judge([capabilities[left], capabilities[right]], similarity_hint=hint)
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
        return MergeResult(capabilities=merged, decisions=decisions, canonical_by_index=canonical_by_index)

    @staticmethod
    def _candidate_pairs(capabilities: List[BusinessCapability]) -> List[Tuple[int, int]]:
        return [(left, right) for left in range(len(capabilities)) for right in range(left + 1, len(capabilities))]

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
