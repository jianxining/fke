from __future__ import annotations

from typing import List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BusinessCapability(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    feature_name: str = Field(alias="功能点")
    description: str = Field(alias="业务描述")
    related_heading_ids: List[str] = Field(alias="相关标题")
    related_body_ids: List[str] = Field(alias="关联正文")

    @model_validator(mode="after")
    def _non_empty_references(self):
        if not self.feature_name.strip():
            raise ValueError("功能点不能为空")
        if not self.related_heading_ids:
            raise ValueError("相关标题不能为空")
        if not self.related_body_ids:
            raise ValueError("关联正文不能为空")
        return self


class LlmCapabilityResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    business_capabilities: List[BusinessCapability] = Field(alias="business_capabilities")

    def validate_references(self, heading_ids: Set[str], body_ids: Set[str]) -> None:
        for capability in self.business_capabilities:
            missing_headings = set(capability.related_heading_ids) - heading_ids
            missing_bodies = set(capability.related_body_ids) - body_ids
            if missing_headings or missing_bodies:
                raise ValueError(
                    f"LLM returned unknown references: headings={sorted(missing_headings)}, "
                    f"bodies={sorted(missing_bodies)}"
                )


class MergeDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    should_merge: bool = Field(alias="合并结果")
    canonical_name: Optional[str] = Field(default=None, alias="合并后名称")
    confidence: float = Field(alias="置信度")
    reason: str = Field(alias="理由")

    @model_validator(mode="after")
    def _canonical_required_when_merging(self):
        if self.should_merge and not self.canonical_name:
            raise ValueError("合并结果为 true 时，合并后名称不能为空")
        return self
