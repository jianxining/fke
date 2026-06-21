from __future__ import annotations

import json
from typing import Any, List

from feature_keyword_extractor.agents.feature_agent import _extract_json
from feature_keyword_extractor.agents.schemas import BusinessCapability, MergeDecision


MERGE_PROMPT = """你是产品需求分析专家。请判断以下功能点是否表达同一项业务能力。

判断要求：
1. 最终判断必须基于业务语义。
2. 字面相似但业务不同，不要合并。
3. 表述不同但业务能力相同，应合并。
4. 如果合并，请给出统一后的功能点名称。

严格输出 JSON：
{{
  "合并结果": true,
  "合并后名称": "手机号验证码登录能力",
  "置信度": 0.92,
  "理由": "二者均描述通过手机号和短信验证码完成登录认证。"
}}

输入：
{payload}
"""


class MergeAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def judge(self, capabilities: List[BusinessCapability], similarity_hint: dict | None = None) -> MergeDecision:
        payload = {
            "capabilities": [cap.model_dump(by_alias=True) for cap in capabilities],
            "similarity_hint": similarity_hint or {},
        }
        result = self.llm.invoke(MERGE_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False, indent=2)))
        raw = getattr(result, "content", result)
        return MergeDecision.model_validate_json(_extract_json(raw))
