from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from feature_keyword_extractor.agents.schemas import LlmCapabilityResponse
from feature_keyword_extractor.schemas import HeadingNode, ParsedDocument


FEATURE_PROMPT = """你是产品策划文档分析专家。请根据标题依赖图和正文记号，判断哪些标题共同描述同一项原子业务能力。

要求：
1. 不要按标题层级机械拆分。
2. 结合标题的父子依赖、祖先链、子标题、挂靠正文判断业务能力边界。
3. 每个业务能力必须返回业务能力名称、业务描述、相关标题ID、关联正文ID。
4. 功能点名称使用“对象 + 动作/能力”的业务能力命名方式。
5. 关联正文ID必须来自输入，不能编造。

严格输出 JSON：
{{
  "business_capabilities": [
    {{
      "功能点": "手机号验证码登录能力",
      "业务描述": "支持用户通过手机号和短信验证码完成登录认证。",
      "相关标题": ["H002"],
      "关联正文": ["B001", "B002"]
    }}
  ]
}}

输入：
{payload}
"""


class FeatureExtractionAgent:
    def __init__(self, llm: Any | None = None):
        self.llm = llm or self._build_default_llm()

    def extract(self, document: ParsedDocument, heading_nodes: List[HeadingNode] | None = None) -> LlmCapabilityResponse:
        nodes = heading_nodes if heading_nodes is not None else document.heading_nodes
        payload = self._payload(document, nodes)
        raw = self._invoke(FEATURE_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False, indent=2)))
        response = LlmCapabilityResponse.model_validate_json(_extract_json(raw))
        response.validate_references(
            heading_ids={node.heading_id for node in document.heading_nodes},
            body_ids={block.body_id for block in document.body_blocks},
        )
        return response

    def _invoke(self, prompt: str) -> str:
        result = self.llm.invoke(prompt)
        return getattr(result, "content", result)

    @staticmethod
    def _payload(document: ParsedDocument, heading_nodes: List[HeadingNode]) -> Dict[str, Any]:
        heading_by_id = document.heading_by_id
        body_by_id = document.body_by_id
        nodes = []
        body_ids = set()
        for node in heading_nodes:
            body_ids.update(node.body_ids)
            nodes.append(
                {
                    "heading_id": node.heading_id,
                    "title": node.title,
                    "parent_id": node.parent_id,
                    "parent_title": heading_by_id[node.parent_id].title if node.parent_id else None,
                    "ancestor_titles": node.ancestor_titles,
                    "child_ids": node.child_ids,
                    "body_ids": node.body_ids,
                    "body_preview": [
                        {"body_id": body_id, "text": body_by_id[body_id].text[:500]}
                        for body_id in node.body_ids
                        if body_id in body_by_id
                    ],
                }
            )
        return {
            "doc_id": document.doc_id,
            "heading_dependency_graph": nodes,
            "available_body_ids": sorted(body_ids),
        }

    @staticmethod
    def _build_default_llm() -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Missing langchain-openai. Install dependencies before using LLM agents.") from exc

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for LLM feature extraction.")
        return ChatOpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseekV4.pro"),
            temperature=0,
        )


def _extract_json(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain JSON.")
    return text[start : end + 1]
