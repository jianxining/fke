from feature_keyword_extractor.agents.schemas import BusinessCapability, LlmCapabilityResponse
from feature_keyword_extractor.cluster import Clusterer
from feature_keyword_extractor.exporter import ResultAggregator
from feature_keyword_extractor.merge import CapabilityMerger
from feature_keyword_extractor.schemas import BodyBlock, HeadingNode, ParsedDocument


def _parsed_doc(doc_id: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        heading_nodes=[
            HeadingNode(
                heading_id="H001",
                title="账号体系",
                parent_id=None,
                ancestor_titles=[],
                child_ids=["H002"],
                body_ids=[],
            ),
            HeadingNode(
                heading_id="H002",
                title="手机号登录",
                parent_id="H001",
                ancestor_titles=["账号体系"],
                child_ids=[],
                body_ids=["B001", "B002"],
            ),
            HeadingNode(
                heading_id="H003",
                title="风险策略",
                parent_id="H001",
                ancestor_titles=["账号体系"],
                child_ids=[],
                body_ids=["B003"],
            ),
        ],
        body_blocks=[
            BodyBlock(body_id="B001", text="用户输入手机号后获取短信验证码。"),
            BodyBlock(body_id="B002", text="验证码通过后写入登录态。"),
            BodyBlock(body_id="B003", text="命中风控策略时限制账号登录。"),
        ],
    )


def test_llm_capability_response_validates_required_body_references():
    response = LlmCapabilityResponse(
        business_capabilities=[
            BusinessCapability(
                feature_name="手机号验证码登录能力",
                description="支持通过手机号和短信验证码完成登录认证。",
                related_heading_ids=["H002"],
                related_body_ids=["B001", "B002"],
            )
        ]
    )

    response.validate_references(heading_ids={"H001", "H002"}, body_ids={"B001", "B002"})


def test_clusterer_groups_similar_heading_contexts_without_llm():
    parsed = _parsed_doc("需求文档A.docx")
    clusterer = Clusterer(similarity_threshold=0.15)

    clusters = clusterer.cluster(parsed.heading_nodes)

    assert clusters
    assert sorted(item.heading_id for cluster in clusters for item in cluster.items) == [
        "H001",
        "H002",
        "H003",
    ]
    assert all(cluster.cluster_id for cluster in clusters)


def test_result_aggregator_merges_documents_keywords_and_trace():
    parsed_a = _parsed_doc("需求文档A.docx")
    parsed_b = _parsed_doc("需求文档B.md")
    capability_a = BusinessCapability(
        feature_name="手机号验证码登录能力",
        description="支持通过手机号和短信验证码完成登录认证。",
        related_heading_ids=["H002"],
        related_body_ids=["B001", "B002"],
    )
    capability_b = BusinessCapability(
        feature_name="手机号验证码登录能力",
        description="支持通过手机号和短信验证码完成登录认证。",
        related_heading_ids=["H002"],
        related_body_ids=["B001"],
    )

    result = ResultAggregator().aggregate(
        [(parsed_a, capability_a), (parsed_b, capability_b)],
        {
            "手机号验证码登录能力": ["手机号", "短信验证码", "登录态"],
        },
    )

    item = result["手机号验证码登录能力"]
    assert item["出现文档"] == ["需求文档A.docx", "需求文档B.md"]
    assert item["合并关键词"] == ["手机号", "短信验证码", "登录态"]
    assert item["关联正文记号"] == ["B001", "B002"]
    assert "账号体系 > 手机号登录" in item["相关标题"]


def test_capability_merger_uses_llm_decision_to_canonicalize_similar_business_names():
    first = BusinessCapability(
        feature_name="手机号快捷登录能力",
        description="支持用户通过手机号和验证码完成登录。",
        related_heading_ids=["H002"],
        related_body_ids=["B001"],
    )
    second = BusinessCapability(
        feature_name="短信验证码登录能力",
        description="支持输入短信验证码完成登录认证。",
        related_heading_ids=["H003"],
        related_body_ids=["B002"],
    )

    class FakeMergeAgent:
        def judge(self, capabilities, similarity_hint=None):
            from feature_keyword_extractor.agents.schemas import MergeDecision

            return MergeDecision(
                should_merge=True,
                canonical_name="手机号验证码登录能力",
                confidence=0.91,
                reason="二者均为手机号短信验证码登录。",
            )

    merged = CapabilityMerger(FakeMergeAgent(), confidence_threshold=0.8).merge([first, second])

    assert len(merged.capabilities) == 1
    assert merged.capabilities[0].feature_name == "手机号验证码登录能力"
    assert merged.capabilities[0].related_heading_ids == ["H002", "H003"]
    assert merged.capabilities[0].related_body_ids == ["B001", "B002"]
    assert merged.decisions[0]["合并结果"] is True


def test_capability_merger_accepts_null_canonical_name_when_llm_rejects_merge():
    first = BusinessCapability(
        feature_name="手机号验证码登录能力",
        description="支持用户通过手机号和验证码完成登录。",
        related_heading_ids=["H002"],
        related_body_ids=["B001"],
    )
    second = BusinessCapability(
        feature_name="SIM卡状态识别能力",
        description="识别无卡、换卡、停机和漫游等状态。",
        related_heading_ids=["H003"],
        related_body_ids=["B002"],
    )

    class FakeMergeAgent:
        def judge(self, capabilities, similarity_hint=None):
            from feature_keyword_extractor.agents.schemas import MergeDecision

            return MergeDecision(
                should_merge=False,
                canonical_name=None,
                confidence=0.2,
                reason="二者属于不同业务能力。",
            )

    merged = CapabilityMerger(FakeMergeAgent(), confidence_threshold=0.8, name_similarity_threshold=0.0).merge([first, second])

    assert [item.feature_name for item in merged.capabilities] == [
        "手机号验证码登录能力",
        "SIM卡状态识别能力",
    ]
    assert merged.decisions[0]["合并结果"] is False


def test_capability_merger_skips_pairs_below_name_similarity_threshold():
    first = BusinessCapability(
        feature_name="手机号验证码登录能力",
        description="支持用户通过手机号和验证码完成登录。",
        related_heading_ids=["H002"],
        related_body_ids=["B001"],
    )
    second = BusinessCapability(
        feature_name="SIM卡状态识别能力",
        description="识别无卡、换卡、停机和漫游等状态。",
        related_heading_ids=["H003"],
        related_body_ids=["B002"],
    )

    class FakeMergeAgent:
        def judge(self, capabilities, similarity_hint=None):
            raise AssertionError("LLM judge should not be called for dissimilar names")

    merged = CapabilityMerger(FakeMergeAgent(), confidence_threshold=0.8, name_similarity_threshold=0.3).merge([first, second])

    assert len(merged.capabilities) == 2
    assert merged.decisions == []
