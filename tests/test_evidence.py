"""测试证据评分、去重、过滤功能。"""
from app.mult_agents.nodes._evidence import (
    _score_evidence,
    _dedupe_evidence_by_url,
    _is_official_domain,
    _is_bad_web_domain,
    _estimate_relevance,
)


class TestDomainClassification:
    """测试域名分类。"""

    def test_official_gov(self):
        assert _is_official_domain("www.gov.cn") is True

    def test_official_edu(self):
        assert _is_official_domain("www.tsinghua.edu.cn") is True

    def test_not_official(self):
        assert _is_official_domain("www.example.com") is False

    def test_bad_domain(self):
        assert _is_bad_web_domain("www.doc88.com") is True

    def test_normal_domain(self):
        assert _is_bad_web_domain("www.bbc.com") is False


class TestEvidenceScoring:
    """测试证据可信度评分。"""

    def test_local_evidence(self):
        """本地知识库证据 → 0.92，含 breakdown。"""
        score, reason, breakdown = _score_evidence({"source_type": "local"})
        assert score == 0.92
        assert breakdown is not None
        assert "authority" in breakdown
        assert breakdown["authority"] >= 0.9

    def test_official_domain(self):
        """官方域名 → 0.88，含 breakdown。"""
        score, reason, breakdown = _score_evidence({"source_type": "web", "domain": "www.gov.cn"})
        assert score == 0.88
        assert breakdown is not None
        assert breakdown["authority"] >= 0.9

    def test_media_domain(self):
        """知名媒体 → 0.72。"""
        score, reason, breakdown = _score_evidence({"source_type": "web", "domain": "reuters.com"})
        assert score == 0.72
        assert breakdown is not None

    def test_unknown_domain(self):
        """普通域名 → 0.58。"""
        score, reason, breakdown = _score_evidence({"source_type": "web", "domain": "example.com"})
        assert score == 0.58
        assert breakdown is not None

    def test_no_domain(self):
        """无域名 → 0.45。"""
        score, reason, breakdown = _score_evidence({"source_type": "web", "domain": ""})
        assert score == 0.45
        assert breakdown is not None

    def test_xinhuanet_media(self):
        score, reason, breakdown = _score_evidence({"source_type": "web", "domain": "xinhuanet.com"})
        assert score == 0.72
        assert breakdown is not None


class TestEvidenceDedup:
    """测试 URL 级别去重。"""

    def test_dedup_same_url_keep_highest_score(self):
        """同一 URL 保留最高分。"""
        items = [
            {"source_id": "WEB1_1-1", "url": "https://example.com/a", "reliability_score": 0.58},
            {"source_id": "WEB2_1-2", "url": "https://example.com/a", "reliability_score": 0.88},
            {"source_id": "WEB1_2-1", "url": "https://example.com/b", "reliability_score": 0.72},
        ]
        result = _dedupe_evidence_by_url(items)
        assert len(result) == 2  # 去重后剩 2 条
        ids = [r["source_id"] for r in result]
        assert "WEB2_1-2" in ids  # 高分被保留
        assert "WEB1_1-1" not in ids  # 低分被去掉

    def test_dedup_no_url_kept(self):
        """无 URL 的条目（本地证据）不去重。"""
        items = [
            {"source_id": "LOC1_1-1", "url": "", "reliability_score": 0.92},
            {"source_id": "LOC2_1-2", "url": "", "reliability_score": 0.92},
        ]
        result = _dedupe_evidence_by_url(items)
        assert len(result) == 2

    def test_dedup_empty_list(self):
        assert _dedupe_evidence_by_url([]) == []


class TestRelevanceEstimation:
    """测试查询与文本的相关性估算。"""

    def test_high_relevance(self):
        score = _estimate_relevance("AI芯片市场规模", "2025年AI芯片市场规模达到1200亿美元")
        assert score > 0.3

    def test_no_relevance(self):
        score = _estimate_relevance("AI芯片市场规模", "今天的天气很好")
        assert score == 0.0

    def test_partial_match(self):
        score = _estimate_relevance("NVIDIA GPU 市场份额", "NVIDIA在AI训练芯片市场占据80%份额")
        assert score > 0

    def test_empty_query(self):
        score = _estimate_relevance("", "一些文本内容")
        assert score == 0.0
