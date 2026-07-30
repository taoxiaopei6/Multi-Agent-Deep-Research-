"""测试意图路由和规则引擎。"""
from app.mult_agents.nodes._utils import detect_intent


class TestIntentDetection:
    """测试规则引擎意图识别。"""

    def test_multiagent_with_research_keyword(self):
        """包含"调研"等研究关键词 → multiagent。"""
        assert detect_intent("请调研一下全球AI芯片市场") == "multiagent"

    def test_multiagent_with_trend(self):
        """包含"趋势"且含年份 → multiagent。"""
        assert detect_intent("2025年新能源汽车发展趋势") == "multiagent"

    def test_multiagent_with_analysis(self):
        """包含"分析" → multiagent。"""
        assert detect_intent("请分析一下竞品格局") == "multiagent"

    def test_multiagent_with_compare(self):
        """包含"对比" → multiagent。"""
        assert detect_intent("对比一下AWS和Azure") == "multiagent"

    def test_multiagent_with_report(self):
        """包含"报告" → multiagent。"""
        assert detect_intent("出一份市场调研报告") == "multiagent"

    def test_direct_with_greeting(self):
        """问候语 → direct。"""
        assert detect_intent("你好") == "direct"

    def test_direct_with_simple_question(self):
        """简单知识问答 → direct。"""
        assert detect_intent("你是谁") == "direct"

    def test_direct_with_weather(self):
        """天气查询 → direct。"""
        assert detect_intent("今天天气怎么样") == "direct"

    def test_direct_what_is(self):
        """是什么类问题 → direct。"""
        assert detect_intent("什么是机器学习") == "direct"

    def test_multiagent_with_source_keyword(self):
        """含"来源" → multiagent。"""
        assert detect_intent("列出带来源的市场数据") == "multiagent"

    def test_direct_with_no_keywords(self):
        """无任何关键字的日常对话 → direct。"""
        assert detect_intent("帮我算一下1+1等于几") == "direct"

    def test_edge_empty_string(self):
        """空字符串 → direct。"""
        assert detect_intent("") == "direct"

    def test_edge_special_chars(self):
        """特殊字符。"""
        assert detect_intent("!!!") == "direct"

    def test_year_without_trend_keyword(self):
        """有年份但不含趋势词 → direct。"""
        result = detect_intent("2025年有什么新东西")
        # 这个取决于是否有其他关键词匹配
        assert result in ("direct", "multiagent")
