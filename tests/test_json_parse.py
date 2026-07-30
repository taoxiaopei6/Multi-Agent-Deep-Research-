"""测试 JSON 解析和 LLM 输出修复功能。"""
import json
from app.mult_agents.nodes._utils import _extract_json_block, _load_json, _repair_json


class TestJsonExtraction:
    """测试从 LLM 输出中提取 JSON 的能力。"""

    def test_plain_json(self):
        """标准 JSON 直接通过。"""
        text = '{"route": "multiagent", "reason": "需要调研"}'
        result = _extract_json_block(text)
        assert json.loads(result)["route"] == "multiagent"

    def test_json_in_markdown_block(self):
        """```json ... ``` 包装的情况。"""
        text = '```json\n{"route": "direct", "reason": "简单问答"}\n```'
        result = _extract_json_block(text)
        assert json.loads(result)["route"] == "direct"

    def test_json_with_pre_text(self):
        """前置文本 + JSON。"""
        text = '根据分析，结果如下：\n{"route": "multiagent", "reason": "复杂问题"}'
        result = _extract_json_block(text)
        assert json.loads(result)["route"] == "multiagent"

    def test_multiple_json_blocks(self):
        """多个 JSON 块，取最外层完整的。"""
        text = '{"a": 1} some text {"b": 2, "c": {"d": 3}}'
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert "b" in parsed
        assert parsed["b"] == 2

    def test_malformed_markdown(self):
        """残缺的 markdown 标记。"""
        text = '```\n{"key": "value"}\n```'
        result = _extract_json_block(text)
        assert json.loads(result)["key"] == "value"

    def test_empty_input(self):
        """空输入或纯空白 → 空字符串。"""
        assert _extract_json_block("") == ""
        assert _extract_json_block("  ") == ""

    def test_no_json(self):
        """没有 JSON 时返回原文。"""
        text = "你好，这是一个普通回答"
        result = _extract_json_block(text)
        assert result == text

    def test_deep_nested_json(self):
        """深层嵌套的 JSON。"""
        text = '{"level1": {"level2": {"level3": {"level4": "deep"}}}}'
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["level3"]["level4"] == "deep"


class TestJsonRepair:
    """测试 JSON 修复功能。"""

    def test_trailing_comma_in_object(self):
        """对象尾部逗号。"""
        text = '{"a": 1, "b": 2,}'
        result = _repair_json(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        """数组尾部逗号。"""
        text = '{"items": [1, 2, 3,]}'
        result = _repair_json(text)
        assert json.loads(result) == {"items": [1, 2, 3]}

    def test_nested_trailing_commas(self):
        """嵌套结构中的尾部逗号。"""
        text = '{"data": {"list": [1, 2,], "name": "test",},}'
        result = _repair_json(text)
        assert json.loads(result) == {"data": {"list": [1, 2], "name": "test"}}

    def test_control_characters(self):
        """控制字符。"""
        text = '{"text": "hello\x00world\x1f"}'
        result = _repair_json(text)
        assert json.loads(result) == {"text": "helloworld"}


class TestLoadJson:
    """测试完整的 JSON 加载流程。"""

    def test_load_simple(self):
        text = '{"score": 4.5, "items": ["a", "b"]}'
        fallback = {"score": 0}
        result = _load_json(text, fallback)
        assert result["score"] == 4.5
        assert len(result["items"]) == 2

    def test_load_with_fix(self):
        """需要修复的 JSON。"""
        text = '{"score": 4.5, "items": ["a", "b",]}'
        fallback = {"score": 0}
        result = _load_json(text, fallback)
        assert result["score"] == 4.5
        assert len(result["items"]) == 2

    def test_load_fallback(self):
        """完全无法解析时返回 fallback。"""
        text = "完全不是 JSON 的文本"
        fallback = {"score": 0, "items": []}
        result = _load_json(text, fallback)
        assert result == fallback

    def test_load_with_markdown(self):
        """Markdown 包装的 JSON。"""
        text = '```\n{"score": 3.0, "name": "test"}\n```'
        fallback = {"score": 0}
        result = _load_json(text, fallback)
        assert result["score"] == 3.0
        assert result["name"] == "test"
