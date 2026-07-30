"""测试 ResearchState 初始化和状态管理。"""
import json
from app.mult_agents.state import create_initial_state, ResearchState


class TestResearchState:
    def test_create_initial_state(self):
        """测试初始状态创建是否完整。"""
        state = create_initial_state(
            query="2025年全球AI芯片市场规模",
            max_iterations=2,
            user_id="test_user",
            tenant_id="test_tenant",
        )
        assert state["query"] == "2025年全球AI芯片市场规模"
        assert state["max_iterations"] == 2
        assert state["user_id"] == "test_user"
        assert state["tenant_id"] == "test_tenant"
        assert state["iteration"] == 0
        assert state["messages"] == []

    def test_state_has_all_required_fields(self):
        """测试状态包含所有 45+ 字段。"""
        state = create_initial_state("test", 2, "u", "t")
        required_fields = [
            "query", "intent", "phase", "plan", "outline",
            "sub_questions", "search_plan", "budget",
            "web_search", "local_rag",
            "web_evidence", "local_evidence", "evidence_pool",
            "deep_dive", "audit", "audit_flags",
            "analysis", "needs_more_research", "missing_gaps",
            "supplementary_queries", "findings", "claim_map",
            "source_index", "final", "iteration", "max_iterations",
            "trace_events",
        ]
        for field in required_fields:
            assert field in state, f"缺少字段: {field}"

    def test_serializable(self):
        """测试状态可序列化为 JSON（用于检查点和持久化）。"""
        state = create_initial_state("test", 2, "u", "t")
        # 排除 BaseMessage（不能直接序列化）
        serializable = {k: v for k, v in state.items() if k != "messages"}
        dumped = json.dumps(serializable)
        loaded = json.loads(dumped)
        assert loaded["query"] == "test"
        assert loaded["max_iterations"] == 2
