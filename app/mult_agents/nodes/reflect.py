"""反思节点：reflect_node (ResearchPlanner)，生成补搜计划。"""

import json
import logging

from ..state import ResearchState
from ._utils import _invoke_json_agent, log_inputs, colorize


logger = logging.getLogger("mult_agents")


def reflect_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("start | agent=%s", "[reflect]", agent_name)

    missing_gaps = state.get("missing_gaps", [])
    log_inputs("reflect", agent_name, {"missing_gaps": str(missing_gaps)})

    fallback = {
        "reflection_summary": "默认补搜",
        "supplementary_queries": [{"section_id": "gap_1", "query": state["query"], "source_preference": "hybrid", "reason": "fallback"}]
    }

    prompt = (
        f"分析师指出当前证据不足以完全回答问题，存在以下信息缺口：\n{json.dumps(missing_gaps)}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []))}\n"
        f"已执行过的搜索计划：\n{json.dumps(state.get('search_plan', []))}\n"
        f"已执行过的补搜计划：\n{json.dumps(state.get('supplementary_queries', []))}\n\n"
        "请生成新的补搜计划以填补缺口。"
    )

    payload, content, messages = _invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "reflect",
        fallback,
    )

    return {
        "iteration": state.get("iteration", 0) + 1,
        "supplementary_queries": payload.get("supplementary_queries", fallback["supplementary_queries"]),
        "messages": messages,
    }
