"""意图路由节点：intent_node 和 direct_answer_node。"""

import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from ..state import ResearchState
from ._utils import (
    _invoke_json_agent, detect_intent, with_memory_context,
    _last_content, colorize, auto_record_trace,
)


logger = logging.getLogger("mult_agents")


def intent_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("start | agent=%s", "[intent]", agent_name)
    rule_route = detect_intent(state["query"])
    prompt = (
        f"用户问题：{state['query']}\n"
        f"规则引擎初判：{rule_route}\n"
        "请输出 JSON：{\"route\":\"direct|multiagent\",\"reason\":\"...\"}"
    )
    payload, content, messages = _invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "intent",
        {"route": rule_route, "reason": "rule"},
    )
    route = str(payload.get("route", rule_route)).strip().lower()
    if route not in {"direct", "multiagent"}:
        route = rule_route
    logger.info("%s 路由: %s", "[intent]", route)
    return {"intent": route, "draft": content, "messages": messages}


def direct_answer_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("start | agent=%s", "[direct_answer]", agent_name)
    start_time = time.time()
    prompt = f"用户问题：{state['query']}"
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    content = _last_content(result).strip()
    logger.info("%s 输出: %s", colorize("[direct_answer]", "yellow"), content.replace("\n", " ")[:400])
    trace = auto_record_trace(state, "direct_answer", start_time,
                              output_summary={"answer_length": len(content)})
    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "messages": [human, result["messages"][-1]],
        **trace,
    }
