"""规划节点：plan_node 及问题拆解相关的辅助函数。"""

import json
import logging

from ..state import ResearchState
from ._utils import _invoke_json_agent, log_inputs, colorize


logger = logging.getLogger("mult_agents")


def _default_plan(state: ResearchState) -> dict:
    return {
        "objective": state["query"],
        "sub_questions": [state["query"]],
        "outline": [
            {
                "id": "sec_1",
                "title": "默认大纲",
                "description": "默认生成的大纲",
                "section_type": "mixed",
                "requires_data": False,
                "requires_chart": False,
                "priority": 1,
                "search_queries": [state["query"]],
                "status": "pending",
            }
        ],
        "research_questions": [state["query"]],
        "budget": {"max_rounds": 2, "max_sources": 12, "max_tokens": 12000, "max_seconds": 45},
    }


def _validate_query(query: str) -> bool:
    """只做语法级校验，不做语义判断。"""
    text = query.strip()
    if not text:
        return False
    if len(text) < 3:
        return False
    return True


def _derive_search_plan(outline: list[dict], sub_questions: list[str],
                         _research_questions: list[str], query: str) -> list[dict]:
    """搜索计划仅来自 Planner LLM 生成的大纲搜索词。

    不在主流程中混入 heuristic / 模板猜测的搜索词。
    heuristic 完全不参与 merge，不会污染搜索预算。

    Fallback 链：
        1. LLM 大纲中的 search_queries
        2. 如果 LLM 没有生成任何有效搜索词，用用户原问题兜底
    """
    from ._utils import _dedupe_sources

    plan: list[dict] = []
    for section in outline:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "sec")
        for item in section.get("search_queries", []) or []:
            text = str(item).strip() if isinstance(item, str) else ""
            if _validate_query(text):
                plan.append({
                    "section_id": section_id,
                    "query": text,
                    "source_preference": "hybrid",
                    "reason": f"来自大纲章节 {section_id}",
                })

    if not plan:
        plan.append({"section_id": "sec_1", "query": query,
                     "source_preference": "hybrid", "reason": "fallback"})

    deduped = _dedupe_sources(plan, ["query"])
    return deduped[:6]


def plan_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("start | agent=%s", "[plan]", agent_name)
    log_inputs("plan", agent_name, {"query": state["query"]})
    fallback = _default_plan(state)
    payload, content, messages = _invoke_json_agent(
        state,
        f"用户需求：{state['query']}\n请先做大纲与问题拆解，再输出规划 JSON。",
        agent,
        agent_name,
        "plan",
        fallback,
    )
    outline = payload.get("outline") if isinstance(payload.get("outline"), list) else fallback["outline"]
    sub_questions = payload.get("sub_questions") if isinstance(payload.get("sub_questions"), list) else fallback["sub_questions"]
    research_questions = payload.get("research_questions") if isinstance(payload.get("research_questions"), list) else fallback["research_questions"]
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else fallback["budget"]
    search_plan = _derive_search_plan(outline, sub_questions, research_questions, state["query"])
    plan_summary = payload.get("objective") or state["query"]

    # Planner 指标统计
    planner_queries_raw = sum(len(s.get("search_queries", [])) if isinstance(s, dict) else 0 for s in outline)
    planner_queries_valid = len([q for q in search_plan if q.get("reason", "").startswith("来自大纲")])
    planner_queries_fallback = len([q for q in search_plan if q.get("reason") == "fallback"])

    return {
        "phase": "planning completed",
        "plan": plan_summary,
        "outline": outline,
        "sub_questions": sub_questions,
        "research_questions": research_questions,
        "search_plan": search_plan,
        "budget": budget,
        "messages": messages,
        "draft": content,
        "iteration": 0,
        "planner_stats": {
            "queries_generated": planner_queries_raw,
            "queries_valid": planner_queries_valid,
            "queries_fallback": planner_queries_fallback,
            "queries_executed": len(search_plan),
        },
    }
