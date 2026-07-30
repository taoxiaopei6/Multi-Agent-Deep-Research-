"""Shared utilities for agent node implementations."""

import json
import logging
import os
import re
import time
from functools import partial
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from ..state import ResearchState


logger = logging.getLogger("mult_agents")


ANSI = {
    "reset": "\033[0m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def colorize(text: str, color: str) -> str:
    if os.getenv("NO_COLOR"):
        return text
    code = ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{ANSI['reset']}"


def emit(node: str, content: str):
    preview = content.replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "..."
    logger.info("%s 输出: %s", f"[{node}]", preview)


# ── Trace 系统 ──


TRACE_NODE_LABELS = {
    "intent": "Intent Routing",
    "direct_answer": "Direct Answer",
    "plan": "Research Planning",
    "web_search": "Web Search",
    "local_rag": "Local Knowledge Base",
    "deep_dive": "Evidence Judge",
    "analyze": "Analysis & Gap Detection",
    "reflect": "Reflection & Gap Search",
    "write": "Report Writing",
}


def record_trace(state: ResearchState, node: str, status: str,
                 output_summary: dict | None = None,
                 latency_ms: float | None = None,
                 error: str | None = None,
                 metrics: dict | None = None) -> ResearchState:
    """Record a structured trace event in state.trace_events.
        metrics: 附加指标（token用量等）
    """
    event = {
        "node": node,
        "label": TRACE_NODE_LABELS.get(node, node),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if latency_ms is not None:
        event["latency_ms"] = round(latency_ms, 1)
    if error:
        event["error"] = error
    if output_summary:
        event["output_summary"] = output_summary
    if metrics:
        event["metrics"] = metrics

    existing = state.get("trace_events", [])
    existing.append(event)
    return {"trace_events": existing}


def auto_record_trace(state: ResearchState, node: str, start_time: float,
                      output_summary: dict | None = None,
                      error: str | None = None,
                      token_usage: dict | None = None) -> ResearchState:
    """计算耗时并自动记录 trace，含可选 token 用量。"""
    latency_ms = (time.time() - start_time) * 1000
    status = "failed" if error else "completed"
    metrics = {}
    if token_usage:
        metrics["tokens"] = token_usage
    return record_trace(state, node, status,
                        output_summary=output_summary,
                        latency_ms=latency_ms,
                        error=error,
                        metrics=metrics)


def collect_tool_calls(messages) -> tuple[list, list]:
    tools = []
    tool_outputs = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    tools.append(name)
        name = getattr(msg, "name", None)
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and name:
            tools.append(name)
            output = getattr(msg, "content", "")
            if output:
                tool_outputs.append(f"{name}: {output}")
    return tools, tool_outputs


def with_memory_context(state: ResearchState, user_prompt: str) -> str:
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"


def log_inputs(node: str, agent_name: str, payload: dict):
    preview = {
        key: (value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value)
        for key, value in payload.items()
    }
    logger.info("%s 输入 | agent=%s | data=%s", f"[{node}]", agent_name, preview)


def bind_agent(node_func, agent, agent_name: str):
    return partial(node_func, agent=agent, agent_name=agent_name)


def _last_content(result) -> str:
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _extract_json_block(text: str) -> str:
    """从 LLM 输出中鲁棒地提取 JSON。

    处理各种边缘情况：
    - markdown 代码块包装（```json ... ```）
    - 前置/后置多余文本
    - 多个 {} 块（取最外层结构最完整的）
    - 残缺的 markdown 标记
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # 统一去掉 markdown 代码块标记（各种残缺形式）
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 定位最外层 {}
    brace_stack = []
    valid_ranges = []
    for i, ch in enumerate(cleaned):
        if ch == "{":
            brace_stack.append(i)
        elif ch == "}":
            if brace_stack:
                start = brace_stack.pop()
                if not brace_stack:  # 匹配到最外层
                    valid_ranges.append((start, i + 1))

    if valid_ranges:
        # 取最后一个完整的 {} 块（通常是最完整的）
        start, end = valid_ranges[-1]
        candidate = cleaned[start:end]
        # 验证是否能解析
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            # 尝试修复常见问题
            pass

    # 兜底：找不到合法的 JSON，尝试找第一个 {
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
        # 修复尾部多余的逗号（LLM 常见问题）
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        return candidate

    return cleaned


def _repair_json(raw: str) -> str:
    """修复 LLM 返回中常见的 JSON 格式问题。"""
    text = raw.strip()
    # 1. 修复尾部逗号
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    # 2. 移除控制字符（除 \t \n \r 外）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # 3. 修复单引号为双引号（仅当显然不是合法用法时）
    # 简单策略：替换不在字符串内的单引号为双引号
    # 这块太复杂，留给 json.loads 处理
    return text


def _load_json(text: str, fallback: dict) -> dict:
    raw = _extract_json_block(text)
    # 尝试直接解析
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    # 尝试修复后解析
    repaired = _repair_json(raw)
    try:
        value = json.loads(repaired)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    # 最后尝试用 json.decoder 的 strict=False
    try:
        import json.decoder
        value = json.loads(repaired, strict=False)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return fallback


def _validate_with_schema(data: dict, node: str) -> dict:
    """用 Pydantic schema 校验解析后的 JSON，无效字段自动用默认值补全。"""
    try:
        from ..schemas import SCHEMA_MAP
        schema_cls = SCHEMA_MAP.get(node)
        if schema_cls is None:
            return data
        validated = schema_cls(**data)
        cleaned = validated.to_dict() if hasattr(validated, "to_dict") else validated.model_dump()
        # 只保留 cleaned 中存在的顶层 key，忽略未知字段
        filtered = {k: v for k, v in data.items() if k in cleaned if v is not None and v != "" and v != []}
        # 把 cleaned 中的有效值补进去
        for k, v in cleaned.items():
            if v is not None and v != "" and v != []:
                filtered.setdefault(k, v)
        return filtered
    except Exception as exc:
        logger.warning("%s schema 校验失败，使用原始解析数据: %s", f"[{node}]", exc)
        return data


def _summarize_output(node: str, parsed: dict) -> dict | None:
    """从解析后的 JSON 中提取关键摘要用于 trace 展示。"""
    try:
        if node in ("intent", "intent_router"):
            return {"route": parsed.get("route", "unknown"), "reason": parsed.get("reason", "")[:80]}
        if node == "plan":
            sq = parsed.get("sub_questions", [])
            outline = parsed.get("outline", [])
            queries = sum(len(s.get("search_queries", [])) for s in outline if isinstance(s, dict))
            return {"sub_questions": len(sq), "search_queries": queries or len(outline)}
        if node == "web_search":
            ev = parsed.get("evidence", [])
            rej = parsed.get("rejected_source_ids", [])
            return {"evidence_kept": len(ev), "evidence_rejected": len(rej) if rej else None}
        if node == "local_rag":
            ev = parsed.get("evidence", [])
            return {"evidence_kept": len(ev)}
        if node == "deep_dive":
            pool = parsed.get("evidence_pool", [])
            flags = parsed.get("audit_flags", [])
            high = sum(1 for e in pool if isinstance(e, dict) and e.get("reliability_score", 0) >= 0.8)
            low = sum(1 for e in pool if isinstance(e, dict) and e.get("reliability_score", 0) < 0.6)
            return {"evidence_total": len(pool), "high_confidence": high, "low_confidence": low, "audit_flags": len(flags)}
        if node == "analyze":
            return {
                "needs_more_research": parsed.get("needs_more_research", False),
                "findings": len(parsed.get("findings", [])),
                "missing_gaps": len(parsed.get("missing_gaps", [])),
            }
        if node == "reflect":
            return {"supplementary_queries": len(parsed.get("supplementary_queries", []))}
    except Exception:
        pass
    return None


def _extract_token_usage(result: dict) -> dict | None:
    """从 LangChain LLM 调用结果中提取 token 用量。

    LangChain 的 AIMessage.response_metadata 中包含:
        - token_usage.prompt_tokens
        - token_usage.completion_tokens
        - token_usage.total_tokens
    """
    try:
        last_msg = result.get("messages", [{}])[-1]
        metadata = getattr(last_msg, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
        if isinstance(usage, dict) and usage:
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        # 兼容不同格式
        if hasattr(usage, "__dict__"):
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
    except Exception:
        pass
    return None


def _invoke_json_agent(state: ResearchState, prompt: str, agent, agent_name: str, node: str, fallback: dict) -> tuple[dict, str, list]:
    start_time = time.time()
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    tools, tool_outputs = collect_tool_calls(result["messages"])
    logger.info("%s 工具: %s", f"[{node}]", ", ".join(tools) if tools else "无")
    for item in tool_outputs[:5]:
        logger.info("%s 工具输出: %s", f"[{node}]", item[:400])
    logger.info("%s LLM调用: 是 | 思考: 不可见", f"[{node}]")
    content = _last_content(result)
    emit(node, content)
    parsed = _load_json(content, fallback)
    validated = _validate_with_schema(parsed, node)
    # 提取 token 用量
    token_usage = _extract_token_usage(result)
    # 自动记录 trace（含 latency + tokens）
    output_summary = _summarize_output(node, validated)
    state.update(auto_record_trace(state, node, start_time, output_summary=output_summary, token_usage=token_usage))
    return validated, content, [human, result["messages"][-1]]


def detect_intent(query: str) -> str:
    normalized_query = query.strip()
    force_multiagent_keywords = [
        "调查", "调研", "来源", "证据", "检索统计", "来源清单",
        "重大新闻", "热门项目", "趋势", "新闻", "最新", "盘点",
    ]
    if re.search(r"20\d{2}年", normalized_query) and any(word in normalized_query for word in ["趋势", "新闻", "调研", "调查", "盘点"]):
        return "multiagent"
    if any(word in query for word in force_multiagent_keywords):
        return "multiagent"
    keywords = [
        "调研", "研究", "调查", "盘点", "热门", "趋势", "榜单",
        "分析", "方案", "架构", "设计", "对比", "报告", "代码",
        "实现", "落地", "检索", "知识库", "证据", "来源", "溯源",
        "资料", "手册", "验证", "数据", "模型",
    ]
    return "multiagent" if any(word in query for word in keywords) else "direct"


def _extract_query_terms(query: str) -> list[str]:
    parts = re.findall(r"[一-鿿]{2,}|[A-Za-z0-9_-]{3,}", query.lower())
    terms = []
    stopwords = {"什么", "如何", "以及", "一个", "关于", "这个", "那个", "进行", "基于", "附带", "来源", "清单", "是什么", "有哪些"}
    for part in parts:
        if part in stopwords:
            continue
        terms.append(part)
    return terms[:12]


def _estimate_relevance(query: str, text: str) -> float:
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)


def _minimal_record_filter(records: list[dict], required_any: list[str]) -> list[dict]:
    kept: list[dict] = []
    for record in records:
        if any(str(record.get(field, "")).strip() for field in required_any):
            kept.append(record)
    return kept


def _assign_source_ids(records: list[dict], prefix: str) -> list[dict]:
    assigned: list[dict] = []
    for index, record in enumerate(records, 1):
        item = dict(record)
        item["source_id"] = f"{prefix}-{index}"
        assigned.append(item)
    return assigned


def _normalize_source_ids(values) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _dedupe_sources(items: list[dict], key_fields: list[str]) -> list[dict]:
    seen = set()
    results = []
    for item in items:
        key = tuple(str(item.get(field, "")).strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def _normalize_query(query: str) -> str:
    """搜索 query 归一化：去空格、去干扰字符"""
    text = query.strip()
    text = " ".join(text.split())
    text = text.strip("，。,．!！?？、；：\"'")
    return text
