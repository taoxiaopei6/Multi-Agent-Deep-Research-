import asyncio
from datetime import datetime
from threading import Lock, Thread
from typing import AsyncIterator, Callable

from mult_agents.config import AppConfig
from mult_agents.graph import build_app as build_workflow_app
from mult_agents.main import build_agents, build_checkpointer, build_memory_manager
from mult_agents.state import create_initial_state


class WorkflowService:
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._lock = Lock()
        self._initialized = False
        self._base_config: AppConfig | None = None
        self._memory_manager = None
        self._app = None

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            base_config = AppConfig.from_file(self._config_path)
            self._memory_manager = build_memory_manager(base_config)
            agents = build_agents(base_config.model, base_config.api_key, base_config.base_url, base_config)
            checkpointer = build_checkpointer(base_config)
            self._app = build_workflow_app(agents, checkpointer)
            self._base_config = base_config
            self._initialized = True

    def _build_runtime_config(
        self,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        max_iterations: int | None,
        enable_memory: bool | None,
    ) -> AppConfig:
        if self._base_config is None:
            raise RuntimeError("service not initialized")
        overrides = {
            "user_id": user_id,
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "max_iterations": max_iterations if max_iterations is not None else self._base_config.max_iterations,
        }
        if enable_memory is not None:
            overrides["enable_memory"] = enable_memory
        return self._base_config.with_overrides(**overrides)

    def _run_sync(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        max_iterations: int | None,
        enable_memory: bool | None,
    ) -> tuple[str, str]:
        self._ensure_initialized()
        runtime_config = self._build_runtime_config(
            user_id=user_id,
            thread_id=thread_id,
            tenant_id=tenant_id,
            max_iterations=max_iterations,
            enable_memory=enable_memory,
        )
        memory_context = ""
        if self._memory_manager and runtime_config.enable_memory:
            memory_context = self._memory_manager.build_personalized_prompt_context(
                user_id=runtime_config.user_id,
                thread_id=runtime_config.thread_id,
                query=query,
                tenant_id=runtime_config.tenant_id,
                max_memories=runtime_config.memory_top_k,
            )
        state = create_initial_state(
            query=query,
            max_iterations=runtime_config.max_iterations,
            user_id=runtime_config.user_id,
            tenant_id=runtime_config.tenant_id,
            memory_context=memory_context,
        )
        result = self._app.invoke(
            state,
            {"configurable": {"thread_id": runtime_config.thread_id}},
        )
        final = result.get("final", "")
        route = str(result.get("intent", "multiagent"))
        if self._memory_manager and runtime_config.enable_memory:
            self._memory_manager.persist_turn(
                tenant_id=runtime_config.tenant_id,
                user_id=runtime_config.user_id,
                thread_id=runtime_config.thread_id,
                query=query,
                answer=final,
            )
        return final, route

    @staticmethod
    def _node_message(node_name: str) -> str:
        mapping = {
            "intent": "Intent Router 正在识别问题意图",
            "direct_answer": "Direct Responder 正在快速作答",
            "plan": "Planner 正在拆解问题",
            "web_search": "Web Scout 正在检索网络证据",
            "local_rag": "Local Scout 正在检索本地知识库",
            "deep_dive": "Evidence Judge 正在进行证据裁判",
            "analyze": "Analyst 正在生成结论",
            "reflect": "Reflect 正在生成补搜计划",
            "write": "Writer 正在撰写最终报告",
        }
        return mapping.get(node_name, f"{node_name} 正在执行")

    @staticmethod
    def _extract_trace_metrics(node_output: dict) -> dict | None:
        """从 node_output 中提取最新的 trace 指标（latency + tokens）。"""
        try:
            traces = node_output.get("trace_events", [])
            if not traces:
                return None
            latest = traces[-1]
            metrics = {}
            lm = latest.get("latency_ms")
            if lm is not None:
                metrics["duration_ms"] = round(lm, 1)
            m = latest.get("metrics", {})
            if m and m.get("tokens"):
                metrics["tokens"] = m["tokens"]
            return metrics if metrics else None
        except Exception:
            return None

    @staticmethod
    def _build_trace_event(node_name: str, node_output: dict, accumulated: list[dict]) -> dict | None:
        """从节点输出构建 Trace 事件。

        每个节点完成时，从它的输出中提取关键指标，生成一个结构化的 trace 事件。
        """
        event = {
            "node": node_name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
        # 提取 latency + tokens 注入到每个 trace event
        trace_metrics = WorkflowService._extract_trace_metrics(node_output)
        if trace_metrics:
            event["metrics"] = trace_metrics
        # 从 Plan 节点提取子问题数和搜索计划数
        if node_name == "plan":
            plan = node_output.get("plan", "")
            sub_q = node_output.get("sub_questions", [])
            search_plan = node_output.get("search_plan", [])
            event["summary"] = f"Generated {len(sub_q)} sub-questions, {len(search_plan)} search queries"
            event["output_summary"] = {
                "sub_questions": len(sub_q),
                "search_queries": len(search_plan),
            }
            return event

        # 从 Web Search 节点提取网页数量
        if node_name == "web_search":
            web_ev = node_output.get("web_evidence", [])
            # 计算本轮的增量
            prev_web = sum(1 for t in accumulated if t.get("node") == "web_search")
            stats = node_output.get("web_retrieval_stats", {})
            raw = stats.get("raw_count", 0)
            kept = stats.get("kept_count", 0)
            event["summary"] = f"Web search: {raw} raw → {kept} kept"
            event["output_summary"] = {"raw_results": raw, "evidence_kept": kept}
            return event

        # 从 Local RAG 节点提取本地知识库结果
        if node_name == "local_rag":
            stats = node_output.get("local_retrieval_stats", {})
            raw = stats.get("raw_count", 0)
            kept = stats.get("kept_count", 0)
            event["summary"] = f"Local RAG: {raw} raw → {kept} kept"
            event["output_summary"] = {"raw_results": raw, "evidence_kept": kept}
            return event

        # 从 Evidence Judge 节点提取证据池统计
        if node_name == "deep_dive":
            pool = node_output.get("evidence_pool", [])
            flags = node_output.get("audit_flags", [])
            high = sum(1 for e in pool if isinstance(e, dict) and e.get("reliability_score", 0) >= 0.8)
            low = sum(1 for e in pool if isinstance(e, dict) and e.get("reliability_score", 0) < 0.6)
            event["summary"] = f"Evidence pool: {len(pool)} sources ({high} high-confidence, {low} low-confidence)"
            event["output_summary"] = {
                "evidence_total": len(pool),
                "high_confidence": high,
                "low_confidence": low,
                "audit_flags": len(flags),
            }
            return event

        # 从 Analyst 节点提取发现和缺口
        if node_name == "analyze":
            findings = node_output.get("findings", [])
            gaps = node_output.get("missing_gaps", [])
            needs_more = node_output.get("needs_more_research", False)
            event["summary"] = f"Analysis: {len(findings)} claims, {len(gaps)} gaps, needs_more={needs_more}"
            event["output_summary"] = {
                "findings": len(findings),
                "missing_gaps": len(gaps),
                "needs_more_research": needs_more,
            }
            return event

        # 从 Reflect 节点提取补搜计划
        if node_name == "reflect":
            supp = node_output.get("supplementary_queries", [])
            iteration = node_output.get("iteration", 0)
            event["summary"] = f"Gap search (round {iteration}): {len(supp)} supplementary queries"
            event["output_summary"] = {"iteration": iteration, "supplementary_queries": len(supp)}
            return event

        # 从 Writer 节点提取报告和 Artifact
        if node_name == "write":
            draft = node_output.get("draft", "") or node_output.get("final", "")
            report_len = len(draft) if draft else 0
            event["summary"] = f"Report generated: ~{report_len} chars"
            event["output_summary"] = {"report_length": report_len}
            # Research Artifact 由 emit 循环单独发送（见 _emit_artifact）
            return event

        # 从 Intent 节点提取路由结果
        if node_name == "intent":
            intent = node_output.get("intent", "unknown")
            event["summary"] = f"Intent: {intent}"
            event["output_summary"] = {"route": intent}
            return event

        return None

    @staticmethod
    def _build_research_artifact(node_output: dict) -> dict | None:
        """从 write 节点的输出构建 Research Artifact。

        Artifact 将 findings（结论）、claim_map（结论-证据映射）
        和 evidence_pool（证据池）关联成一个结构化产物，
        让用户能追溯"报告中的每个结论来自哪些来源、可信度如何"。
        """
        # node_output 中可能包含完整 state 的部分字段
        # find actual keys — they come from write_node's return dict
        findings = node_output.get("findings", []) or []
        claim_map = node_output.get("claim_map", []) or []
        evidence_pool = node_output.get("evidence_pool", []) or []
        source_index = node_output.get("source_index", []) or []
        audit_flags = node_output.get("audit_flags", []) or []

        if not findings and not evidence_pool:
            return None

        # 构建 evidence 查询表
        ev_lookup = {}
        for ev in evidence_pool:
            sid = str(ev.get("source_id", "")).strip()
            if sid:
                ev_lookup[sid] = {
                    "source_id": sid,
                    "title": ev.get("title") or str(ev.get("source_label", "")),
                    "url": ev.get("url", ""),
                    "source_type": ev.get("source_type", "web"),
                    "reliability_score": ev.get("reliability_score", 0.5),
                    "reliability_breakdown": ev.get("reliability_breakdown"),
                    "reliability_reason": ev.get("reliability_reason", ""),
                }

        # 从 source_index 补充
        for si in source_index:
            sid = str(si.get("source_id", "")).strip()
            if sid and sid not in ev_lookup:
                ev_lookup[sid] = {
                    "source_id": sid,
                    "title": si.get("label", ""),
                    "url": si.get("locator", ""),
                    "source_type": si.get("source_type", "source"),
                    "reliability_score": None,
                }

        # 将每个 finding 关联到对应的 evidence
        claims = []
        for f in findings:
            cid = str(f.get("claim_id", ""))
            claim_text = str(f.get("claim", ""))
            confidence = str(f.get("confidence", "medium"))
            source_ids = f.get("source_ids", [])

            supporting = []
            for sid in source_ids:
                sid_clean = str(sid).strip()
                ev = ev_lookup.get(sid_clean)
                if ev:
                    supporting.append(ev)

            claims.append({
                "claim_id": cid,
                "claim": claim_text[:200],
                "confidence": confidence,
                "evidence_count": len(supporting),
                "supporting_evidence": supporting[:5],  # 最多展示5条
            })

        # 统计
        high = sum(1 for ev in evidence_pool if isinstance(ev, dict) and (ev.get("reliability_score") or 0) >= 0.8)
        low = sum(1 for ev in evidence_pool if isinstance(ev, dict) and (ev.get("reliability_score") or 0) < 0.6)

        return {
            "artifact_version": "1.0",
            "query": node_output.get("query", ""),
            "claims": claims,
            "evidence_pool_summary": {
                "total": len(evidence_pool),
                "high_confidence": high,
                "low_confidence": low,
            },
            "audit_flags": len(audit_flags),
            "total_claims": len(claims),
            "total_evidence": len(ev_lookup),
        }

    def _run_sync_with_events(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        max_iterations: int | None,
        enable_memory: bool | None,
        emit: Callable[[dict], None],
    ) -> tuple[str, str]:
        self._ensure_initialized()
        runtime_config = self._build_runtime_config(
            user_id=user_id,
            thread_id=thread_id,
            tenant_id=tenant_id,
            max_iterations=max_iterations,
            enable_memory=enable_memory,
        )
        memory_context = ""
        if self._memory_manager and runtime_config.enable_memory:
            memory_context = self._memory_manager.build_personalized_prompt_context(
                user_id=runtime_config.user_id,
                thread_id=runtime_config.thread_id,
                query=query,
                tenant_id=runtime_config.tenant_id,
                max_memories=runtime_config.memory_top_k,
            )
        state = create_initial_state(
            query=query,
            max_iterations=runtime_config.max_iterations,
            user_id=runtime_config.user_id,
            tenant_id=runtime_config.tenant_id,
            memory_context=memory_context,
        )
        final = ""
        route = "multiagent"
        config = {"configurable": {"thread_id": runtime_config.thread_id}}
        trace_events_accumulated = []
        for update in self._app.stream(state, config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, node_output in update.items():
                emit({"type": "phase", "node": node_name, "message": self._node_message(str(node_name))})
                if isinstance(node_output, dict):
                    if node_name == "intent":
                        detected = str(node_output.get("intent", route)).strip().lower()
                        if detected in {"direct", "multiagent"}:
                            route = detected

                    # B2: evidence_pool from deep_dive node
                    if node_name == "deep_dive":
                        evidence_pool = node_output.get("evidence_pool")
                        if evidence_pool:
                            emit({
                                "type": "evidence",
                                "evidence_pool": [
                                    {
                                        "source_id": e.get("source_id", ""),
                                        "title": e.get("title", ""),
                                        "reliability_score": e.get("reliability_score", 0),
                                        "reliability_reason": e.get("reliability_reason", ""),
                                        "url": e.get("url", ""),
                                        "reliability_breakdown": e.get("reliability_breakdown"),
                                    }
                                    for e in evidence_pool
                                ]
                            })

                    # B3: iteration round display on reflect
                    if node_name == "reflect":
                        iteration = node_output.get("iteration", 0)
                        if iteration > 0:
                            emit({"type": "iteration", "round": iteration})

                    # ── Trace: 从 node_output 中提取结构化 Trace ──
                    trace_event = self._build_trace_event(node_name, node_output, trace_events_accumulated)
                    if trace_event:
                        trace_events_accumulated.append(trace_event)
                        emit({"type": "trace", "trace": trace_event})

                    # ── Research Artifact: write 节点完成时生成 ──
                    if node_name == "write":
                        artifact = self._build_research_artifact(node_output)
                        if artifact:
                            emit({"type": "artifact", "artifact": artifact})

                    value = node_output.get("final")
                    if value:
                        final = str(value)
        if not final:
            result = self._app.invoke(state, config)
            final = str(result.get("final", ""))
            route = str(result.get("intent", route)).strip().lower()
        if self._memory_manager and runtime_config.enable_memory:
            self._memory_manager.persist_turn(
                tenant_id=runtime_config.tenant_id,
                user_id=runtime_config.user_id,
                thread_id=runtime_config.thread_id,
                query=query,
                answer=final,
            )
        return final, route

    async def run(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        max_iterations: int | None,
        enable_memory: bool | None,
    ) -> str:
        final, _ = await asyncio.to_thread(
            self._run_sync,
            query,
            user_id,
            thread_id,
            tenant_id,
            max_iterations,
            enable_memory,
        )
        return final

    async def run_with_route(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        max_iterations: int | None,
        enable_memory: bool | None,
    ) -> tuple[str, str]:
        return await asyncio.to_thread(
            self._run_sync,
            query,
            user_id,
            thread_id,
            tenant_id,
            max_iterations,
            enable_memory,
        )

    async def stream_events(
        self,
        query: str,
        user_id: str,
        thread_id: str,
        tenant_id: str,
        max_iterations: int | None,
        enable_memory: bool | None,
    ) -> AsyncIterator[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(event: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

        def worker() -> None:
            try:
                final, route = self._run_sync_with_events(
                    query=query,
                    user_id=user_id,
                    thread_id=thread_id,
                    tenant_id=tenant_id,
                    max_iterations=max_iterations,
                    enable_memory=enable_memory,
                    emit=emit,
                )
                emit({"type": "route", "message": "已走直接回答路径" if route == "direct" else "已走多智能体研究路径"})
                emit(
                    {
                        "type": "final",
                        "query": query,
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "tenant_id": tenant_id,
                        "final": final,
                    }
                )
            except Exception as exc:
                emit({"type": "error", "message": str(exc)})
            finally:
                emit({"type": "__done__"})

        Thread(target=worker, daemon=True).start()
        while True:
            event = await queue.get()
            if event.get("type") == "__done__":
                break
            yield event
