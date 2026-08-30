import asyncio
from datetime import datetime
from threading import Lock, Thread
from typing import AsyncIterator, Callable

from mult_agents.config import AppConfig
from mult_agents.graph import build_app as build_workflow_app
from mult_agents.main import build_agents, build_checkpointer, build_memory_manager
from mult_agents.state import create_initial_state


class WorkflowService:
    """
    多智能体深度研究工作流服务类。

    该类负责封装和管理基于 LangGraph 构建的多智能体工作流（Multi-Agent Workflow）的生命周期与执行逻辑。
    主要职责包括：
    1. 延迟初始化（Lazy Initialization）：在首次调用时加载配置、构建智能体、内存管理器和 LangGraph 应用。
    2. 运行时配置管理：支持针对每次请求动态覆盖用户、线程、租户、最大迭代次数及内存开关等参数。
    3. 同步与异步执行：提供阻塞式执行（`run`, `run_with_route`）以及基于后台线程和异步队列的流式事件推送（`stream_events`）。
    4. 流式事件与状态追踪：在流式执行过程中，实时解析各节点（如意图识别、规划、检索、分析、反思、写作等）的输出，
       生成并推送阶段状态（phase）、证据池（evidence）、迭代轮次（iteration）、执行追踪（trace）及研究产物（artifact）等事件。
    5. 记忆管理：在请求前后与 MemoryManager 交互，构建个性化上下文并持久化当前对话轮次。
    """
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._lock = Lock()
        self._initialized = False
        self._base_config: AppConfig | None = None
        self._memory_manager = None
        self._app = None

    def _ensure_initialized(self) -> None:
        """
        确保工作流服务已完成初始化（延迟初始化 / Lazy Initialization）。
        
        该方法采用双重检查锁定（Double-Checked Locking）机制，以保证在多线程环境下的线程安全。
        如果服务尚未初始化，它将依次执行以下操作：
        1. 从指定路径加载基础配置 (AppConfig)。
        2. 构建内存管理器 (MemoryManager)。
        3. 构建多智能体集群 (Agents)。
        4. 构建状态检查点器 (Checkpointer)。
        5. 组装并构建最终的 LangGraph 工作流应用 (App)。
        6. 将初始化标志位设为 True。
        """
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
        """
        构建当前请求的运行时配置。
        
        【这是干什么的】：
        该方法用于将全局的基础配置 (self._base_config) 与当前请求的动态参数
        (如 user_id, thread_id, max_iterations 等) 进行合并，生成一份专属于
        本次请求的运行时配置 (AppConfig) 副本。
        这样可以在不污染全局配置的前提下，实现针对不同用户、会话和请求参数的
        个性化配置覆盖（例如动态调整最大迭代次数或开关记忆功能）。
        
        参数:
            user_id (str): 用户唯一标识。
            thread_id (str): 对话/线程唯一标识。
            tenant_id (str): 租户唯一标识。
            max_iterations (int | None): 最大迭代次数。若为 None，则使用基础配置中的默认值。
            enable_memory (bool | None): 是否启用记忆。若为 None，则保留基础配置中的默认设置。
            
        返回:
            AppConfig: 包含运行时覆盖参数的新配置对象。
            
        异常:
            RuntimeError: 当服务尚未初始化（即 _base_config 为 None）时抛出。
        """
        if self._base_config is None:
            raise RuntimeError("服务尚未初始化，无法构建运行时配置")
            
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
        """
        同步执行多智能体深度研究工作流。

        【这是干什么的】：
        该方法以阻塞（同步）方式执行完整的 LangGraph 工作流，通常由外层的异步方法
        （如 `run` 或 `run_with_route`）通过 `asyncio.to_thread` 放入后台线程中调用。
        其核心职责包括：
        1. 确保服务已初始化，并构建当前请求的运行时配置。
        2. 根据配置决定是否从 MemoryManager 加载个性化记忆上下文。
        3. 组装初始状态（Initial State），并调用底层 LangGraph 应用的 `invoke` 方法执行推理。
        4. 从执行结果中提取最终回答（final）和意图路由（route）。
        5. 若启用了记忆功能，将当前的问答轮次持久化到 MemoryManager 中。

        参数:
            query (str): 用户的查询问题。
            user_id (str): 用户唯一标识。
            thread_id (str): 对话/线程唯一标识。
            tenant_id (str): 租户唯一标识。
            max_iterations (int | None): 最大迭代次数。若为 None，则使用基础配置中的默认值。
            enable_memory (bool | None): 是否启用记忆。若为 None，则保留基础配置中的默认设置。

        返回:
            tuple[str, str]: 包含两个元素的元组：
                - final (str): 工作流生成的最终回答或研究报告内容。
                - route (str): 意图路由结果（如 "direct" 表示直接回答，"multiagent" 表示多智能体研究）。
        """

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
        """
        获取指定工作流节点的执行状态提示信息。

        【这是干什么的】：
        该方法用于将 LangGraph 工作流中各个节点的内部标识（node_name）
        转换为对人类友好的、面向前端或日志展示的执行状态描述。
        在流式执行过程中，当工作流进入某个节点时，系统会调用此方法
        生成类似 "Planner 正在拆解问题" 的提示语，并通过事件流（如 SSE）推送给前端，
        以便用户实时了解当前深度研究的进度和具体环节。
        如果传入的节点名称不在预定义的映射表中，则返回一个默认的通用提示。

        参数:
            node_name (str): LangGraph 工作流中的节点名称（如 "intent", "plan", "web_search" 等）。

        返回:
            str: 描述该节点当前正在执行操作的中文提示信息。
        """
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
        """从 node_output 中提取最新的 trace 指标（latency + tokens）。
        提取节点输出中最新一次 Trace Event 的性能指标。

    该函数从 `node_output["trace_events"]` 中获取最后一个 Trace Event，
    并提取其中的延迟（latency）和 Token 使用量，统一转换为前端展示
    所需的格式。

    提取规则：
        - latency_ms -> duration_ms（保留 1 位小数）
        - metrics.tokens -> tokens
        """

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
        根据工作流节点输出构建标准化 Trace Event。

        该函数负责将各个 Workflow 节点的执行结果转换为统一格式的
        Trace Event，供前端执行轨迹（Timeline）、调试日志以及运行
        状态展示使用。

        每个节点都会生成一个基础事件（节点名称、状态、时间戳），并根据
        不同节点类型提取具有代表性的业务指标（如搜索结果数量、证据统计、
        分析结论数量、报告长度等），生成简洁的执行摘要。

        同时会调用 `_extract_trace_metrics()` 提取底层 Trace 指标
        （例如耗时、Token 消耗），统一附加到事件中。

        支持的节点包括：
            - intent      ：意图识别结果
            - plan        ：问题拆解及搜索计划
            - web_search  ：网络检索统计
            - local_rag   ：本地知识库检索统计
            - deep_dive   ：证据池质量统计
            - analyze     ：分析结论与研究缺口
            - reflect     ：补充检索计划
            - write       ：最终报告生成

        Args:
            node_name:
                当前执行完成的 Workflow 节点名称。

            node_output:
                当前节点输出的完整结果字典，用于提取节点摘要信息。

            accumulated:
                当前 Workflow 已生成的 Trace Event 列表，可用于计算
                增量统计或历史状态（部分节点可能需要）。

        Returns:
            dict:
                标准化 Trace Event，例如：

                {
                    "node": "web_search",
                    "status": "completed",
                    "timestamp": "...",
                    "metrics": {...},
                    "summary": "...",
                    "output_summary": {...}
                }

            None:
                当前节点无需生成 Trace Event 时返回 None。

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
        根据 Write 节点输出构建结构化 Research Artifact。

        该函数负责将最终研究结果组织为一个可追溯（Traceable）、
        可解释（Explainable）的结构化产物，用于前端展示报告依据、
        证据来源以及可信度分析。

        函数会从 Write 节点输出中提取研究结论（findings）、证据池
        （evidence_pool）、来源索引（source_index）以及审计信息，
        建立结论与证据之间的映射关系（Claim → Supporting Evidence），
        同时统计整体证据质量，为最终报告提供透明的来源追踪能力。

        Artifact 的核心内容包括：
            - 查询内容（Query）
            - 研究结论（Claims）
            - 每条结论对应的支持证据（Supporting Evidence）
            - 证据池统计信息（Evidence Summary）
            - 审计结果（Audit Flags）

        Args:
            node_output:
                Write 节点输出的完整结果字典，应包含 findings、
                evidence_pool、source_index、audit_flags 等字段。

        Returns:
            dict:
                结构化 Research Artifact，例如：

                {
                    "artifact_version": "1.0",
                    "query": "...",
                    "claims": [...],
                    "evidence_pool_summary": {...},
                    "audit_flags": 0,
                    "total_claims": 6,
                    "total_evidence": 28
                }

            None:
                当不存在研究结论且证据池为空时返回 None。
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
        """
            同步执行一次 Agent 对话任务，并在执行过程中实时产生事件(Event)。

            整个流程包括：
            1. 初始化运行环境和运行配置(RuntimeConfig)；
            2. 根据用户信息构建记忆上下文(Memory Context)；
            3. 创建本轮任务的初始状态(State)；
            4. 调用 LangGraph 执行整个多Agent工作流；
            5. 在工作流运行过程中持续监听各节点(Node)输出，并实时发送事件：
               - 当前执行阶段(phase)
               - 检索证据(evidence)
               - 迭代轮次(iteration)
               - Trace调试信息(trace)
               - Research Artifact(研究成果)
            6. 获取最终回答(final)及路由方式(route)；
            7. 将本轮问答保存到长期记忆(Memory)；
            8. 返回最终答案和执行路由。
            """
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
            
        # 5. 持久化记忆
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
        """以异步事件流方式执行工作流，并实时输出运行事件。

            该函数负责将同步执行的 Workflow 封装为异步事件流（Async
            Event Stream），供前端通过 SSE、WebSocket 或其他流式协议
            实时接收工作流执行过程。

            函数会创建线程安全的事件队列，在后台线程中运行同步 Workflow，
            并将执行过程中产生的 Trace Event、Research Artifact、状态
            更新、路由信息以及最终结果持续写入队列，再以异步生成器
            （AsyncIterator）的形式逐条输出。

            事件流生命周期如下：

                创建事件队列
                    ↓
                启动后台 Worker
                    ↓
                执行同步 Workflow
                    ↓
                emit(...) 持续产生事件
                    ↓
                Queue 接收事件
                    ↓
                yield 实时返回前端
                    ↓
                收到 __done__ 后结束事件流

            Args:
                query:
                    用户查询内容。

                user_id:
                    用户唯一标识。

                thread_id:
                    当前会话线程 ID。

                tenant_id:
                    租户标识。

                max_iterations:
                    最大研究迭代次数。

                enable_memory:
                    是否启用记忆模块。

            Yields:
                dict:
                    Workflow 产生的事件对象，包括但不限于：

                    - trace：节点执行轨迹
                    - artifact：Research Artifact
                    - route：工作流路由结果
                    - final：最终回答
                    - error：执行异常

            Raises:
                本函数不会直接抛出工作流异常，异常会被捕获并转换为
                error 类型事件发送给前端。
            """
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
