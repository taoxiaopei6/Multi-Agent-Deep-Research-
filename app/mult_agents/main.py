"""Entry point: builds agents, memory, checkpointer, and runs the workflow."""

"""
Multi-Agent Deep Research — CLI entry point.
"""
import argparse
import json
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mult_agents"

from .config import AppConfig
from .graph import build_app as build_workflow_app
from .memory import MemoryManager
from .prompts import PROMPTS
from .state import create_initial_state
from .tools import init_rag_system
from .rag.core import RAGConfig


logger = logging.getLogger("mult_agents")

MEMORY_MANAGER: Optional[MemoryManager] = None
CHECKPOINTER_CONTEXT = None

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


def build_memory_manager(config: AppConfig) -> Optional[MemoryManager]:
    if not config.enable_memory:
        return None
    try:
        return MemoryManager(
            short_term_ttl=config.short_term_ttl_seconds,
            short_term_max_messages=config.short_term_max_messages,
            short_term_summary_threshold=config.short_term_summary_threshold,
            tenant_id=config.tenant_id,
            short_term_backend=config.short_term_backend,
            long_term_backend=config.long_term_backend,
            long_term_scope=config.long_term_scope,
            save_conversation_task=config.save_conversation_task,
            enable_milvus=config.enable_milvus,
            redis_url=config.redis_url,
            postgres_dsn=config.postgres_dsn,
            milvus_host=config.milvus_host,
            milvus_port=config.milvus_port,
            milvus_collection=config.milvus_collection,
            embedding_api_key=config.api_key,
            embedding_model_path=config.embedding_model_path,
            base_url=config.base_url,
        )
    except Exception as exc:
        logger.exception("MemoryManager init failed, memory disabled: %s", exc)
        return None


def build_checkpointer(config: AppConfig):
    """根据应用配置构建并初始化 Workflow Checkpointer。

       该函数负责根据 AppConfig 中的配置选择合适的 Checkpointer
       后端，用于保存 LangGraph Workflow 的执行状态、Checkpoint
       以及记忆（Memory）数据。

       支持的后端包括：

           - PostgreSQL：适用于生产环境，可持久化保存 Checkpoint。
           - Redis：适用于 Redis Stack，支持快速状态存储。
           - In-Memory：仅保存在进程内存中，适用于开发、测试或作为降级方案。

       初始化顺序由配置决定：

           PostgreSQL（若启用）
                   ↓
           Redis（若启用）
                   ↓
           In-Memory（默认降级）

       当指定后端不可用（例如依赖缺失、连接失败或服务不可用）时，
       函数会记录日志并自动降级到下一级可用实现，确保 Workflow
       仍能够继续运行。

       同时，对于支持上下文管理的 Checkpointer（如 PostgreSQL、
       Redis），函数会保存其 Context，便于应用退出时统一释放资源。

       Args:
           config:
               应用配置对象，包含 Checkpointer 后端类型、数据库连接、
               Redis 地址、是否启用 Memory 等配置项。

       Returns:
           BaseCheckpointSaver:
               已初始化完成的 Checkpointer 实例，可能为：

               - PostgresSaver
               - RedisSaver
               - InMemorySaver

           返回对象可直接传入 LangGraph 编译 Workflow 时作为
           checkpointer 使用。
       """
    global CHECKPOINTER_CONTEXT
    backend = config.checkpointer_backend
    if backend in {"postgres", "auto"} and config.enable_memory and config.postgres_dsn:
        postgres_saver = None
        postgres_import_error = ""
        try:
            module = importlib.import_module("langgraph.checkpoint.postgres")
            postgres_saver = getattr(module, "PostgresSaver", None)
        except Exception as exc:
            postgres_import_error = str(exc)
        if postgres_saver is None:
            try:
                module = importlib.import_module("langgraph_checkpoint_postgres")
                postgres_saver = getattr(module, "PostgresSaver", None)
            except Exception as exc:
                postgres_import_error = postgres_import_error or str(exc)
        if postgres_saver is None:
            message = (
                "PostgreSQL checkpointer 模块不可用。请安装: pip install langgraph-checkpoint-postgres "
                f"| import_error={postgres_import_error or 'unknown'}"
            )
            if backend == "postgres":
                logger.warning("%s %s", colorize("[memory]", "yellow"), message)
            else:
                logger.info("%s %s", colorize("[memory]", "cyan"), message)
        else:
            try:
                CHECKPOINTER_CONTEXT = postgres_saver.from_conn_string(config.postgres_dsn)
                checkpointer = CHECKPOINTER_CONTEXT.__enter__()
                checkpointer.setup()
                logger.info("%s Using PostgreSQL checkpointer", colorize("[memory]", "green"))
                return checkpointer
            except Exception as exc:
                logger.warning("%s PostgreSQL checkpointer 初始化失败: %s", colorize("[memory]", "yellow"), exc)
    if backend in {"redis", "auto"} and config.enable_memory and config.redis_url:
        from langgraph.checkpoint.redis import RedisSaver

        candidate_urls = [config.redis_url]
        if "redis://root:" in config.redis_url:
            candidate_urls.append(config.redis_url.replace("redis://root:", "redis://:"))
        last_exc = None
        for url in candidate_urls:
            try:
                CHECKPOINTER_CONTEXT = RedisSaver.from_conn_string(url)
                checkpointer = CHECKPOINTER_CONTEXT.__enter__()
                checkpointer.setup()
                logger.info("%s Using Redis checkpointer", colorize("[memory]", "green"))
                return checkpointer
            except Exception as exc:
                last_exc = exc
        if last_exc and "FT._LIST" in str(last_exc):
            logger.warning(
                "%s Redis checkpointer 依赖 RediSearch(FT._LIST)。当前 Redis 非 Redis Stack，已降级。",
                colorize("[memory]", "yellow"),
            )
        else:
            logger.warning("%s Redis checkpointer 初始化失败，降级内存: %s", colorize("[memory]", "yellow"), last_exc)
    if backend == "memory":
        logger.info("%s Using in-memory checkpointer", colorize("[memory]", "green"))
    return InMemorySaver()


def parse_cli_args() -> argparse.Namespace:#解析命令行参数
    parser = argparse.ArgumentParser(description="multi-agent memory runner")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tenant-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--short-term-backend", choices=["postgres", "redis", "memory"], default=None)
    parser.add_argument("--long-term-backend", choices=["postgres", "sqlite", "disabled"], default=None)
    parser.add_argument("--long-term-scope", choices=["user", "thread"], default=None)
    parser.add_argument("--save-conversation-task", choices=["true", "false"], default=None)
    parser.add_argument("--checkpointer-backend", choices=["postgres", "redis", "memory", "auto"], default=None)
    parser.add_argument("--enable-memory", choices=["true", "false"], default=None)
    parser.add_argument("--enable-milvus", choices=["true", "false"], default=None)
    parser.add_argument("--memory-top-k", type=int, default=None)
    parser.add_argument("--once-query", type=str, default=None)
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> AppConfig:#建立运行时配置
    config = AppConfig.from_file(args.config) if args.config else AppConfig.from_file()
    overrides = {
        "tenant_id": args.tenant_id,
        "user_id": args.user_id,
        "thread_id": args.thread_id,
        "short_term_backend": args.short_term_backend,
        "long_term_backend": args.long_term_backend,
        "long_term_scope": args.long_term_scope,
        "checkpointer_backend": args.checkpointer_backend,
        "memory_top_k": args.memory_top_k,
    }
    if args.enable_memory is not None:
        overrides["enable_memory"] = args.enable_memory == "true"
    if args.enable_milvus is not None:
        overrides["enable_milvus"] = args.enable_milvus == "true"
    if args.save_conversation_task is not None:
        overrides["save_conversation_task"] = args.save_conversation_task == "true"
    config = config.with_overrides(**overrides)
    logger.info(
        "%s tenant=%s user=%s thread=%s short=%s long=%s scope=%s save_task=%s checkpointer=%s milvus=%s",
        colorize("[config]", "cyan"),
        config.tenant_id,
        config.user_id,
        config.thread_id,
        config.short_term_backend,
        config.long_term_backend,
        config.long_term_scope,
        config.save_conversation_task,
        config.checkpointer_backend,
        config.enable_milvus,
    )
    return config


@dataclass(frozen=True)
class AgentBundle:#代理集合
    intent_router: any#意图路由器
    planner: any#计划器
    scout_web: any#网络侦察
    scout_local: any#本地侦察
    evidence_judge: any#证据判断器
    analyst: any#分析师
    direct_responder: any#直接响应器
    writer: any#写入器


def build_agent(model: str, api_key: str, base_url: str, prompt_key: str, temperature: float, tools: list):#建立单个代理
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)
    prompt = PROMPTS[prompt_key]
    return create_agent(model=llm, tools=tools, system_prompt=prompt)


def build_agents(model: str, api_key: str, base_url: str, config: AppConfig) -> AgentBundle:#建立所有代理
    rag_config = RAGConfig(
        milvus_host=config.milvus_host,#Milvus 主机
        milvus_port=config.milvus_port,#Milvus 端口
        collection_name=config.milvus_collection,#Milvus 集合名称
    )
    init_rag_system(api_key=api_key, config=rag_config)#初始化 RAG 系统
    # 去掉每个 Agent 强制绑定的 tools，只做信息抽取，降低 System Prompt 长度
    return AgentBundle(
        intent_router=build_agent(model, api_key, base_url, "intent_router", 0.0, []),
        planner=build_agent(model, api_key, base_url, "plan", 0.3, []),
        scout_web=build_agent(model, api_key, base_url, "web_search", 0.4, []),
        scout_local=build_agent(model, api_key, base_url, "local_rag", 0.4, []),
        evidence_judge=build_agent(model, api_key, base_url, "deep_dive", 0.2, []),
        analyst=build_agent(model, api_key, base_url, "analyze", 0.3, []),
        direct_responder=build_agent(model, api_key, base_url, "direct_answer", 0.2, []),
        writer=build_agent(model, api_key, base_url, "write", 0.4, []),
    )


def run_query(app, config: AppConfig, query: str, return_state: bool = False):#运行查询
    """同步执行一次 Workflow 查询，并完成记忆读取与持久化。

        该函数是 Workflow 的同步执行入口，负责组织一次完整的查询流程。
        在执行前，会根据当前用户、会话及租户信息读取相关历史记忆，
        构建个性化 Prompt 上下文，并注入到 Workflow 初始状态中。

        Workflow 执行完成后，函数会提取最终回答，并将本轮问答作为新的
        对话记忆持久化存储，供后续查询进行上下文增强。

        查询生命周期如下：

            读取历史记忆
                ↓
            构建 Workflow 初始状态
                ↓
            执行 LangGraph Workflow
                ↓
            获取最终回答
                ↓
            持久化本轮对话记忆
                ↓
            返回结果（可选返回完整状态）

        Args:
            app:
                已编译完成的 LangGraph Workflow 实例。

            config:
                应用运行配置，包含用户信息、线程信息、租户信息、
                最大迭代次数以及记忆相关配置。

            query:
                用户输入的问题。

            return_state:
                是否同时返回 Workflow 最终状态。
                - False：仅返回最终回答（默认）。
                - True：返回 (final_answer, workflow_state)。

        Returns:
            str:
                Workflow 生成的最终回答。

            tuple[str, dict]:
                当 ``return_state=True`` 时，返回
                ``(final_answer, workflow_state)``，
                其中 workflow_state 为 Workflow 执行结束后的完整状态。

        Notes:
            - 历史记忆读取失败不会中断 Workflow，仅记录日志并继续执行。
            - 记忆持久化失败不会影响本次查询结果，仅记录日志。
            - Workflow 状态通过 `app.invoke()` 同步执行完成。
        """
    memory_context = ""
    if MEMORY_MANAGER:
        try:
            memory_context = MEMORY_MANAGER.build_personalized_prompt_context(
                user_id=config.user_id,
                thread_id=config.thread_id,
                query=query,
                tenant_id=config.tenant_id,
                max_memories=config.memory_top_k,
            )
        except Exception as exc:
            logger.warning("%s 读取记忆失败，忽略本轮注入: %s", colorize("[memory]", "yellow"), exc)
    state = create_initial_state(
        query=query,
        max_iterations=config.max_iterations,
        user_id=config.user_id,
        tenant_id=config.tenant_id,
        memory_context=memory_context,
    )
    result = app.invoke(
        state,
        {"configurable": {"thread_id": config.thread_id}},
    )
    final = result["final"]
    if MEMORY_MANAGER:
        try:
            MEMORY_MANAGER.persist_turn(
                tenant_id=config.tenant_id,
                user_id=config.user_id,
                thread_id=config.thread_id,
                query=query,
                answer=final,
            )
        except Exception as exc:
            logger.warning("%s 持久化记忆失败，已跳过: %s", colorize("[memory]", "yellow"), exc)
    if return_state:
        return final, result
    return final


def read_user_input(prompt: str = "你: "):#读取用户输入
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print(prompt, end="", flush=True)
        raw = sys.stdin.buffer.readline()
        if raw == b"":
            raise EOFError
        encoding = sys.stdin.encoding or "utf-8"
        recovered = raw.decode(encoding, errors="replace").rstrip("\r\n")
        logger.warning("%s 检测到输入编码异常，已使用容错解码。", colorize("[input]", "yellow"))
        return recovered


def main():
    """应用程序主入口，负责初始化系统并启动 Workflow 服务。

        该函数负责完成整个应用的生命周期管理，包括日志初始化、
        命令行参数解析、运行配置构建、Memory 管理器初始化、
        Agent 创建、Checkpointer 初始化以及 LangGraph Workflow
        的构建。

        应用启动后支持两种运行模式：

            1. 单次查询模式（--once-query）
               执行一次 Workflow 查询后立即退出。

            2. 交互模式（CLI）
               持续读取用户输入，循环执行 Workflow，直到用户主动退出。

        在交互模式下，还支持 Memory 调试命令，例如查看记忆统计信息
        或最近一次记忆检索过程。

        应用退出时，会自动关闭 Checkpointer 等需要释放的资源，
        保证数据库连接、Redis 连接等能够正常清理。

        Application 生命周期如下：

            初始化日志
                ↓
            解析命令行参数
                ↓
            构建运行配置
                ↓
            初始化 Memory Manager
                ↓
            创建 LLM Agents
                ↓
            初始化 Checkpointer
                ↓
            构建 LangGraph Workflow
                ↓
            单次执行 或 进入交互循环
                ↓
            释放资源并退出

        Notes:
            - `--once-query` 模式执行一次查询后立即结束程序。
            - 默认进入交互模式，持续响应用户输入。
            - `/memory` 用于查看 Memory 统计信息。
            - `/memory-trace` 用于查看最近一次 Memory 检索过程。
            - 程序退出时会自动关闭 Checkpointer Context，释放持久化资源。
        """
    global MEMORY_MANAGER
    global CHECKPOINTER_CONTEXT
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_cli_args()
    config = build_runtime_config(args)
    MEMORY_MANAGER = build_memory_manager(config)
    agents = build_agents(config.model, config.api_key, config.base_url, config)
    checkpointer = build_checkpointer(config)
    app = build_workflow_app(agents, checkpointer)
    if args.once_query:
        response = run_query(app, config, args.once_query)
        print(f"\nAI: {response}\n")
    else:
        while True:
            try:
                query = read_user_input("你: ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"quit", "exit", "退出"}:
                break
            if query.lower() in {"/memory", "memory-status"} and MEMORY_MANAGER:
                print(json.dumps(MEMORY_MANAGER.get_memory_stats(config.user_id), indent=2))
                continue
            if query.lower() in {"/memory-trace", "memory-trace"} and MEMORY_MANAGER:
                print(json.dumps(MEMORY_MANAGER.get_last_trace(), indent=2))
                continue
            response = run_query(app, config, query)
            print(f"\nAI: {response}\n")
    if CHECKPOINTER_CONTEXT:
        CHECKPOINTER_CONTEXT.__exit__(None, None, None)
        CHECKPOINTER_CONTEXT = None


if __name__ == "__main__":
    main()
