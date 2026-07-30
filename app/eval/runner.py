"""评测 Runner：批量执行测试用例，调用评分器，输出 Markdown 报告。

重要：每条用例的 pipeline 输出（报告 + 搜索证据）会持久化到 output/eval/raw/，
重新评分直接从缓存读取，不消耗 Bocha 调用次数。
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.mult_agents.config import AppConfig
from app.mult_agents.graph import build_app as build_workflow_app
from app.mult_agents.main import (
    build_agents,
    build_checkpointer,
    build_memory_manager,
    run_query,
)
from app.eval.test_cases import all_test_cases, get_case_by_id, TestCase
from app.eval.scorer import LLMScorer, generate_eval_report, EvalResult

logger = logging.getLogger("eval")


# ── 缓存路径管理 ──

def _raw_dir(output_dir: str) -> Path:
    return Path(output_dir) / "raw"


def _raw_path(case_id: str, output_dir: str) -> Path:
    return _raw_dir(output_dir) / f"{case_id}.json"


def _load_raw(case_id: str, output_dir: str) -> Optional[dict]:
    """从缓存加载原始 pipeline 输出，不存在返回 None。"""
    path = _raw_path(case_id, output_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("[%s] 从缓存读取原始结果: %s", case_id, path)
        return data
    except Exception as e:
        logger.warning("[%s] 缓存读取失败，将重新运行: %s", case_id, e)
        return None


def _save_raw(case_id: str, data: dict, output_dir: str):
    """持久化原始 pipeline 输出。"""
    path = _raw_path(case_id, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("[%s] 原始结果已持久化: %s", case_id, path)


def _load_score(case_id: str, output_dir: str) -> Optional[dict]:
    """从缓存加载评分结果，不存在返回 None。"""
    path = _raw_path(case_id, output_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("score")
    except Exception:
        return None


# ── 核心逻辑 ──

def count_llm_calls(messages: list) -> int:
    """统计一条 pipeline 执行中的 LLM 调用次数。

    每个 AI message 代表一次模型 invoke，不受其他消息类型（User、Tool 等）影响。
    """
    return sum(1 for msg in messages if getattr(msg, "type", "") == "ai")

def _extract_evidence(state: dict) -> dict:
    """从完整 ResearchState 中提取搜索证据链，用于持久化。"""
    return {
        "plan": state.get("plan", ""),
        "search_plan": state.get("search_plan", []),
        "sub_questions": state.get("sub_questions", []),
        "web_evidence": state.get("web_evidence", []),
        "local_evidence": state.get("local_evidence", []),
        "evidence_pool": state.get("evidence_pool", []),
        "source_index": state.get("source_index", []),
        "audit_flags": state.get("audit_flags", []),
        "findings": state.get("findings", []),
        "claim_map": state.get("claim_map", []),
        "web_search_trace": state.get("web_search_trace", []),
        "local_rag_trace": state.get("local_rag_trace", []),
        "supplementary_queries": state.get("supplementary_queries", []),
        "web_retrieval_stats": state.get("web_retrieval_stats", {}),
        "local_retrieval_stats": state.get("local_retrieval_stats", {}),
        "iteration": state.get("iteration", 0),
        "needs_more_research": state.get("needs_more_research", False),
        "missing_gaps": state.get("missing_gaps", []),
        "planner_stats": state.get("planner_stats", {}),
    }


def run_single_case(
    app,
    config: AppConfig,
    case: TestCase,
    output_dir: str,
    force_rerun: bool = False,
) -> Optional[str]:
    """对单条用例执行 pipeline，优先使用缓存。

    Args:
        force_rerun: True 表示强制重新跑 pipeline（忽略缓存）

    Returns:
        report 字符串，失败返回 None
    """
    # 先检查缓存
    if not force_rerun:
        cached = _load_raw(case.id, output_dir)
        if cached:
            return cached.get("report")

    logger.info("=" * 60)
    logger.info("[%s] 执行 pipeline: %s", case.id, case.query)
    logger.info("难度: %s | 期望主题: %s", case.difficulty, ", ".join(case.expected_topics))
    logger.info("=" * 60)

    # 每个 Benchmark Case 使用独立 thread_id，防止 Checkpointer 跨用例污染
    thread_id = f"benchmark_{case.id}_{int(time.time())}"
    config = config.with_overrides(thread_id=thread_id)

    start = time.time()
    try:
        report, state = run_query(app, config, case.query, return_state=True)
        elapsed = time.time() - start
        logger.info("[%s] 完成耗时: %.1fs | 报告长度: %d 字", case.id, elapsed, len(report))

        # 提取搜索证据链
        evidence = _extract_evidence(state)
        bocha_calls = len(state.get("web_search_trace", []))
        logger.info("[%s] 搜索证据: Bocha调用=%d次 | web_evidence=%d条 | local_evidence=%d条",
                     case.id, bocha_calls, len(evidence["web_evidence"]), len(evidence["local_evidence"]))
        total_searches = sum(t.get("raw_count", 0) for t in evidence.get("web_search_trace", []))

        # 统计 LLM 调用：AI message 数 = 模型调用次数
        llm_calls = count_llm_calls(state.get("messages", []))

        # 持久化：报告 + 完整搜索证据 + 调用统计
        _save_raw(case.id, {
            "report": report,
            "query": case.query,
            "elapsed": round(elapsed, 1),
            "bocha_calls": bocha_calls,
            "llm_calls": llm_calls,
            "reflection_triggered": state.get("iteration", 0) > 0,
            "total_raw_records": total_searches,
            "evidence": evidence,
        }, output_dir)
        return report

    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] 执行失败 (%.1fs): %s", case.id, elapsed, e)
        _save_raw(case.id, {
            "error": str(e), "query": case.query, "elapsed": round(elapsed, 1),
        }, output_dir)
        return None


def run_eval(
    case_ids: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    max_iterations: int = 2,
    enable_memory: bool = False,
    force_rerun: bool = False,
    re_score: Optional[list[str]] = None,
) -> list[EvalResult]:
    """批量评测主入口。

    自动跳过已有缓存的用例（除非 force_rerun=True）。

    Args:
        re_score: 指定要重新评分的用例 ID 列表（从缓存读报告，重新 LLM 评分）
    """
    # ── 筛选用例 ──
    all_cases = all_test_cases()
    if case_ids:
        cases = [c for c in all_cases if c.id in case_ids]
        if not cases:
            logger.warning("未找到指定用例: %s", case_ids)
            return []
    elif categories:
        cases = [c for c in all_cases if c.category in categories]
    else:
        cases = all_cases

    if output_dir is None:
        output_dir = str(_PROJECT_ROOT / "output" / "eval")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 区分需要跑 pipeline 和只需要评分的 ──
    need_pipeline = []
    need_score_only = []
    cached_results = []

    for c in cases:
        raw = _load_raw(c.id, output_dir)

        # 如果指定了 re_score，强制重新评分
        if re_score and c.id in re_score:
            if raw and raw.get("report"):
                need_score_only.append(c)
            else:
                logger.warning("[%s] 指定了 re_score 但没有缓存报告，需重新跑 pipeline", c.id)
                need_pipeline.append(c)
            continue

        if raw and raw.get("report"):
            # 已有缓存 → 检查是否已有评分
            score = raw.get("score")
            if score:
                # 全部已有 → 跳过
                cached_results.append(_reconstruct_result(c, raw))
                logger.info("[%s] 已有完整结果，跳过", c.id)
            else:
                need_score_only.append(c)
        elif force_rerun or not raw:
            need_pipeline.append(c)
        else:
            # 缓存存在但无 report（例如之前执行出错）
            logger.warning("[%s] 缓存不完整，重新执行", c.id)
            need_pipeline.append(c)

    if not need_pipeline and not need_score_only:
        logger.info("所有用例已有完整结果，跳过执行和评分")
        return cached_results

    # ── 初始化 pipeline（只在需要跑的时候）──
    app, config = None, None
    if need_pipeline:
        logger.info("初始化多 Agent 系统...")
        config = AppConfig.from_file()
        config = config.with_overrides(
            max_iterations=max_iterations,
            enable_memory=enable_memory,
        )
        agents = build_agents(config.model, config.api_key, config.base_url, config)
        checkpointer = build_checkpointer(config)
        app = build_workflow_app(agents, checkpointer)
        logger.info("多 Agent 系统初始化完成 | 需执行 %d 条用例", len(need_pipeline))

    # ── 初始化评分器 ──
    scorer = LLMScorer(
        model=config.model if config else AppConfig.from_file().model,
        api_key=config.api_key if config else AppConfig.from_file().api_key,
        base_url=config.base_url if config else AppConfig.from_file().base_url,
    )

    # ── 跑 pipeline ──
    results: list[EvalResult] = cached_results
    total_start = time.time()
    total_bocha_calls_saved = 0

    if need_pipeline:
        logger.info("开始执行 %d 条 pipeline（将消耗 Bocha 额度）", len(need_pipeline))
        for i, case in enumerate(need_pipeline, 1):
            logger.info("[%d/%d] 执行: %s", i, len(need_pipeline), case.id)
            report = run_single_case(app, config, case, output_dir, force_rerun=force_rerun)
            if report and len(report.strip()) >= 50:
                need_score_only.append(case)
            else:
                logger.warning("[%s] 报告过短或为空，跳过评分", case.id)

    # ── 评分 ──
    if need_score_only:
        saved = len(need_score_only)
        logger.info("开始评分 %d 条（0 次 Bocha 调用，从缓存读取报告）", saved)
        total_bocha_calls_saved += saved

        for case in need_score_only:
            raw = _load_raw(case.id, output_dir)
            report = raw.get("report", "") if raw else ""
            if not report or len(report.strip()) < 50:
                logger.warning("[%s] 报告不可用，跳过评分", case.id)
                continue

            try:
                result = scorer.score(
                    query=case.query,
                    report=report,
                    expected_topics=case.expected_topics,
                    min_sources=case.min_sources,
                    case_id=case.id,
                    category=case.category,
                )
                result.raw_report = report
                results.append(result)

                # 把评分结果追加到缓存文件
                if raw is not None:
                    raw["score"] = result.to_dict()
                    _save_raw(case.id, raw, output_dir)

                logger.info(
                    "[%s] 评分完成: 完整度=%.1f 引用=%.1f 相关度=%.1f 总分=%.2f",
                    case.id,
                    result.completeness.score,
                    result.citation_quality.score,
                    result.relevance.score,
                    result.total_score,
                )
            except Exception as e:
                logger.error("[%s] 评分失败: %s", case.id, e)

    # ── 汇总报告 ──
    total_elapsed = time.time() - total_start
    logger.info("全部完成: %d 条, 耗时 %.1fs", len(results), total_elapsed)
    if total_bocha_calls_saved > 0:
        logger.info("本次省了约 %d 次 Bocha 调用（从缓存读报告）", total_bocha_calls_saved)

    report_path = str(Path(output_dir) / f"eval_report_{time.strftime('%Y%m%d_%H%M%S')}.md")
    generate_eval_report(results, report_path)

    json_path = report_path.replace(".md", ".json")
    json.dump(
        {
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": len(results),
            "elapsed_seconds": round(total_elapsed, 1),
            "bocha_calls_saved": total_bocha_calls_saved,
            "results": [r.to_dict() for r in results],
        },
        Path(json_path).open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    logger.info("JSON 报告已写入: %s", json_path)

    return results


def _reconstruct_result(case: TestCase, raw: dict) -> EvalResult:
    """从缓存重建 EvalResult 对象（仅用于统计，不包含详细评分明细）。"""
    from app.eval.scorer import ScoreDetail
    score = raw.get("score", {})
    return EvalResult(
        case_id=case.id,
        query=case.query,
        category=case.category,
        completeness=ScoreDetail(
            score=score.get("completeness", {}).get("score", 0),
            reason=score.get("completeness", {}).get("reason", "缓存结果"),
        ),
        citation_quality=ScoreDetail(
            score=score.get("citation_quality", {}).get("score", 0),
            reason=score.get("citation_quality", {}).get("reason", "缓存结果"),
        ),
        relevance=ScoreDetail(
            score=score.get("relevance", {}).get("score", 0),
            reason=score.get("relevance", {}).get("reason", "缓存结果"),
        ),
        total_score=score.get("total_score", 0),
        raw_report=raw.get("report", ""),
    )


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="多 Agent 研报系统 — 评测工具")
    parser.add_argument("--cases", type=str, default=None,
                        help="指定用例 ID，逗号分隔，如 MR-01,TA-02")
    parser.add_argument("--categories", type=str, default=None,
                        help="按类别过滤，逗号分隔，如 market_research,tech_analysis")
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="pipeline 最大迭代轮次，默认 2")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="报告输出目录")
    parser.add_argument("--list-cases", action="store_true",
                        help="列出所有可用用例")
    parser.add_argument("--force-rerun", action="store_true",
                        help="强制重新跑 pipeline（忽略缓存）")
    parser.add_argument("--re-score", type=str, default=None,
                        help="重新评分指定用例（从缓存读报告，不跑 pipeline），逗号分隔")
    parser.add_argument("--status", action="store_true",
                        help="显示各用例的缓存状态")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.list_cases:
        print(f"{'ID':<10} {'类别':<18} {'难度':<8} {'问题'}")
        print("-" * 80)
        for c in all_test_cases():
            cat_label = {"market_research": "市场调研", "tech_analysis": "技术分析", "knowledge_qa": "知识问答"}.get(c.category, c.category)
            print(f"{c.id:<10} {cat_label:<18} {c.difficulty:<8} {c.query}")
        return

    output_dir = args.output_dir or str(_PROJECT_ROOT / "output" / "eval")

    if args.status:
        print(f"{'ID':<10} {'report':<8} {'score':<8} {'status'}")
        print("-" * 60)
        for c in all_test_cases():
            raw = _load_raw(c.id, output_dir)
            if raw and raw.get("report"):
                has_report = "Y"
                has_score = "Y" if raw.get("score") else "N"
                status = "complete" if raw.get("score") else "need_score"
            else:
                has_report = "-"
                has_score = "-"
                status = "pending"
            print(f"{c.id:<10} {has_report:<8} {has_score:<8} {status}")
        return

    case_ids = args.cases.split(",") if args.cases else None
    re_score = args.re_score.split(",") if args.re_score else None
    categories = args.categories.split(",") if args.categories else None

    logger.info("=" * 60)
    logger.info("多 Agent 研报系统 — 评测启动")
    logger.info("用例: %s", args.cases or "全部")
    logger.info("类别: %s", args.categories or "全部")
    logger.info("最大迭代: %d", args.max_iterations)
    if args.force_rerun:
        logger.info("模式: 强制重跑（忽略缓存）")
    if args.re_score:
        logger.info("模式: 仅重新评分（节省 Bocha 额度）")
    logger.info("=" * 60)

    run_eval(
        case_ids=case_ids,
        categories=categories,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        force_rerun=args.force_rerun,
        re_score=re_score,
    )


if __name__ == "__main__":
    main()
