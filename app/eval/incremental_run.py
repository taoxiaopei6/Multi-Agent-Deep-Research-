"""
增量式评测运行器

用法：
  # 首次运行 (v1.0)
  python -m app.eval.incremental_run --cases MR-01,TA-01,KQ-01 --notes "初始10条基线"

  # 修复后重新运行 (v1.1)
  python -m app.eval.incremental_run --cases MR-01,TA-01,KQ-01 --notes "修复Planner问题"

  # 对比两个版本
  python -m app.eval.incremental_run --compare v1.0 v1.1

  # 查看历史
  python -m app.eval.incremental_run --history

  # 记录问题
  python -m app.eval.incremental_run --add-issue "MR-01" "引用质量偏低" "发现的引用来源单一"
"""

import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.eval.runner import (
    run_single_case,
    _load_raw,
    _raw_dir,
    count_llm_calls,
    _extract_evidence,
)
from app.eval.test_cases import all_test_cases, get_case_by_id
from app.mult_agents.config import AppConfig
from app.mult_agents.main import (
    build_agents,
    build_checkpointer,
    build_memory_manager,
    run_query,
)
from app.eval.tracker import (
    EvalTracker,
    CaseRunResult,
    record_issue,
    auto_issue_id,
    IssueRecord,
)

from app.eval.scorer import LLMScorer, EvalResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("incremental_eval")

OUTPUT_DIR = str(_PROJECT_ROOT / "output" / "eval")
TRACKING_DIR = str(_PROJECT_ROOT / "output" / "eval_tracking")


def _init_pipeline():
    """初始化一次 pipeline，复用。"""
    config = AppConfig.from_file()
    config = config.with_overrides(
        max_iterations=0,
        enable_memory=False,
        checkpointer_backend="memory",
    )
    agents = build_agents(config.model, config.api_key, config.base_url, config)
    checkpointer = build_checkpointer(config)
    from app.mult_agents.graph import build_app as build_workflow_app
    app = build_workflow_app(agents, checkpointer)
    scorer = LLMScorer(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    return app, config, scorer


def run_incremental(case_ids: list[str], notes: str = ""):
    """执行一轮增量评测。"""
    tracker = EvalTracker(TRACKING_DIR)
    version = tracker.start_run(notes)

    cases = [c for c in all_test_cases() if c.id in case_ids]
    found = {c.id for c in cases}
    missing = [cid for cid in case_ids if cid not in found]
    if missing:
        logger.warning("未找到用例: %s", missing)
    if not cases:
        logger.error("没有可用的用例")
        return

    logger.info("本轮评测 %d 条用例: %s", len(cases), [c.id for c in cases])

    app, config, scorer = _init_pipeline()
    total_start = time.time()

    for i, case in enumerate(cases, 1):
        logger.info("[%d/%d] 执行: %s — %s", i, len(cases), case.id, case.query[:50])
        case_start = time.time()

        report = run_single_case(app, config, case, OUTPUT_DIR, force_rerun=True)
        elapsed = time.time() - case_start

        result = CaseRunResult(
            case_id=case.id,
            category=case.category,
            query=case.query,
            duration=round(elapsed, 1),
        )

        if report and len(report.strip()) >= 50:
            # 有报告 → 评分
            result.has_report = True
            result.report_length = len(report)

            # 从缓存读取搜索统计
            raw = _load_raw(case.id, OUTPUT_DIR)
            if raw:
                result.bocha_calls = raw.get("bocha_calls", 0)
                result.llm_calls = raw.get("llm_calls", 0)
                ev = raw.get("evidence", {})
                result.evidence_count = len(ev.get("evidence_pool", []))

            # LLM 三维评分
            try:
                sr: EvalResult = scorer.score(
                    query=case.query,
                    report=report,
                    expected_topics=case.expected_topics,
                    min_sources=case.min_sources,
                    case_id=case.id,
                    category=case.category,
                )
                result.score = sr.total_score
                result.completeness_score = sr.completeness.score
                result.citation_score = sr.citation_quality.score
                result.relevance_score = sr.relevance.score
                logger.info("  → 评分: %.2f (完整度: %.1f 引用: %.1f 相关度: %.1f)",
                            sr.total_score, sr.completeness.score,
                            sr.citation_quality.score, sr.relevance.score)
            except Exception as e:
                logger.error("  → 评分失败: %s", e)
                result.score = 0.0
        else:
            logger.warning("  → 报告为空或过短")
            result.has_report = False
            result.score = 0.0

        tracker.add_result(result)

    tracker.finish_run()

    total_elapsed = time.time() - total_start
    logger.info("本轮评测总耗时: %.1fs", total_elapsed)
    tracker.print_history()


def run_with_retry(case_ids: list[str], max_retries: int = 3, notes: str = ""):
    """带重试机制的增量评测：失败用例自动重试。"""
    tracker = EvalTracker(TRACKING_DIR)
    version = tracker.start_run(notes)

    cases = [c for c in all_test_cases() if c.id in case_ids]
    if not cases:
        logger.error("没有可用的用例")
        return

    logger.info("本轮评测 %d 条用例，最大重试 %d 次", len(cases), max_retries)

    app, config, scorer = _init_pipeline()
    total_start = time.time()
    remaining = list(cases)

    for attempt in range(1, max_retries + 1):
        if not remaining:
            break
        logger.info("--- 第 %d 轮执行 (%d 条) ---", attempt, len(remaining))
        next_remaining = []

        for case in remaining:
            case_start = time.time()
            logger.info("  [%s] 执行...", case.id)
            report = run_single_case(app, config, case, OUTPUT_DIR, force_rerun=True)
            elapsed = time.time() - case_start

            result = CaseRunResult(
                case_id=case.id,
                category=case.category,
                query=case.query,
                duration=round(elapsed, 1),
            )

            if report and len(report.strip()) >= 50:
                result.has_report = True
                result.report_length = len(report)
                raw = _load_raw(case.id, OUTPUT_DIR)
                if raw:
                    result.bocha_calls = raw.get("bocha_calls", 0)
                    result.llm_calls = raw.get("llm_calls", 0)
                    ev = raw.get("evidence", {})
                    result.evidence_count = len(ev.get("evidence_pool", []))
                try:
                    sr = scorer.score(
                        query=case.query,
                        report=report,
                        expected_topics=case.expected_topics,
                        min_sources=case.min_sources,
                        case_id=case.id,
                        category=case.category,
                    )
                    result.score = sr.total_score
                    result.completeness_score = sr.completeness.score
                    result.citation_score = sr.citation_quality.score
                    result.relevance_score = sr.relevance.score
                    logger.info("  → %.2f | 完整度: %.1f 引用: %.1f 相关度: %.1f",
                                sr.total_score, sr.completeness.score,
                                sr.citation_quality.score, sr.relevance.score)
                except Exception as e:
                    logger.error("  → 评分失败: %s", e)
                    next_remaining.append(case)
                    continue
            else:
                logger.warning("  → 报告为空")
                next_remaining.append(case)
                continue

            tracker.add_result(result)

        remaining = next_remaining
        if remaining:
            logger.info("仍有 %d 条失败，准备重试...", len(remaining))

    if remaining:
        logger.error("重试耗尽，仍有 %d 条失败: %s", len(remaining), [c.id for c in remaining])

    tracker.finish_run()
    total_elapsed = time.time() - total_start
    logger.info("总耗时: %.1fs", total_elapsed)
    tracker.print_history()


def show_history():
    """显示评测历史。"""
    tracker = EvalTracker(TRACKING_DIR)
    tracker.print_history()
    issues_open = tracker.get_issues(status="open") + tracker.get_issues(status="fixing")
    if issues_open:
        print(f"\n待处理问题 ({len(issues_open)}):")
        for i in issues_open:
            print(f"  {i['id']} | {i['case_id']} | {i['title']} [{i['severity']}]")
    issues_fixed = tracker.get_issues(status="verified")
    if issues_fixed:
        print(f"\n已验证修复 ({len(issues_fixed)}):")
        for i in issues_fixed:
            print(f"  {i['id']} | {i['case_id']} | {i['title']}")


def compare_versions(v1: str, v2: str):
    """对比两个版本。"""
    tracker = EvalTracker(TRACKING_DIR)
    result = tracker.compare(v1, v2)
    if "error" in result:
        print(result["error"])
        return
    print("\n" + "=" * 60)
    print(f"版本对比: {v1} → {v2}")
    print("=" * 60)
    print(f"总分: {result['score_delta']:+.2f}")
    print(f"完整度: {result['completeness_delta']:+.2f}")
    print(f"引用质量: {result['citation_delta']:+.2f}")
    print(f"相关度: {result['relevance_delta']:+.2f}")
    print("-" * 40)
    print("逐条变化:")
    for cid, delta in sorted(result.get("cases", {}).items()):
        d = delta["score_delta"]
        sign = "+" if d > 0 else ""
        print(f"  {cid}: {delta['score_before']:.2f} → {delta['score_after']:.2f} ({sign}{d:.2f})")
    print()


def add_issue(case_id: str, title: str, description: str = "", severity: str = "medium"):
    """快速添加问题记录。"""
    tracker = EvalTracker(TRACKING_DIR)
    record_issue(tracker, case_id, title, description or title, severity=severity)


def add_changelog(version: str, changes: list, issues_fixed: list | None = None):
    """添加版本变更记录。"""
    tracker = EvalTracker(TRACKING_DIR)
    tracker.add_changelog(version, changes, issues_fixed)


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="增量式评测运行器")
    parser.add_argument("--cases", type=str, default=None,
                        help="用例 ID 逗号分隔")
    parser.add_argument("--notes", type=str, default="",
                        help="本轮评测说明")
    parser.add_argument("--history", action="store_true",
                        help="显示评测历史")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("v1", "v2"),
                        help="对比两个版本")
    parser.add_argument("--add-issue", type=str, nargs="+", metavar=("CASE_ID", "TITLE"),
                        help="记录问题: CASE_ID TITLE [描述...]")
    parser.add_argument("--severity", type=str, default="medium",
                        choices=["critical", "high", "medium", "low", "enhancement"])
    parser.add_argument("--retry", type=int, default=1,
                        help="失败自动重试次数 (默认1=不重试)")
    parser.add_argument("--default-10", action="store_true",
                        help="使用默认10条初始用例")

    args = parser.parse_args()

    if args.history:
        show_history()
        return

    if args.compare:
        compare_versions(args.compare[0], args.compare[1])
        return

    if args.add_issue:
        case_id = args.add_issue[0]
        title = args.add_issue[1]
        desc = " ".join(args.add_issue[2:]) if len(args.add_issue) > 2 else ""
        add_issue(case_id, title, desc, severity=args.severity)
        print(f"问题已记录: {case_id} — {title}")
        return

    # 确定用例
    if args.default_10:
        case_ids = ["MR-01", "MR-04", "TA-01", "TA-04", "TA-08",
                     "KQ-01", "KQ-04", "KQ-07", "MR-07", "TA-06"]
    elif args.cases:
        case_ids = [c.strip() for c in args.cases.split(",")]
    else:
        print("请指定 --cases 或用 --default-10")
        return

    if args.retry > 1:
        run_with_retry(case_ids, max_retries=args.retry, notes=args.notes)
    else:
        run_incremental(case_ids, notes=args.notes)


if __name__ == "__main__":
    main()
