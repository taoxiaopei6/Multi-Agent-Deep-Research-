"""
增量式评测跟踪器

记录每次评测的版本、分数、发现的问题、修复措施，形成一个可追溯的迭代历史。

数据存储：
  output/eval_tracking/
  ├── tracking.json        # 主跟踪文件，记录所有历史
  ├── issues.md            # 问题看板，人可读
  └── v<version>/          # 每次运行的快照
      ├── summary.json
      └── raw/             # 原始结果（引用 runner 的缓存）
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("eval.tracker")

TRACKING_DIR = Path(__file__).resolve().parents[2] / "output" / "eval_tracking"

# ── 数据结构 ──

@dataclass
class CaseRunResult:
    """单条用例在单次运行中的结果。"""
    case_id: str
    category: str
    query: str
    score: float = 0.0
    completeness_score: float = 0.0
    citation_score: float = 0.0
    relevance_score: float = 0.0
    evidence_count: int = 0
    bocha_calls: int = 0
    llm_calls: int = 0
    duration: float = 0.0
    report_length: int = 0
    has_report: bool = False
    error: str = ""


@dataclass
class RunSummary:
    """单次评测运行的总体摘要。"""
    version: str
    timestamp: str
    notes: str = ""
    total_cases: int = 0
    passed: int = 0
    avg_score: float = 0.0
    avg_completeness: float = 0.0
    avg_citation: float = 0.0
    avg_relevance: float = 0.0
    total_duration: float = 0.0
    total_bocha_calls: int = 0
    total_llm_calls: int = 0


@dataclass
class IssueRecord:
    """一个问题记录，从发现到修复全链路。"""
    id: str
    case_id: str
    title: str
    description: str
    discovered_in: str          # 在哪个版本发现的
    root_cause: str = ""
    fix_applied: str = ""
    fixed_in: str = ""           # 在哪个版本修复的
    verified_in: str = ""        # 在哪个版本验证通过的
    status: str = "open"         # open → fixing → fixed → verified
    severity: str = "medium"     # critical / high / medium / low / enhancement


@dataclass
class ChangelogEntry:
    """版本变更记录。"""
    version: str
    date: str
    changes: list = field(default_factory=list)
    issues_fixed: list = field(default_factory=list)


@dataclass
class TrackingData:
    """主跟踪数据。"""
    runs: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    changelog: list = field(default_factory=list)
    current_version: str = "v0.0"


# ── 序列化辅助 ──

def _to_dict(obj):
    """把 dataclass 转 dict，跳过 None 和空字符串。"""
    d = asdict(obj)
    return {k: v for k, v in d.items() if v is not None and v != ""}


# ── 主跟踪器 ──

class EvalTracker:
    """评测跟踪器，管理多轮迭代评测的历史记录。"""

    def __init__(self, tracking_dir: str | Path | None = None):
        self.tracking_dir = Path(tracking_dir) if tracking_dir else TRACKING_DIR
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self._data = self._load_or_init()
        self._current_run_cases: list[CaseRunResult] = []

    def _tracking_path(self) -> Path:
        return self.tracking_dir / "tracking.json"

    def _issues_path(self) -> Path:
        return self.tracking_dir / "issues.md"

    def _load_or_init(self) -> TrackingData:
        path = self._tracking_path()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return TrackingData(
                    runs=raw.get("runs", []),
                    issues=raw.get("issues", []),
                    changelog=raw.get("changelog", []),
                    current_version=raw.get("current_version", "v0.0"),
                )
            except Exception as exc:
                logger.warning("加载 tracking.json 失败，重新初始化: %s", exc)
        return TrackingData()

    def _save(self):
        path = self._tracking_path()
        path.write_text(
            json.dumps({
                "current_version": self._data.current_version,
                "runs": self._data.runs,
                "issues": self._data.issues,
                "changelog": self._data.changelog,
            }, indent=2),
            encoding="utf-8",
        )
        self._render_issues_md()

    def _next_version(self) -> str:
        """根据已有 runs 自动递增版本号。"""
        parts = self._data.current_version.lstrip("v").split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return f"v{major}.{minor + 1}"

    def start_run(self, notes: str = "") -> str:
        """开始一次新的评测运行，返回版本号。"""
        version = self._next_version()
        self._data.current_version = version
        self._current_run_cases = []
        logger.info("=" * 50)
        logger.info("评测运行启动: %s", version)
        logger.info("说明: %s", notes)
        logger.info("=" * 50)
        return version

    def add_result(self, result: CaseRunResult):
        """添加一条用例的结果。"""
        self._current_run_cases.append(result)

    def finish_run(self):
        """完成当前运行，生成摘要并保存。"""
        if not self._current_run_cases:
            logger.warning("无用例结果，跳过保存")
            return

        cases = self._current_run_cases
        version = self._data.current_version

        total = len(cases)
        passed = sum(1 for c in cases if c.score >= 3.0 and not c.error)
        avg_score = sum(c.score for c in cases) / total
        avg_comp = sum(c.completeness_score for c in cases) / total
        avg_cite = sum(c.citation_score for c in cases) / total
        avg_rel = sum(c.relevance_score for c in cases) / total
        total_dur = sum(c.duration for c in cases)
        total_bocha = sum(c.bocha_calls for c in cases)
        total_llm = sum(c.llm_calls for c in cases)

        run_summary = {
            "version": version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "",
            "total_cases": total,
            "passed": passed,
            "avg_score": round(avg_score, 2),
            "avg_completeness": round(avg_comp, 2),
            "avg_citation": round(avg_cite, 2),
            "avg_relevance": round(avg_rel, 2),
            "total_duration": round(total_dur, 1),
            "total_bocha_calls": total_bocha,
            "total_llm_calls": total_llm,
            "cases": {c.case_id: _to_dict(c) for c in cases},
        }

        self._data.runs.append(run_summary)
        self._save()

        # 保存快照
        snap_dir = self.tracking_dir / version
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "summary.json").write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )

        # 打印总结
        logger.info("=" * 50)
        logger.info("评测完成: %s", version)
        logger.info("用例: %d | 通过: %d/%d", total, passed, total)
        logger.info("平均分: %.2f | 完整度: %.2f | 引用: %.2f | 相关度: %.2f",
                     avg_score, avg_comp, avg_cite, avg_rel)
        logger.info("总耗时: %.1fs | Bocha调用: %d | LLM调用: %d",
                     total_dur, total_bocha, total_llm)
        logger.info("=" * 50)

        self._current_run_cases = []

    def report_issue(self, issue: IssueRecord):
        """记录一个发现的问题。"""
        existing = [i for i in self._data.issues if i["id"] == issue.id]
        if existing:
            # 更新已有 issue
            idx = self._data.issues.index(existing[0])
            self._data.issues[idx] = _to_dict(issue)
        else:
            self._data.issues.append(_to_dict(issue))
        self._save()

    def get_issues(self, status: str | None = None, case_id: str | None = None) -> list[dict]:
        """查询问题列表。"""
        results = self._data.issues
        if status:
            results = [i for i in results if i.get("status") == status]
        if case_id:
            results = [i for i in results if i.get("case_id") == case_id]
        return results

    def add_changelog(self, version: str, changes: list, issues_fixed: list | None = None):
        """添加版本变更记录。"""
        entry = _to_dict(ChangelogEntry(
            version=version,
            date=datetime.now().strftime("%Y-%m-%d"),
            changes=changes,
            issues_fixed=issues_fixed or [],
        ))
        self._data.changelog.append(entry)
        self._save()

    def compare(self, v1: str, v2: str) -> dict:
        """对比两个版本的评测结果。"""
        r1 = next((r for r in self._data.runs if r["version"] == v1), None)
        r2 = next((r for r in self._data.runs if r["version"] == v2), None)
        if not r1 or not r2:
            return {"error": f"未找到版本: {v1} 或 {v2}"}

        cases1 = r1.get("cases", {})
        cases2 = r2.get("cases", {})
        deltas = {}
        for cid, c2 in cases2.items():
            c1 = cases1.get(cid, {})
            deltas[cid] = {
                "score_delta": round(c2.get("score", 0) - c1.get("score", 0), 2),
                "score_before": c1.get("score", 0),
                "score_after": c2.get("score", 0),
            }

        return {
            "v1": v1,
            "v2": v2,
            "score_delta": round(r2["avg_score"] - r1["avg_score"], 2),
            "completeness_delta": round(r2["avg_completeness"] - r1["avg_completeness"], 2),
            "citation_delta": round(r2["avg_citation"] - r1["avg_citation"], 2),
            "relevance_delta": round(r2["avg_relevance"] - r1["avg_relevance"], 2),
            "cases": deltas,
        }

    def _render_issues_md(self):
        """将问题记录渲染为 Markdown 问题看板。"""
        path = self._issues_path()
        lines = ["# 评测问题看板\n"]
        lines.append(f"> 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # 按状态分组
        for status, label in [("open", "待修复"), ("fixing", "修复中"),
                               ("fixed", "已修复待验证"), ("verified", "已验证关闭")]:
            items = [i for i in self._data.issues if i.get("status") == status]
            if not items:
                continue
            lines.append(f"## {label} ({len(items)})\n")
            lines.append("| ID | 用例 | 标题 | 严重度 | 发现版本 | 根因 | 修复措施 |")
            lines.append("|----|------|------|--------|---------|------|---------|")
            for i in items:
                lines.append(
                    f"| {i.get('id', '')} | {i.get('case_id', '')} | "
                    f"{i.get('title', '')} | {i.get('severity', '')} | "
                    f"{i.get('discovered_in', '')} | "
                    f"{i.get('root_cause', '-')} | {i.get('fix_applied', '-')} |"
                )
            lines.append("")

        lines.append("---\n")
        lines.append("## 版本记录\n")
        for entry in reversed(self._data.changelog):
            lines.append(f"### {entry['version']} ({entry['date']})")
            for change in entry.get("changes", []):
                lines.append(f"- {change}")
            for fixed in entry.get("issues_fixed", []):
                lines.append(f"  - 修复: {fixed}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def print_history(self):
        """打印评测历史。"""
        print("\n" + "=" * 60)
        print("评测迭代历史")
        print("=" * 60)
        for run in self._data.runs:
            v = run["version"]
            t = run["timestamp"][:16]
            s = run["avg_score"]
            p = run["passed"]
            n = run["total_cases"]
            print(f"  {v} ({t}) | 平均分: {s:.2f} | 通过: {p}/{n} | 耗时: {run['total_duration']:.0f}s")
        print("=" * 60)

        issues_open = len([i for i in self._data.issues if i.get("status") in ("open", "fixing")])
        issues_fixed = len([i for i in self._data.issues if i.get("status") == "verified"])
        if issues_open or issues_fixed:
            print(f"问题: {issues_open} 待修复, {issues_fixed} 已验证关闭")
        print()

    def all_versions(self) -> list[str]:
        """返回所有版本列表。"""
        return [r["version"] for r in self._data.runs]


# ── 便捷函数 ──

def auto_issue_id(tracker: EvalTracker) -> str:
    """自动生成问题 ID。"""
    existing = [i["id"] for i in tracker._data.issues if i["id"].startswith("ISS-")]
    nums = [int(i.split("-")[1]) for i in existing if i.split("-")[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"ISS-{next_num:03d}"


def record_issue(
    tracker: EvalTracker,
    case_id: str,
    title: str,
    description: str,
    severity: str = "medium",
    root_cause: str = "",
):
    """快速记录一个新问题。"""
    issue = IssueRecord(
        id=auto_issue_id(tracker),
        case_id=case_id,
        title=title,
        description=description,
        discovered_in=tracker._data.current_version,
        root_cause=root_cause,
        severity=severity,
        status="open",
    )
    tracker.report_issue(issue)
    logger.info("问题已记录: %s | %s | %s", issue.id, case_id, title)
