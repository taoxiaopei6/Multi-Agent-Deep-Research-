"""评分模块：对研报进行三维度评估（主题覆盖度 / 引用质量 / 相关性）。"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 加载项目 .env 文件（与 config.py 一致）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

logger = logging.getLogger("eval")


# ──────────────────────────────────────────────
# 评分结果数据结构
# ──────────────────────────────────────────────

@dataclass
class ScoreDetail:
    """单个维度的评分明细。"""
    score: float          # 1-5 分
    max_score: float = 5.0
    reason: str = ""      # 评分理由
    matched: list[str] = field(default_factory=list)       # 命中的主题/引用
    missed: list[str] = field(default_factory=list)        # 缺失项

    @property
    def percentage(self) -> float:
        return round(self.score / self.max_score * 100, 1)


@dataclass
class EvalResult:
    """一条用例的完整评分结果。"""
    case_id: str
    query: str
    category: str
    completeness: ScoreDetail      # 主题覆盖度
    citation_quality: ScoreDetail  # 引用质量
    relevance: ScoreDetail         # 回答相关性
    total_score: float             # 加权总分 (满分 5)
    raw_report: str = ""           # 最终报告全文

    @property
    def total_percentage(self) -> float:
        return round(self.total_score / 5.0 * 100, 1)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "category": self.category,
            "completeness": {"score": self.completeness.score, "reason": self.completeness.reason},
            "citation_quality": {"score": self.citation_quality.score, "reason": self.citation_quality.reason},
            "relevance": {"score": self.relevance.score, "reason": self.relevance.reason},
            "total_score": self.total_score,
            "total_pct": self.total_percentage,
        }


# ──────────────────────────────────────────────
# LLM Judge 评分器
# ──────────────────────────────────────────────

COMPLETENESS_PROMPT = """你是一个严谨的评分专家。请对以下 AI 研报进行「主题覆盖度」评分（1-5分）。

用户问题：{query}

要求的覆盖主题（必须涉及这些关键词/概念）：
{expected_topics}

研报全文：
{report}

评分标准：
- 5分 = 覆盖了 ALL 要求主题，且每个主题都有实质讨论
- 4分 = 覆盖了大部分主题，少数主题略有提及但不够深入
- 3分 = 覆盖了约一半主题，部分主题缺失或一笔带过
- 2分 = 仅覆盖了小部分主题，多数主题缺失
- 1分 = 几乎没有覆盖要求的主题

请以 JSON 格式输出（不要 markdown 代码块）：
{{"score": <1-5的整数>, "matched": ["主题1", "主题2", ...], "missed": ["主题3", ...], "reason": "<一句话评分理由>"}}"""


CITATION_PROMPT = """你是一个严谨的评分专家。请对以下 AI 研报进行「引用质量」评分（1-5分）。

用户问题：{query}

研报全文：
{report}

评分标准：
- 5分 = 引用充分（≥{min_sources}个来源），来源多样，引用标记规范清晰
- 4分 = 引用较充分，但部分观点缺乏来源支撑
- 3分 = 有少量引用，但多数观点缺乏归属
- 2分 = 仅有 1-2 处引用，几乎无溯源
- 1分 = 完全没有引用或来源标记

请以 JSON 格式输出（不要 markdown 代码块）：
{{"score": <1-5的整数>, "matched": ["来源类型描述"], "missed": ["缺失项"], "reason": "<一句话评分理由>"}}"""


RELEVANCE_PROMPT = """你是一个严谨的评分专家。请对以下 AI 研报进行「回答相关性」评分（1-5分）。

用户问题：{query}

研报全文：
{report}

评分标准：
- 5分 = 完全针对问题回答，信息密度高，无跑题内容
- 4分 = 主要围绕问题，少量延伸但不偏离
- 3分 = 部分内容偏离问题，混入了不相关信息
- 2分 = 大量内容与问题无关
- 1分 = 回答与问题几乎无关

请以 JSON 格式输出（不要 markdown 代码块）：
{{"score": <1-5的整数>, "matched": ["优点"], "missed": ["缺点"], "reason": "<一句话评分理由>"}}"""


def _extract_json(text: str) -> dict:
    """从 LLM 回复中提取 JSON。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


class LLMScorer:
    """基于 LLM Judge 的三维评分器。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not base_url:
            base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
        self._llm = ChatOpenAI(
            model=model,
            temperature=0.0,  # 评分需要确定性
            api_key=api_key,
            base_url=base_url,
        )

    def _ask(self, prompt: str) -> dict:
        """向 LLM 发送评分请求，返回解析后的 JSON。"""
        try:
            result = self._llm.invoke([HumanMessage(content=prompt)])
            content = result.content if isinstance(result.content, str) else str(result.content)
            return _extract_json(content)
        except Exception as e:
            logger.warning("LLM 评分调用失败: %s", e)
            return {}

    def score_completeness(self, query: str, report: str, expected_topics: list[str]) -> ScoreDetail:
        """主题覆盖度评分。"""
        prompt = COMPLETENESS_PROMPT.format(
            query=query,
            expected_topics="、".join(expected_topics),
            report=report[:6000],  # 控制 token
        )
        data = self._ask(prompt)
        return ScoreDetail(
            score=data.get("score", 3),
            matched=data.get("matched", []),
            missed=data.get("missed", []),
            reason=data.get("reason", "LLM 评分未返回原因"),
        )

    def score_citation_quality(self, query: str, report: str, min_sources: int) -> ScoreDetail:
        """引用质量评分。"""
        prompt = CITATION_PROMPT.format(query=query, report=report[:6000], min_sources=min_sources)
        data = self._ask(prompt)
        return ScoreDetail(
            score=data.get("score", 3),
            matched=data.get("matched", []),
            missed=data.get("missed", []),
            reason=data.get("reason", "LLM 评分未返回原因"),
        )

    def score_relevance(self, query: str, report: str) -> ScoreDetail:
        """相关性评分。"""
        prompt = RELEVANCE_PROMPT.format(query=query, report=report[:6000])
        data = self._ask(prompt)
        return ScoreDetail(
            score=data.get("score", 3),
            matched=data.get("matched", []),
            missed=data.get("missed", []),
            reason=data.get("reason", "LLM 评分未返回原因"),
        )

    def score(
        self,
        query: str,
        report: str,
        expected_topics: list[str],
        min_sources: int = 2,
        case_id: str = "",
        category: str = "",
    ) -> EvalResult:
        """对一条报告进行三维度评分，返回加权总分。

        权重：完整度 40% + 引用质量 30% + 相关性 30%
        """
        completeness = self.score_completeness(query, report, expected_topics)
        citation_quality = self.score_citation_quality(query, report, min_sources)
        relevance = self.score_relevance(query, report)

        total = (
            completeness.score * 0.4
            + citation_quality.score * 0.3
            + relevance.score * 0.3
        )

        return EvalResult(
            case_id=case_id,
            query=query,
            category=category,
            completeness=completeness,
            citation_quality=citation_quality,
            relevance=relevance,
            total_score=round(total, 2),
            raw_report=report,
        )


# ──────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────

def generate_eval_report(results: list[EvalResult], output_path: Optional[str] = None) -> str:
    """将评分结果渲染为 Markdown 报告。"""
    if not results:
        report = "# Eval Report\n\n暂无评测结果。\n"
    else:
        report = _render_markdown(results)

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info("评测报告已写入: %s", output_path)

    return report


def _render_markdown(results: list[EvalResult]) -> str:
    """渲染 Markdown 格式报告。"""
    lines = []
    lines.append("# 多 Agent 研报系统 — 评测报告\n")
    lines.append(f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**测试用例数**: {len(results)}\n")

    # ── 聚合统计 ──
    lines.append("## 一、总体评分\n")
    lines.append("| 类别 | 用例数 | 完整度(均) | 引用(均) | 相关度(均) | 总分(均) |")
    lines.append("|------|--------|-----------|---------|-----------|---------|")

    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    grand_total = 0.0
    for cat, items in sorted(categories.items()):
        avg_comp = sum(i.completeness.score for i in items) / len(items)
        avg_cite = sum(i.citation_quality.score for i in items) / len(items)
        avg_rel = sum(i.relevance.score for i in items) / len(items)
        avg_total = sum(i.total_score for i in items) / len(items)
        cat_label = {"market_research": "市场调研", "tech_analysis": "技术分析", "knowledge_qa": "知识问答"}.get(cat, cat)
        lines.append(f"| {cat_label} | {len(items)} | {avg_comp:.1f} | {avg_cite:.1f} | {avg_rel:.1f} | {avg_total:.1f} |")
        grand_total += sum(i.total_score for i in items)

    overall_avg = grand_total / len(results) if results else 0
    lines.append(f"| **总计/平均** | {len(results)} | — | — | — | **{overall_avg:.2f}** |\n")

    # ── 分数分布 ──
    lines.append("## 二、分数分布\n")
    buckets = {"优秀 (4.0-5.0)": 0, "良好 (3.0-3.9)": 0, "一般 (2.0-2.9)": 0, "较差 (<2.0)": 0}
    for r in results:
        if r.total_score >= 4.0:
            buckets["优秀 (4.0-5.0)"] += 1
        elif r.total_score >= 3.0:
            buckets["良好 (3.0-3.9)"] += 1
        elif r.total_score >= 2.0:
            buckets["一般 (2.0-2.9)"] += 1
        else:
            buckets["较差 (<2.0)"] += 1
    for label, count in buckets.items():
        bar = "█" * count + "░" * (max(len(results), 5) - count) if results else ""
        lines.append(f"- {label}: {count} 条 {bar}")
    lines.append("")

    # ── 逐条详情 ──
    lines.append("## 三、逐条评分明细\n")
    lines.append("| ID | 类别 | 问题 | 完整度 | 引用 | 相关度 | 总分 | 说明 |")
    lines.append("|----|------|------|--------|------|--------|------|------|")

    for r in results:
        cat_label = {"market_research": "调研", "tech_analysis": "技术", "knowledge_qa": "问答"}.get(r.category, r.category)
        query_short = r.query[:30] + ("..." if len(r.query) > 30 else "")
        lines.append(
            f"| {r.case_id} | {cat_label} | {query_short} | "
            f"{r.completeness.score} | {r.citation_quality.score} | "
            f"{r.relevance.score} | {r.total_score} | {r.completeness.reason[:40]} |"
        )
    lines.append("")

    # ── 详细扣分分析 ──
    lines.append("## 四、扣分分析\n")
    for r in results:
        issues = []
        if r.completeness.missed:
            issues.append(f"主题缺失: {'、'.join(r.completeness.missed)}")
        if r.citation_quality.missed:
            issues.append(f"引用不足: {'、'.join(r.citation_quality.missed)}")
        if r.relevance.missed:
            issues.append(f"相关性问题: {'、'.join(r.relevance.missed)}")
        if r.total_score < 4.0:
            lines.append(f"### {r.case_id}: {r.query}\n")
            lines.append(f"- 总分: {r.total_score}/5.0")
            lines.append(f"- 完整度: {r.completeness.score}/5 — {r.completeness.reason}")
            lines.append(f"- 引用质量: {r.citation_quality.score}/5 — {r.citation_quality.reason}")
            lines.append(f"- 相关性: {r.relevance.score}/5 — {r.relevance.reason}")
            if issues:
                lines.append(f"- 主要扣分项:")
                for issue in issues:
                    lines.append(f"  - {issue}")
            lines.append("")

    # ── 改进建议 ──
    low_count = sum(1 for r in results if r.total_score < 3.0)
    if low_count > 0:
        lines.append("## 五、改进建议\n")
        lines.append(f"有 {low_count} 条用例总分低于 3.0，需针对性优化：\n")
        for r in results:
            if r.total_score < 3.0:
                lines.append(f"- **{r.case_id}** ({r.query[:20]}…): {r.completeness.reason} / {r.citation_quality.reason}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 独立运行测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # 快速验证评分器是否正常工作
    scorer = LLMScorer()
    sample_report = """# 测试报告
本报告分析了AI芯片市场。NVIDIA在训练芯片市场占据主导地位。
参考来源：[1] NVIDIA财报2025 [2] 某分析报告"""
    result = scorer.score(
        query="AI芯片市场分析",
        report=sample_report,
        expected_topics=["AI芯片", "NVIDIA", "GPU", "训练", "推理"],
        min_sources=2,
        case_id="TEST-00",
        category="market_research",
    )
    print(result.to_dict())
