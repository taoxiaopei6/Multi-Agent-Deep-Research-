"""Agent 输出 Schema 定义：用于校验 LLM 返回的 JSON 结构。

每个 Agent 的 prompt 中已经约定了 JSON 格式，这里用 Pydantic 做二次校验，
确保即使 LLM 输出结构有微小偏差，也能正确解析。
"""

from pydantic import BaseModel, Field
from typing import Optional


class IntentRouterOutput(BaseModel):
    """IntentRouter 输出"""
    route: str = "multiagent"
    reason: str = ""


class PlanOutput(BaseModel):
    """Planner 输出"""
    objective: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    outline: list[dict] = Field(default_factory=list)
    budget: dict = Field(default_factory=dict)
    research_questions: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "sub_questions": self.sub_questions,
            "outline": self.outline,
            "budget": self.budget,
            "research_questions": self.research_questions,
        }


class EvidenceItem(BaseModel):
    source_id: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""
    source_type: str = ""
    doc_id: str = ""
    reliability_hint: str = "unknown"
    supports_questions: list[str] = Field(default_factory=list)
    notes: str = ""


class SearchScoutOutput(BaseModel):
    """WebScout / LocalRAGScout 输出"""
    summary: str = ""
    evidence: list[dict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rejected_source_ids: list[str] = Field(default_factory=list)
    reject_reason: str = ""


class EvidencePoolItem(BaseModel):
    source_id: str = ""
    source_type: str = "web"
    title: str = ""
    url: str = ""
    doc_id: str = ""
    snippet: str = ""
    supports_questions: list[str] = Field(default_factory=list)
    reliability_score: float = 0.5
    reliability_reason: str = ""
    source_label: str = ""


class AuditFlag(BaseModel):
    type: str = ""
    target: str = ""
    reason: str = ""


class SourceIndexItem(BaseModel):
    source_id: str = ""
    label: str = ""
    locator: str = ""
    source_type: str = "source"


class EvidenceJudgeOutput(BaseModel):
    """EvidenceJudge 输出"""
    summary: str = ""
    evidence_pool: list[dict] = Field(default_factory=list)
    audit_flags: list[dict] = Field(default_factory=list)
    source_index: list[dict] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "evidence_pool": self.evidence_pool,
            "audit_flags": self.audit_flags,
            "source_index": self.source_index,
        }


class Finding(BaseModel):
    claim_id: str = ""
    claim: str = ""
    confidence: str = "medium"
    source_ids: list[str] = Field(default_factory=list)


class ClaimMapItem(BaseModel):
    claim_id: str = ""
    source_ids: list[str] = Field(default_factory=list)


class AnalystOutput(BaseModel):
    """Analyst 输出"""
    analysis_summary: str = ""
    needs_more_research: bool = False
    missing_gaps: list[str] = Field(default_factory=list)
    findings: list[dict] = Field(default_factory=list)
    claim_map: list[dict] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "analysis_summary": self.analysis_summary,
            "needs_more_research": self.needs_more_research,
            "missing_gaps": self.missing_gaps,
            "findings": self.findings,
            "claim_map": self.claim_map,
            "next_actions": self.next_actions,
        }


class SupplementaryQuery(BaseModel):
    section_id: str = ""
    query: str = ""
    source_preference: str = "hybrid"
    reason: str = ""


class ReflectOutput(BaseModel):
    """Reflect 输出"""
    reflection_summary: str = ""
    supplementary_queries: list[dict] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reflection_summary": self.reflection_summary,
            "supplementary_queries": self.supplementary_queries,
        }


# 节点名 → Schema 映射
SCHEMA_MAP = {
    "intent": IntentRouterOutput,
    "intent_router": IntentRouterOutput,
    "plan": PlanOutput,
    "web_search": SearchScoutOutput,
    "local_rag": SearchScoutOutput,
    "deep_dive": EvidenceJudgeOutput,
    "analyze": AnalystOutput,
    "reflect": ReflectOutput,
}
