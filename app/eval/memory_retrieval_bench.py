"""
Memory retrieval regression benchmark (hybrid recall baseline).

Usage:
    F:/agent/deep_research_2025/.venv/Scripts/python.exe -m app.eval.memory_retrieval_bench

Seeds a fixed set of memories into an isolated tenant (bench_retrieval), runs
the case set against PG lexical + Milvus vector hybrid recall, and reports per
case: lexical recall, vector recall, final TopK, recall_sources, latency.

This is the baseline for comparing ILIKE (pre-Phase-2) vs FTS (post-Phase-2)
lexical recall. Run the same file after Phase 2 to compare — recall must not
regress.
"""
import os
import sys
import time
import json
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.mult_agents.memory.manager import MemoryManager
from app.mult_agents.memory.base import MemoryType

PG_DSN = "postgresql://root:root123@127.0.0.1:5432/deep_research"
TENANT = "bench_retrieval"
USER = "bench_user"
OTHER_TENANT = "bench_retrieval_other_tenant"
OTHER_USER = "bench_other_user"

# ---------------------------------------------------------------------------
# Seed data: fixed memories that exercise each recall scenario.
# Each entry: (mem_id_slug, text, memory_type, namespace, thread_id, owner)
# owner: "user" (TENANT/USER) | "other_user" (TENANT/OTHER_USER) | "other_tenant"
# ---------------------------------------------------------------------------
SEED = [
    # semantic facts (this user/tenant)
    ("fact_basketball", "用户名字叫小明，喜欢打篮球", MemoryType.SEMANTIC, "facts/sport", "t1", "user"),
    ("fact_icecoffee", "用户喜欢喝冰美式咖啡", MemoryType.SEMANTIC, "facts/drink", "t1", "user"),
    ("fact_techstack", "公司的技术栈是 Python 和 FastAPI", MemoryType.SEMANTIC, "facts/tech", "t1", "user"),
    ("fact_pg_db", "团队使用 PostgreSQL 作为主数据库", MemoryType.SEMANTIC, "facts/tech", "t1", "user"),
    ("fact_geo", "用户居住在上海，工作地点在张江", MemoryType.SEMANTIC, "facts/personal", "t1", "user"),
    # identifier/exact scenarios
    ("fact_gse", "基因表达数据集 GSE183904 包含 24 个样本", MemoryType.SEMANTIC, "facts/bio", "t1", "user"),
    ("fact_cuda", "CUDA 12.8 支持 Blackwell 架构", MemoryType.SEMANTIC, "facts/tech", "t1", "user"),
    # other-user isolation (same tenant, different user — must NOT be recalled)
    ("other_user", "别人喜欢潜水", MemoryType.SEMANTIC, "facts/sport", "t1", "other_user"),
    # other-tenant isolation (must NOT be recalled)
    ("other_tenant", "另一个租户的机密信息 XYZ789", MemoryType.SEMANTIC, "facts/private", "t1", "other_tenant"),
    # episodic (this user/tenant)
    ("task_export", "昨天导出了用户数据报表", MemoryType.EPISODIC, "tasks/conversation", "t1", "user"),
    ("task_search", "上周帮用户搜索了深度学习论文", MemoryType.EPISODIC, "tasks/research", "t1", "user"),
]

# ---------------------------------------------------------------------------
# Cases: (query, expected_mem_id_slug, scenario, memory_type)
# ---------------------------------------------------------------------------
CASES = [
    # keyword direct hit
    ("小明 打篮球", "fact_basketball", "keyword_direct", MemoryType.SEMANTIC),
    ("冰美式 咖啡", "fact_icecoffee", "keyword_direct", MemoryType.SEMANTIC),
    # paraphrase: PG lexical 0-hit expected, Milvus vector should hit
    ("小明平时喜欢什么体育运动", "fact_basketball", "paraphrase_lex0_vec1", MemoryType.SEMANTIC),
    ("用户爱喝什么冷饮", "fact_icecoffee", "paraphrase_lex0_vec1", MemoryType.SEMANTIC),
    # identifier / exact
    ("GSE183904", "fact_gse", "identifier_exact", MemoryType.SEMANTIC),
    ("CUDA 12.8", "fact_cuda", "identifier_exact", MemoryType.SEMANTIC),
    # keyword partial / lexical overlap
    ("技术栈", "fact_techstack", "keyword_partial", MemoryType.SEMANTIC),
    ("PostgreSQL", "fact_pg_db", "keyword_direct", MemoryType.SEMANTIC),
    ("数据库", "fact_pg_db", "paraphrase_lex0_vec1", MemoryType.SEMANTIC),
    # personal fact
    ("张江", "fact_geo", "keyword_direct", MemoryType.SEMANTIC),
    # isolation: these should NOT be recalled
    ("潜水", "other_user", "isolation_user", MemoryType.SEMANTIC),
    ("机密", "other_tenant", "isolation_tenant", MemoryType.SEMANTIC),
    # episodic
    ("导出报表", "task_export", "keyword_direct", MemoryType.EPISODIC),
    ("深度学习论文", "task_search", "keyword_direct", MemoryType.EPISODIC),
    # unrelated query: expect empty or low-rank
    ("量子物理前沿", None, "unrelated", MemoryType.SEMANTIC),
]


def seed(manager):
    """Seed fixed memories into bench tenant(s). Returns {slug: memory_id}."""
    mapping = {}
    for slug, text, mtype, ns, thread, owner in SEED:
        if owner == "other_user":
            uid, tid = OTHER_USER, TENANT
        elif owner == "other_tenant":
            uid, tid = OTHER_USER, OTHER_TENANT
        else:
            uid, tid = USER, TENANT
        if mtype == MemoryType.SEMANTIC:
            mid = manager.save_fact(
                uid, text, category=ns.split("/")[-1], tenant_id=tid, thread_id=thread,
            )
        else:
            mid = manager.save_task(
                uid, "conversation", {"text": text}, outcome=text[:200],
                tenant_id=tid, thread_id=thread,
            )
        mapping[slug] = mid
    return mapping


def run_case(manager, query, expected_slug, mtype, mapping):
    expected_id = mapping.get(expected_slug) if expected_slug else None

    # 独立判断 lexical 是否命中该 id（直接跑 PG ILIKE，不依赖事后 source 标注）
    lex_recalled = False
    if expected_id:
        if mtype == MemoryType.SEMANTIC:
            lex = manager._search_postgres(
                TENANT, USER, query, MemoryType.SEMANTIC.value, None, None, limit=10
            )
        else:
            lex = manager._search_postgres(
                TENANT, USER, query, MemoryType.EPISODIC.value, None, None, limit=10
            )
        lex_recalled = any(e.id == expected_id for e in lex)

    t0 = time.time()
    if mtype == MemoryType.SEMANTIC:
        results = manager.search_semantic(USER, query, tenant_id=TENANT, limit=5)
    else:
        results = manager.search_similar_tasks(USER, query, tenant_id=TENANT, limit=5)
    latency_ms = (time.time() - t0) * 1000

    hit = None
    if expected_id:
        for entry in results:
            if entry.id == expected_id:
                hit = entry
                break

    sources = [e.metadata.get("recall_sources") for e in results if e.metadata.get("recall_sources")]
    vec_hit = hit is not None and "vector" in (hit.metadata.get("recall_sources") or [])

    return {
        "query": query,
        "scenario": mtype.value,
        "expected": expected_slug,
        "recalled": hit is not None,
        "topk": [e.id[:8] for e in results[:3]],
        "recall_sources": sources,
        "lexical_recalled": lex_recalled,
        "vector_recalled": vec_hit,
        "latency_ms": round(latency_ms, 1),
    }


def main():
    manager = MemoryManager(
        short_term_backend="memory", long_term_backend="postgres",
        postgres_dsn=PG_DSN, tenant_id=TENANT, enable_milvus=True,
        milvus_host="127.0.0.1", milvus_port=19530, milvus_collection="mult_agent_memory",
        embedding_model_path="F:/models/bge-m3",
    )
    # 清掉旧 bench 数据（PG + Milvus）
    for uid, tid in [(USER, TENANT), (OTHER_USER, TENANT), (OTHER_USER, OTHER_TENANT)]:
        manager.clear_user_memory(uid, tenant_id=tid)
    mapping = seed(manager)
    print(f"seeded {len(mapping)} memories")

    rows = []
    for query, expected, scenario, mtype in CASES:
        r = run_case(manager, query, expected, mtype, mapping)
        r["scenario"] = scenario
        rows.append(r)

    # 输出
    print("\n=== memory retrieval benchmark ===")
    print(f"{'scenario':<22} {'recalled':<9} {'lex':<5} {'vec':<5} {'latency_ms':<10} query")
    for r in rows:
        print(f"{r['scenario']:<22} {str(r['recalled']):<9} {str(r['lexical_recalled']):<5} "
              f"{str(r['vector_recalled']):<5} {r['latency_ms']:<10} {r['query']}")

    total = len(rows)
    recalled = sum(1 for r in rows if r["recalled"])
    print(f"\nrecall: {recalled}/{total}")
    # 输出 JSON 便于 Phase 2 后对比
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output", "memory_bench_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"baseline saved: {out}")


if __name__ == "__main__":
    main()
