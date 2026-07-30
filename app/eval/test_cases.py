"""测试用例定义：3 类别 × 8 条 = 24 条评测数据。"""

from dataclasses import dataclass, field


@dataclass
class TestCase:
    """一条测试用例。"""
    id: str                         # 唯一标识，如 "MR-01"
    category: str                   # market_research | tech_analysis | knowledge_qa
    query: str                      # 用户查询
    expected_topics: list[str]      # 应覆盖的主题关键词
    min_sources: int = 2            # 至少引用多少个来源
    difficulty: str = "medium"      # easy | medium | hard


# ──────────────────────────────────────────────
# 市场调研类（Market Research）
# ──────────────────────────────────────────────
MARKET_RESEARCH_CASES = [
    TestCase(
        id="MR-01",
        category="market_research",
        query="2025年全球AI芯片市场规模及竞争格局分析",
        expected_topics=["AI芯片", "市场规模", "GPU", "NVIDIA", "竞争格局"],
        min_sources=3,
        difficulty="medium",
    ),
    TestCase(
        id="MR-02",
        category="market_research",
        query="中国新能源汽车市场2025年主要品牌销量对比与趋势",
        expected_topics=["新能源汽车", "比亚迪", "销量", "市场份额", "电动化"],
        min_sources=3,
        difficulty="medium",
    ),
    TestCase(
        id="MR-03",
        category="market_research",
        query="全球云计算市场三大巨头AWS、Azure、GCP最新份额对比",
        expected_topics=["AWS", "Azure", "GCP", "云服务", "市场份额"],
        min_sources=3,
        difficulty="hard",
    ),
    TestCase(
        id="MR-04",
        category="market_research",
        query="2025年跨境电商行业发展趋势分析",
        expected_topics=["跨境电商", "出海", "SHEIN", "Temu", "TikTok Shop"],
        min_sources=2,
        difficulty="medium",
    ),
    TestCase(
        id="MR-05",
        category="market_research",
        query="中国半导体产业链国产化进展与挑战",
        expected_topics=["半导体", "芯片", "国产化", "光刻机", "制造工艺"],
        min_sources=3,
        difficulty="hard",
    ),
    TestCase(
        id="MR-06",
        category="market_research",
        query="全球SaaS行业2025年发展现状与趋势",
        expected_topics=["SaaS", "云计算", "企业服务", "订阅经济"],
        min_sources=2,
        difficulty="medium",
    ),
    TestCase(
        id="MR-07",
        category="market_research",
        query="中国宠物经济市场规模及增长驱动因素",
        expected_topics=["宠物经济", "宠物食品", "宠物医疗", "市场规模"],
        min_sources=2,
        difficulty="easy",
    ),
    TestCase(
        id="MR-08",
        category="market_research",
        query="东南亚数字经济2025年发展机遇分析",
        expected_topics=["东南亚", "数字经济", "印尼", "越南", "电商"],
        min_sources=2,
        difficulty="medium",
    ),
]


# ──────────────────────────────────────────────
# 技术分析类（Tech Analysis）
# ──────────────────────────────────────────────
TECH_ANALYSIS_CASES = [
    TestCase(
        id="TA-01",
        category="tech_analysis",
        query="LangChain vs LlamaIndex：RAG框架选型对比分析",
        expected_topics=["LangChain", "LlamaIndex", "RAG", "Agent", "对比"],
        min_sources=2,
        difficulty="easy",
    ),
    TestCase(
        id="TA-02",
        category="tech_analysis",
        query="向量数据库Milvus与Pinecone技术对比",
        expected_topics=["Milvus", "Pinecone", "向量数据库", "相似度搜索"],
        min_sources=2,
        difficulty="medium",
    ),
    TestCase(
        id="TA-03",
        category="tech_analysis",
        query="GraphRAG技术原理与实现方案分析",
        expected_topics=["GraphRAG", "知识图谱", "实体抽取", "社区检测"],
        min_sources=2,
        difficulty="hard",
    ),
    TestCase(
        id="TA-04",
        category="tech_analysis",
        query="Agentic RAG架构设计方案比较",
        expected_topics=["Agent", "RAG", "多Agent编排", "路由", "反思"],
        min_sources=2,
        difficulty="medium",
    ),
    TestCase(
        id="TA-05",
        category="tech_analysis",
        query="大模型推理优化技术：KV Cache与Speculative Decoding",
        expected_topics=["KV Cache", "推理优化", "显存", "批处理"],
        min_sources=2,
        difficulty="hard",
    ),
    TestCase(
        id="TA-06",
        category="tech_analysis",
        query="RAG系统中Embedding模型选型对比",
        expected_topics=["Embedding", "向量化", "bge", "召回率", "语义搜索"],
        min_sources=2,
        difficulty="medium",
    ),
    TestCase(
        id="TA-07",
        category="tech_analysis",
        query="多模态大模型技术路线对比分析",
        expected_topics=["多模态", "视觉", "CLIP", "GPT-4V", "跨模态"],
        min_sources=2,
        difficulty="medium",
    ),
    TestCase(
        id="TA-08",
        category="tech_analysis",
        query="MCP协议在AI Agent中的应用前景分析",
        expected_topics=["MCP", "协议", "工具调用", "标准化", "Agent"],
        min_sources=2,
        difficulty="medium",
    ),
]


# ──────────────────────────────────────────────
# 知识问答类（Knowledge QA）
# ──────────────────────────────────────────────
KNOWLEDGE_QA_CASES = [
    TestCase(
        id="KQ-01",
        category="knowledge_qa",
        query="什么是RAG检索增强生成？其核心组件有哪些？",
        expected_topics=["RAG", "检索", "生成", "向量数据库", "Embedding"],
        min_sources=1,
        difficulty="easy",
    ),
    TestCase(
        id="KQ-02",
        category="knowledge_qa",
        query="Transformer架构中的自注意力机制原理是什么？",
        expected_topics=["自注意力", "QKV", "多头注意力", "位置编码", "缩放点积"],
        min_sources=1,
        difficulty="medium",
    ),
    TestCase(
        id="KQ-03",
        category="knowledge_qa",
        query="什么是向量数据库？与传统关系型数据库有什么区别？",
        expected_topics=["向量数据库", "非结构化数据", "相似度搜索", "关系型数据库"],
        min_sources=1,
        difficulty="easy",
    ),
    TestCase(
        id="KQ-04",
        category="knowledge_qa",
        query="LangGraph中StateGraph的工作原理是什么？",
        expected_topics=["StateGraph", "状态管理", "节点", "边", "条件路由"],
        min_sources=1,
        difficulty="medium",
    ),
    TestCase(
        id="KQ-05",
        category="knowledge_qa",
        query="Embedding模型训练中的对比学习原理是什么？",
        expected_topics=["对比学习", "正样本", "负样本", "损失函数", "表示学习"],
        min_sources=1,
        difficulty="hard",
    ),
    TestCase(
        id="KQ-06",
        category="knowledge_qa",
        query="数据库事务的ACID特性是什么？各特性有什么意义？",
        expected_topics=["ACID", "原子性", "一致性", "隔离性", "持久性"],
        min_sources=1,
        difficulty="easy",
    ),
    TestCase(
        id="KQ-07",
        category="knowledge_qa",
        query="RAG系统中的文档分块策略有哪些？各有什么优缺点？",
        expected_topics=["分块", "语义分块", "重叠", "固定大小", "递归分块"],
        min_sources=1,
        difficulty="medium",
    ),
    TestCase(
        id="KQ-08",
        category="knowledge_qa",
        query="AI Agent中ReAct模式的工作原理是什么？",
        expected_topics=["ReAct", "推理", "行动", "观察", "思维链"],
        min_sources=1,
        difficulty="medium",
    ),
]


# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────

def all_test_cases() -> list[TestCase]:
    """返回全部测试用例。"""
    return MARKET_RESEARCH_CASES + TECH_ANALYSIS_CASES + KNOWLEDGE_QA_CASES


def get_cases_by_category(category: str) -> list[TestCase]:
    """按类别筛选测试用例。"""
    return [c for c in all_test_cases() if c.category == category]


def get_case_by_id(case_id: str) -> TestCase | None:
    """按 ID 查找测试用例。"""
    for c in all_test_cases():
        if c.id == case_id:
            return c
    return None
