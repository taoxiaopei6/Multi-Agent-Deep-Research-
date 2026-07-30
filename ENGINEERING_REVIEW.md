# Engineering Review: Multi-Agent Deep Research

> 项目时间：2025
> 作者：陶晓佩

---

## 目录

1. [架构设计](#1-架构设计)
2. [实验结果](#2-实验结果)
3. [Ablation 分析](#3-ablation-分析)
4. [踩坑记录](#4-踩坑记录)
5. [Future Work](#5-future-work)

---

## 1. 架构设计

### 1.1 为什么是 8 Agent？

系统将一次"深度调研"拆解为 6 个阶段，每个阶段由一个专用 Agent 负责：

| 阶段 | Agent | 职责 |
|------|-------|------|
| 路由 | IntentRouter | 判断走快速回答还是调研链路 |
| 规划 | Planner | 拆解问题为子问题 + 生成搜索计划 |
| 检索 | WebScout + LocalRAGScout | 并行搜索网络和本地知识库 |
| 裁判 | EvidenceJudge | 证据评分 + 可信度审计 |
| 分析 | Analyst | 评估证据完备性，决策是否补搜 |
| 写作 | Writer | 整合证据输出深度研报 |

**设计原则：**

- 每个 Agent 职责单一，System Prompt 不超过 300 字
- 通过 LangGraph StateGraph 编排，状态字段 45+，支持条件路由
- Agent 之间不直接通信，通过共享 State 协作

### 1.2 Planner 设计演化

**第一版（Heuristic Query）：**

从用户问题猜测"核心实体"，然后拼接模板：{实体}是什么 / {实体} GitHub / {实体} 官方文档...

问题：5/6 的搜索词被浪费在"全球是什么"、"全球 GitHub"这类垃圾词上。

**第二版（Pure LLM）：**

删除 heuristic，搜索计划完全来自 LLM 大逻辑的 search_queries。同时保留 `_default_plan()` 作为 LLM 失败时的兜底。

效果：有效搜索从 1/6 提升到 6/6。

---

## 2. 实验结果

### 2.1 全量 Benchmark (24 cases)

**配置：** max_iterations=2, DeepSeek, Memory OFF, max_iterations=2

| 指标 | 值 |
|------|-----|
| 成功率 | 24/24 (100%) |
| 平均分 | 4.38 / 5.0 |
| 平均 Evidence | 22.4 条 |
| 平均 Bocha 调用 | 6.9 次 |
| 平均耗时 | 223.6 秒 |
| 平均报告长度 | 7,837 字 |

**按类别：**

| 类别 | 得分 | 完整度 | 引用 | 相关度 | Evidence | Bocha | 耗时 |
|------|:----:|:-----:|:---:|:-----:|:-------:|:----:|:---:|
| 市场调研 | 4.46 | 4.9 | 3.5 | 4.9 | 24 | 7.5 | 234s |
| 技术分析 | 4.40 | 5.0 | 3.0 | 5.0 | 28 | 8.0 | 284s |
| 知识问答 | 4.28 | 4.9 | 2.9 | 4.9 | 15 | 5.2 | 152s |

### 2.2 Planner 优化 A/B (MR-01)

| 指标 | Heuristic | LLM Planner |
|------|:---------:|:-----------:|
| 总分 | 3.80 | **4.40** |
| 有效搜索 | 1/6 | **6/6** |
| Evidence 数 | 7 | **24** |

### 2.3 配置文件

Benchmark 固定配置记录在 `benchmark_config.json`：

```json
{
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "max_iterations": 2,
  "enable_memory": false,
  "planner": "pure LLM (heuristic removed)",
  "commit": "b02ca3c"
}
```

---

## 3. Ablation 分析

### 3.1 实验设计

- **目标：** 评估 Reflect 迭代补搜模块对调研质量和搜索成本的影响
- **方法：** 相同 4 条 research task，仅改变 max_iterations（0 = OFF, 2 = ON）
- **固定变量：** 模型、Prompt、thread_id 策略、Memory OFF

### 3.2 结果

| Metric | Reflect OFF | Reflect ON | Delta |
|--------|:-----------:|:----------:|:-----:|
| Score | **4.55** | 4.38 | -0.18 |
| Completeness | **5.0** | 4.8 | -0.2 |
| Citation | **3.5** | 3.2 | -0.2 |
| Evidence | 22.5 | **26.3** | +3.8 |
| Bocha | 6.0 | 7.8 | +1.8 |
| LLM | 6.0 | 13.0 | +7.0 |
| Time | 129s | 249s | +120s |

### 3.3 归因分析（关键）

Reflect ON 为何 Evidence 增加但分数下降？

深入分析 MR-01 后发现：

1. **Writer 确实引用了所有新增证据**（21/21 source_id 出现在报告中）
2. **但新增证据 29% 是重复 URL**——同一个 iim.net.cn 文章在首轮和补搜轮次被多次搜到，只是 source_id 前缀不同（WEB1、WEB2、WEB3）
3. 新增证据的质量评分也不高（reliability_score 集中在 0.58-0.72）

**结论：** 瓶颈不在 Writer，而在证据去重层。Reflect 不是没有价值，而是当前搜索源（Bocha）对同一话题返回结果高度重叠，Reflect 难以找到真正的新信息。

### 3.4 工程决策

| Feature | Score Delta | LLM Delta | Time Delta | Decision |
|---------|:----------:|:---------:|:----------:|----------|
| Planner 优化 | +0.60 | - | - | **保留（正式上线）** |
| Reflect | -0.18 | +7 | +120s | **默认关闭（Feature Flag）** |

---

## 4. 踩坑记录

### 4.1 Heuristic Query 浪费搜索预算

**问题：** Planner 的 `_derive_direct_search_queries()` 从用户问题中猜一个词作为"核心实体"，然后套模板搜索"实体是什么"、"实体 GitHub"、"实体 GitHub 官方文档"。对于"2025年全球AI芯片市场规模及竞争格局分析"这样的长查询，它把"全球"当成了核心实体，生成"全球是什么"、"全球 GitHub"等 5 个无效搜索词，导致 5/6 的 Bocha 调用被浪费。

**修复：** 删除 heuristic 全部代码，搜索计划仅来自 LLM 大纲中的 search_queries。

**数据验证：** 有效搜索从 1/6 提升到 6/6，总分从 3.80 提升到 4.40。

**启示：** 2025 年的大模型生成搜索词的能力已经比人工模板强，heuristic 不应参与主流程。

### 4.2 Benchmark thread_id 状态累积

**问题：** Benchmark Runner 复用了默认 thread_id（"default"），导致 LangGraph Checkpointer 跨用例累积 messages。统计 `llm_calls` 时用 `len(messages)/2`，结果越跑越大（MR-01=14, MR-02=28, MR-03=42...），实际每条 pipeline 只调用了 ~6-14 次 LLM。

**修复：** 每条用例使用独立 thread_id（`benchmark_MR-01_时间戳`），llm_calls 改为统计 AI Message 数量。

**启示：** Benchmark 的 case isolation 是基础设施层的问题，不是统计公式的问题。

### 4.3 IntentRouter LLM 抖动

**问题：** 相同输入（KQ-01 "什么是RAG"）在两次运行中分别走了 direct answer 和 multi-agent 两条路线，导致 ablation 数据污染。

**处理：** 从 ablation 统计中剔除该用例，在分析中单独说明。

**启示：** 基于 LLM 的路由器存在固有随机性。高频场景可以考虑规则预检（已有 `detect_intent` 函数但未作为主判断）。

### 4.4 Ablation 配置错误

**问题：** 最初使用 max_iterations=1 作为 Reflect OFF，但代码中 `iteration >= max_iter` 的判断意味着 iteration=0 时 max_iterations=1 不会触发 write，Reflect 仍会执行一次。

**修复：** max_iterations=0 才是真正的 Reflect OFF。

---

## 5. Future Work

### P1: URL 级别去重

当前问题：同一 URL 的证据在 evidence_pool 中出现多次（WEB1_1-1、WEB2_1-2、WEB3_1-3...），在 MR-01 中观察到 29% 的 evidence entries 对应跨检索轮次的重复 URL，属于 retrieval artifact。

改进方向：evidence_pool 中基于 URL 去重，保留最高分的一条。预期可提高信噪比，但具体对 Score 的提升幅度需进一步实验验证。

### P2: Evidence 排序优化

当前 EvidenceJudge 主要依靠规则评分（域名判断）。可以引入 cross-encoder 对 evidence 按 query 相关性重排序，把最有价值的证据排在前面，帮助 Writer 更好地利用。

### P3: 搜索源扩展

Reflect 效果受限于 Bocha 的搜索结果重叠度。增加更多搜索源（如 SerpAPI、Google Custom Search）可以让 Reflect 真正找到不同的信息。

---

## 总结

这个项目从"搭建 8 Agent Demo"开始，经历了代码重构、Bug 修复、Benchmark、Ablation 的完整迭代。最终产出的不仅是一个能跑的系统，更是一组可以用来做工程决策的数据：

- 24 条 Benchmark 100% 通过，平均分 4.38/5.0
- Planner 优化带来 +0.60 分的提升（有效搜索 1/6 → 6/6）
- Reflect Ablation：Evidence +17%，但 Score -0.18，Latency +92%

**核心工程决策（Trade-off）：**

| Feature | Quality | Latency | LLM Cost | Decision |
|---------|:-------:|:-------:|:--------:|----------|
| Planner 优化 | ↑ +0.60 | — | — | **默认开启** |
| Reflect | ↔ -0.18 | ↑ +120s (1.9x) | ↑ +7 LLM (2.1x) | **默认关闭** |

**归因与不确定性：**

当前数据表明 Reflect 的边际收益为负，且根源推测为搜索引擎结果重叠（MR-01 中 29% 的 evidence 为重复 URL）导致的信息增益趋零。但这一推断仍需进一步实验来解耦以下两个因素：
- Search Diversity：搜索结果本身的信息丰富度
- Downstream Utilization：Writer 对新增证据的利用能力

> *Further controlled experiments are required to isolate the impact of search diversity from downstream reasoning limitations.*

**三个真实踩坑：**
- Heuristic Query 导致 5/6 的搜索预算浪费（实验数据驱动修复）
- Benchmark 共享 thread_id 导致 llm_calls 跨用例累积（架构级修复）
- IntentRouter LLM 抖动导致 KQ-01 路由不一致（实验数据剔除处理）

对于 AI Agent 系统的开发，最重要的收获是：**system performance is dominated by retrieval quality and signal diversity, not agent complexity.**
