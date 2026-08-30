<p align="center">
  <img src="docs/logo.png" alt="DeepResearch" width="200"/>
</p>

<h1 align="center">DeepResearch</h1>

<p align="center">
  <em>Evidence-grounded autonomous research framework</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://www.langchain.com/langgraph"><img src="https://img.shields.io/badge/built%20with-LangGraph-3399FF.svg" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/API-FastAPI-009688.svg" alt="FastAPI"></a>
  <img src="https://img.shields.io/badge/tests-71%20passed-brightgreen" alt="Tests">
  <a href="https://github.com/taoxiaopei6/Multi-Agent-Deep-Research-/blob/main/app/eval/security_cases.py"><img src="https://img.shields.io/badge/security-109%2F109%20passed-brightgreen" alt="Security"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#security">Security</a> •
  <a href="#faq">FAQ</a>
</p>

---

Not just generating answers — **building traceable research reports where every claim is backed by evidence**.

```
Research Question
      ↓
   Planning ──→ Sub-questions + Search Plan
      ↓
   Evidence Collection ──→ Web + Local KB
      ↓
   Evidence Judge ──→ Confidence Scoring + Audit
      ↓
   Claim-Evidence Mapping
      ↓
   Report Generation ──→ Markdown + Research Artifact
```

## Core Differentiators

|  | DeepResearch | Typical Agent |
|--|------------|---------------|
| **Evidence Pool** | Every source scored by authority, freshness, and cross-verification | Binary "found/not found" |
| **Claim-Evidence Map** | Every conclusion explicitly traces to supporting sources | Opaque generation |
| **Research Trace** | Full execution graph with latency, tokens, and decisions | Black box |
| **Research Artifact** | Structured output: claims + evidence + confidence + audit | Just text |
| **Evaluation Suite** | 300 benchmark cases + LLM-as-Judge + version comparison | No eval |
| **Security Benchmark** | 109 prompt injection tests across 12 attack categories | No defense |

---

## Quick Start

```bash
# 1. Install
git clone https://github.com/taoxiaopei6/Multi-Agent-Deep-Research-.git
cd Multi-Agent-Deep-Research-
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your DeepSeek + Bocha API keys

# 3. Interactive CLI
python main.py

# 4. Launch Web UI
cd app && python app_main.py
# → http://localhost:8000
```

All conclusions come with **traceable source links** and **confidence scores** that you can inspect in the Research Artifact panel.

---

## Benchmark

| Dimension | Score | Description |
|-----------|:-----:|-------------|
| **Completeness** | 5.0 / 5.0 | Full topic coverage |
| **Citation Quality** | 3.3 / 5.0 | Active improvement area |
| **Relevance** | 5.0 / 5.0 | On-topic, no hallucination |
| **Overall** | **4.5 / 5.0** | LLM-as-Judge (3-dimensional) |

### Security Benchmark

| Category | Cases | Status |
|----------|:-----:|:------:|
| Direct Injection | 10 / 10 | Passed |
| Role Injection | 10 / 10 | Passed |
| Evidence Poisoning | 10 / 10 | Passed |
| Data Exfiltration | 10 / 10 | Passed |
| Jailbreak | 10 / 10 | Passed |
| Unicode & Encoding | 9 / 9 | Passed |
| Context Manipulation | 10 / 10 | Passed |
| Format Confusion | 10 / 10 | Passed |
| Refusal Suppression | 10 / 10 | Passed |
| Authority Manipulation | 10 / 10 | Passed |
| Multi-turn | 5 / 5 | Passed |
| Payload Delivery | 5 / 5 | Passed |
| **Total** | **109 / 109** | **Passed** |

> Evaluation methodology: LLM-as-Judge across 3 dimensions (1-5 scale). Scorer prompts and test cases are publicly defined in `app/eval/`. Security cases in `app/eval/security_cases.py`. Incremental tracking in `output/eval_tracking/`.

---

## Architecture

```
User Input → IntentRouter
                  │
            ┌─────┴─────┐
            │           │
        direct      multiagent
                  │
                  ▼
             Planner
                  │
          ┌───────┴───────┐
          ▼               ▼
      WebSearch       LocalRAG
          │               │
          └───────┬───────┘
                  ▼
           EvidenceJudge
           (scoring + audit + dedup)
                  │
                  ▼
             Analyst
        (gap analysis + claim mapping)
                  │
          ┌───────┴───────┐
          │               │
     needs_more      sufficient
          │               │
      Reflect           Writer
          │               │
          └──→ WebSearch  ▼
                    (max N rounds)  Report + Artifact
```

### Agent Roles

| Agent | Role |
|-------|------|
| **IntentRouter** | Classify query: direct answer or full research |
| **Planner** | Decompose problem, generate outline + search plan |
| **WebScout** | Web retrieval with relevance filtering |
| **LocalRAGScout** | Local knowledge base retrieval |
| **EvidenceJudge** | Score credibility, deduplicate URLs, flag audits |
| **Analyst** | Synthesize findings, build claim-evidence map, detect gaps |
| **Reflect** | Generate supplementary queries for information gaps |
| **Writer** | Compose final report + structured Research Artifact |

---

## Research Artifact

Every research run produces a structured **Research Artifact** alongside the report:

```json
{
  "artifact_version": "1.0",
  "claims": [
    {
      "claim_id": "c_1",
      "claim": "Global AI chip market reaches $120B in 2025",
      "confidence": "high",
      "evidence_count": 3,
      "supporting_evidence": [
        {
          "source_id": "WEB1_1-1",
          "title": "IIM Industry Report",
          "reliability_score": 0.88,
          "reliability_breakdown": {
            "authority": 0.95,
            "freshness": 0.90,
            "semantic_match": 0.93
          }
        }
      ]
    }
  ],
  "evidence_pool_summary": {
    "total": 19,
    "high_confidence": 12,
    "low_confidence": 2
  }
}
```

This is displayed in the Web UI alongside the report, so users can inspect exactly which evidence supports each conclusion.

---

## Security

The system processes external web content, making **Prompt Injection** a primary attack surface. Defense is multi-layered:

1. **Content Boundary Isolation** — External content is wrapped with `--- EXTERNAL CONTENT ---` markers, structurally separated from system instructions
2. **Injection Pattern Detection** — Regex patterns covering 12 attack categories
3. **Content Sanitization** — Known malicious patterns are replaced before reaching the LLM

Security benchmark: **109/109 cases passed** across Direct Injection, Role Injection, Evidence Poisoning, Data Exfiltration, Jailbreak, Unicode & Encoding, Context Manipulation, Format Confusion, Refusal Suppression, Authority Manipulation, Multi-turn, and Payload Delivery.

See `security/content_boundary.py` and `tests/test_security_benchmark.py`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | DeepSeek / GPT / Qwen (OpenAI-compatible) |
| **Workflow** | LangGraph StateGraph + Checkpointer |
| **Vector Search** | Milvus + BGE-M3 (metadata-filtered ANN) |
| **Memory** | PostgreSQL (FTS lexical) + Milvus (vector) hybrid recall; Redis (optional) |
| **Web Search** | Bocha API |
| **Backend** | FastAPI + SSE streaming |
| **Frontend** | Vue 3 + TypeScript + Vite |
| **Eval** | LLM-as-Judge + incremental tracker |

---

## Project Structure

```
deep_research/
├── app/
│   ├── mult_agents/           # Core multi-agent system
│   │   ├── nodes/             # Agent node implementations
│   │   ├── security/          # Prompt injection defense
│   │   ├── memory/            # Short/long-term memory
│   │   └── rag/               # RAG retrieval core
│   ├── backend/               # FastAPI + SSE streaming
│   ├── eval/                  # Benchmark suite (300 cases)
│   └── app_main.py            # Web entry point
├── front/agent_front/         # Vue 3 frontend
├── tests/                     # 71 tests across 6 modules
├── main.py                    # CLI entry point
├── docker-compose.yml         # PostgreSQL + Redis + Milvus
├── config.example.json        # Configuration template
└── .env.example               # Environment template
```

---

## Roadmap

- [x] Research Artifact (claim-evidence-confidence)
- [x] Structured Trace (latency + tokens)
- [x] Security Benchmark (109/109)
- [x] Incremental Evaluation Tracker
- [ ] Multi-search source support (Google / SerpAPI)
- [ ] Evidence re-ranking with cross-encoder
- [ ] URL-level deduplication optimization
- [ ] PDF/image ingestion for local KB
- [ ] CLI rich output formatting

---

## FAQ

**Q: Do I need Docker?**  
A: No. Python + API keys only. Docker is for optional memory persistence.

**Q: What does it cost to run?**  
A: ~220k tokens per research task on DeepSeek + free-tier Bocha API.

**Q: How is this different from GPT Researcher / AutoGen / LangGraph demo?**  
A: Most agent projects generate text. This project generates **traceable research artifacts** where every claim maps to scored evidence.

**Q: What LLMs are supported?**  
A: Any OpenAI-compatible API — DeepSeek, GPT, Qwen, Claude, local models.

---

## License

[MIT](LICENSE) © 2026 taoxiaopei6
