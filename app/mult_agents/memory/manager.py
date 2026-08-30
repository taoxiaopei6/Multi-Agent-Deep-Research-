"""
记忆管理器

统一管理短期记忆和长期记忆，提供统一的接口
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import jieba
from rank_bm25 import BM25Okapi

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .base import MemoryEntry, MemoryType
from .long_term import EpisodicMemoryStore, SemanticMemoryStore, content_to_text, tokenize_for_bm25
from .short_term import ShortTermMemory
from .utils import extract_memory_from_messages, format_memories_for_prompt, merge_user_profile

try:
    import redis
except Exception:
    redis = None

try:
    import psycopg
except Exception:
    psycopg = None

# 使用 langchain-milvus 新包（类名是 Milvus，不是 MilvusVectorStore）
try:
    from langchain_milvus import Milvus as MilvusVectorStore
except ImportError:
    # 降级到旧包
    from langchain_community.vectorstores import Milvus as MilvusVectorStore

logger = logging.getLogger("mult_agents.memory")


class MemoryManager:
    def __init__(
        self,
        short_term_ttl: int = 604800,  # 7天（以秒为单位）
        short_term_max_messages: int = 30,
        short_term_summary_threshold: int = 20,
        db_path: Optional[str] = None,
        # 多租户数据隔离：不同租户的记忆相互独立，防止越权。
        tenant_id: str = "default_tenant",
        short_term_backend: str = "postgres",
        long_term_backend: str = "postgres",
        # 长期记忆隔离粒度："user" 跨会话共享，"thread" 严格会话隔离。
        long_term_scope: str = "user",
        save_conversation_task: bool = False,
        enable_milvus: bool = True,
        redis_url: Optional[str] = None,
        postgres_dsn: Optional[str] = None,
        milvus_host: Optional[str] = None,
        milvus_port: int = 19530,
        milvus_collection: str = "mult_agent_memory",
        embedding_api_key: Optional[str] = None,
        embedding_model_path: str = "",
        base_url: str = "",
        summary_model: str = "deepseek-chat",
    ):
        self.default_tenant_id = tenant_id
        self.short_term_backend = short_term_backend.lower()
        self.long_term_backend = long_term_backend.lower()
        self.long_term_scope = long_term_scope.lower()
        if self.long_term_scope not in {"user", "thread"}:
            self.long_term_scope = "user"
        self.save_conversation_task = save_conversation_task
        self.enable_long_term = self.long_term_backend != "disabled"
        self.enable_milvus = enable_milvus
        self.short_term_ttl = short_term_ttl
        self.short_term_max_messages = short_term_max_messages
        self.short_term_summary_threshold = short_term_summary_threshold
        self.short_term = ShortTermMemory(ttl_seconds=short_term_ttl)
        # 语义/情景 SQLite store 懒加载（见 semantic/episodic property），
        # 避免与 PostgreSQL 主存储同时初始化、重复加载 embedding 模型。
        self._sqlite_db_path = db_path
        self._semantic_store: Optional[SemanticMemoryStore] = None
        self._episodic_store: Optional[EpisodicMemoryStore] = None
        self._rerank_embeddings = None
        self._redis_client = None
        self._postgres_dsn = postgres_dsn
        self._milvus_store = None
        self._summary_llm = None
        self._last_trace: Dict[str, Any] = {}
        self._last_milvus_raw_hits: List[Dict[str, Any]] = []
        if self.short_term_backend == "redis":
            self._init_redis(redis_url)
        # PostgreSQL 初始化：short-term 或 long-term 任一方使用 PG 时都需要建表。
        # _init_postgres 内部按 short_term_backend / long_term_backend 分别决定建哪些表。
        if self.short_term_backend == "postgres" or self.enable_long_term and self.long_term_backend == "postgres":
            self._init_postgres()
        # 按配置初始化后端：只初始化实际使用的长期记忆，避免多个后端同时加载
        if self.enable_long_term:
            # 共享嵌入模型：全系统只加载一次，供统一重排(_rerank)、Milvus、SQLite 复用
            if embedding_model_path:
                self._load_rerank_embeddings(embedding_model_path)
            if self.enable_milvus:
                self._init_milvus(milvus_host, milvus_port, milvus_collection, embedding_model_path)
        self._init_summary_llm(embedding_api_key, base_url, summary_model)
        logger.info(
            "记忆管理器初始化完成 | short_term=%s | long_term=%s | scope=%s | save_task=%s | redis=%s | postgres=%s | milvus=%s",
            self.short_term_backend,
            self.long_term_backend,
            self.long_term_scope,
            self.save_conversation_task,
            bool(self._redis_client),
            bool(psycopg and self._postgres_dsn),
            bool(self._milvus_store),
        )

    def _load_rerank_embeddings(self, embedding_model_path: str) -> None:
        """加载共享嵌入模型，供统一重排(_rerank)、Milvus、SQLite 复用。失败时置空，检索降级纯词法。"""
        try:
            self._rerank_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
            logger.info("重排嵌入模型已加载: %s", embedding_model_path)
        except Exception as exc:
            logger.warning("重排嵌入模型加载失败，检索降级纯词法: %s", exc)
            self._rerank_embeddings = None

    @property
    def semantic(self) -> SemanticMemoryStore:
        """语义记忆 SQLite store（懒加载）：仅在实际用到时才构造，避免与 PG 主存储同时初始化。"""
        if self._semantic_store is None:
            store = SemanticMemoryStore(db_path=self._sqlite_db_path, embedding_model_path="")
            if self._rerank_embeddings is not None:
                store._embeddings = self._rerank_embeddings  # 复用共享模型，不重复加载
            self._semantic_store = store
        return self._semantic_store

    @property
    def episodic(self) -> EpisodicMemoryStore:
        """情景记忆 SQLite store（懒加载）：同上。"""
        if self._episodic_store is None:
            store = EpisodicMemoryStore(db_path=self._sqlite_db_path, embedding_model_path="")
            if self._rerank_embeddings is not None:
                store._embeddings = self._rerank_embeddings  # 复用共享模型，不重复加载
            self._episodic_store = store
        return self._episodic_store

    def _init_redis(self, redis_url: Optional[str]) -> None:
        if not redis_url or redis is None:
            return
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis_client = client
        except Exception as exc:
            try:
                fallback_url = redis_url.replace("redis://root:", "redis://:")
                client = redis.Redis.from_url(fallback_url, decode_responses=True)
                client.ping()
                self._redis_client = client
            except Exception:
                logger.warning("Redis 初始化失败，降级内存短期记忆: %s", exc)

    def _init_postgres(self) -> None:
        if not self._postgres_dsn or psycopg is None:
            return
        try:
            with psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    if self.enable_long_term and self.long_term_backend == "postgres":
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS memory_entries (
                                id TEXT PRIMARY KEY,
                                tenant_id TEXT NOT NULL,
                                user_id TEXT NOT NULL,
                                thread_id TEXT,
                                memory_type TEXT NOT NULL,
                                namespace TEXT,
                                content JSONB NOT NULL,
                                summary TEXT,
                                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            )
                            """
                        )
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_memory_entries_lookup
                            ON memory_entries (tenant_id, user_id, memory_type, created_at DESC)
                            """
                        )
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS user_profiles (
                                tenant_id TEXT NOT NULL,
                                user_id TEXT NOT NULL,
                                profile JSONB NOT NULL,
                                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                PRIMARY KEY (tenant_id, user_id)
                            )
                            """
                        )
                    if self.short_term_backend == "postgres":
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS short_term_messages (
                                id TEXT PRIMARY KEY,
                                tenant_id TEXT NOT NULL,
                                user_id TEXT NOT NULL,
                                thread_id TEXT NOT NULL,
                                role TEXT NOT NULL,
                                content TEXT NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            )
                            """
                        )
                        cur.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_short_term_lookup
                            ON short_term_messages (tenant_id, user_id, thread_id, created_at DESC)
                            """
                        )
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS short_term_summaries (
                                tenant_id TEXT NOT NULL,
                                user_id TEXT NOT NULL,
                                thread_id TEXT NOT NULL,
                                summary TEXT NOT NULL,
                                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                PRIMARY KEY (tenant_id, user_id, thread_id)
                            )
                            """
                        )
                    conn.commit()
        except Exception as exc:
            logger.warning("PostgreSQL 初始化失败；配置为 PostgreSQL 的相关记忆功能将不可用: %s", exc)
            self._postgres_dsn = None

    def _init_milvus(
        self,
        milvus_host: Optional[str],
        milvus_port: int,
        milvus_collection: str,
        embedding_model_path: str,
    ) -> None:
        if not milvus_host or not embedding_model_path:
            return
        try:
            # 复用共享嵌入模型（统一重排已加载），避免重复加载
            embeddings = self._rerank_embeddings or HuggingFaceEmbeddings(model_name=embedding_model_path)
            self._milvus_store = MilvusVectorStore(
                embedding_function=embeddings,
                collection_name=milvus_collection,
                connection_args={"uri": f"http://{milvus_host}:{milvus_port}"},
                auto_id=True,
            )
        except Exception as exc:
            logger.warning("Milvus 初始化失败，降级 PostgreSQL 检索: %s", exc)
            self._milvus_store = None

    def _init_summary_llm(self, api_key: Optional[str], base_url: str, summary_model: str) -> None:
        if not api_key:
            return
        try:
            self._summary_llm = ChatOpenAI(model=summary_model, temperature=0.1, api_key=api_key, base_url=base_url)
        except Exception as exc:
            logger.warning("摘要模型初始化失败，降级规则压缩: %s", exc)
            self._summary_llm = None

    def _redis_thread_key(self, tenant_id: str, user_id: str, thread_id: str) -> str:
        return f"ma:short:{tenant_id}:{user_id}:{thread_id}"

    def _redis_summary_key(self, tenant_id: str, user_id: str, thread_id: str) -> str:
        return f"ma:short:summary:{tenant_id}:{user_id}:{thread_id}"

    def _serialize_message(self, message: BaseMessage) -> Dict[str, str]:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "human"
        return {"role": role, "content": str(message.content)}

    def _deserialize_message(self, payload: Dict[str, str]) -> BaseMessage:
        role = payload.get("role", "human")
        content = payload.get("content", "")
        if role == "ai":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    def _summarize_text(self, existing_summary: str, history_slice: List[Dict[str, str]]) -> str:
        lines = [f"{item.get('role', 'human')}: {item.get('content', '')}" for item in history_slice]
        history_text = "\n".join(lines)
        if self._summary_llm is None:
            combined = f"{existing_summary}\n{history_text}".strip()
            return combined[-4000:]
        prompt = (
            "你是对话压缩引擎。请在保留事实、偏好、结论、待办和约束的前提下进行递归摘要。\n"
            f"已有摘要：{existing_summary or '无'}\n"
            "新增历史：\n"
            f"{history_text}\n"
            "输出要求：100-300字，中文，结构紧凑。"
        )
        response = self._summary_llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()

    def _compress_redis_thread(self, tenant_id: str, user_id: str, thread_id: str) -> None:
        if self._redis_client is None:
            return
        key = self._redis_thread_key(tenant_id, user_id, thread_id)
        summary_key = self._redis_summary_key(tenant_id, user_id, thread_id)
        raw_messages = self._redis_client.lrange(key, 0, -1) or []
        if len(raw_messages) <= self.short_term_max_messages:
            return
        parsed = [json.loads(item) for item in raw_messages]
        split_at = len(parsed) - self.short_term_summary_threshold
        to_summarize = parsed[:split_at]
        keep_messages = parsed[split_at:]
        existing_summary = self._redis_client.get(summary_key) or ""
        new_summary = self._summarize_text(existing_summary, to_summarize)
        pipe = self._redis_client.pipeline()
        pipe.delete(key)
        if keep_messages:
            pipe.rpush(key, *[json.dumps(item) for item in keep_messages])
        pipe.set(summary_key, new_summary, ex=self.short_term_ttl)
        pipe.expire(key, self.short_term_ttl)
        pipe.execute()

    def _save_pg_short_term_message(self, tenant_id: str, user_id: str, thread_id: str, payload: Dict[str, str]) -> None:
        if not self._postgres_dsn or psycopg is None:
            return
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO short_term_messages
                    (id, tenant_id, user_id, thread_id, role, content, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        str(uuid4()),
                        tenant_id,
                        user_id,
                        thread_id,
                        payload.get("role", "human"),
                        payload.get("content", ""),
                    ),
                )
                conn.commit()

    def _get_pg_short_term_messages(self, tenant_id: str, user_id: str, thread_id: str) -> List[Dict[str, str]]:
        if not self._postgres_dsn or psycopg is None:
            return []
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM short_term_messages
                    WHERE tenant_id = %s AND user_id = %s AND thread_id = %s
                    ORDER BY created_at ASC
                    """,
                    (tenant_id, user_id, thread_id),
                )
                rows = cur.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def _set_pg_short_term_summary(self, tenant_id: str, user_id: str, thread_id: str, summary: str) -> None:
        if not self._postgres_dsn or psycopg is None:
            return
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO short_term_summaries (tenant_id, user_id, thread_id, summary, updated_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (tenant_id, user_id, thread_id)
                    DO UPDATE SET summary = EXCLUDED.summary, updated_at = NOW()
                    """,
                    (tenant_id, user_id, thread_id, summary),
                )
                conn.commit()

    def _get_pg_short_term_summary(self, tenant_id: str, user_id: str, thread_id: str) -> str:
        if not self._postgres_dsn or psycopg is None:
            return ""
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT summary
                    FROM short_term_summaries
                    WHERE tenant_id = %s AND user_id = %s AND thread_id = %s
                    """,
                    (tenant_id, user_id, thread_id),
                )
                row = cur.fetchone()
        return row[0] if row else ""

    def _compress_pg_thread(self, tenant_id: str, user_id: str, thread_id: str) -> None:
        if not self._postgres_dsn or psycopg is None:
            return
        history = self._get_pg_short_term_messages(tenant_id, user_id, thread_id)
        if len(history) <= self.short_term_max_messages:
            return
        split_at = len(history) - self.short_term_summary_threshold
        to_summarize = history[:split_at]
        keep_messages = history[split_at:]
        existing_summary = self._get_pg_short_term_summary(tenant_id, user_id, thread_id)
        new_summary = self._summarize_text(existing_summary, to_summarize)
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM short_term_messages
                    WHERE tenant_id = %s AND user_id = %s AND thread_id = %s
                    """,
                    (tenant_id, user_id, thread_id),
                )
                for item in keep_messages:
                    cur.execute(
                        """
                        INSERT INTO short_term_messages
                        (id, tenant_id, user_id, thread_id, role, content, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        """,
                        (
                            str(uuid4()),
                            tenant_id,
                            user_id,
                            thread_id,
                            item.get("role", "human"),
                            item.get("content", ""),
                        ),
                    )
                conn.commit()
        self._set_pg_short_term_summary(tenant_id, user_id, thread_id, new_summary)

    def _upsert_profile_pg(self, tenant_id: str, user_id: str, profile: Dict[str, Any]) -> None:
        if not self._postgres_dsn or psycopg is None:
            return
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_profiles (tenant_id, user_id, profile, updated_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (tenant_id, user_id)
                    DO UPDATE SET profile = EXCLUDED.profile, updated_at = NOW()
                    """,
                    (tenant_id, user_id, json.dumps(profile)),
                )
                conn.commit()

    def _insert_memory_pg(self, entry: MemoryEntry, summary: str = "") -> None:
        if not self._postgres_dsn or psycopg is None:
            return
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_entries
                    (id, tenant_id, user_id, thread_id, memory_type, namespace, content, summary, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        summary = EXCLUDED.summary,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        entry.id,
                        entry.metadata.get("tenant_id", self.default_tenant_id),
                        entry.user_id or "default_user",
                        entry.thread_id,
                        entry.memory_type.value,
                        entry.namespace,
                        json.dumps(entry.content)
                        if isinstance(entry.content, dict)
                        else json.dumps({"text": str(entry.content)}),
                        summary,
                        json.dumps(entry.metadata),
                        entry.created_at,
                    ),
                )
                conn.commit()

    def _index_memory_milvus(self, text: str, metadata: Dict[str, Any]) -> None:
        if not self._milvus_store or not text.strip():
            return
        try:
            safe_metadata = dict(metadata or {})
            safe_metadata.setdefault("source", "memory")
            safe_metadata.setdefault("doc_id", str(safe_metadata.get("memory_id", "")))
            safe_metadata.setdefault("title", str(safe_metadata.get("namespace", "memory")))
            doc = Document(page_content=text, metadata=safe_metadata)
            self._milvus_store.add_documents([doc])
            logger.info(
                "[memory] milvus write ok | tenant=%s user=%s thread=%s type=%s namespace=%s source=%s text_chars=%d",
                safe_metadata.get("tenant_id"),
                safe_metadata.get("user_id"),
                safe_metadata.get("thread_id", ""),
                safe_metadata.get("memory_type", ""),
                safe_metadata.get("namespace", ""),
                safe_metadata.get("source", ""),
                len(text),
            )
        except Exception as exc:
            logger.warning("Milvus 写入失败: %s", exc)

    def _annotate_entries_with_source(self, entries: List[MemoryEntry], source: str) -> List[MemoryEntry]:
        for entry in entries:
            entry.metadata["retrieval_source"] = source
        return entries

    # ------------------------------------------------------------------
    # 统一召回 + 重排（recall + rerank）
    # ------------------------------------------------------------------

    def _rerank_embed(self, text: str) -> List[float]:
        """为统一重排生成 query / 候选向量。无模型或失败时返回空列表。"""
        if self._rerank_embeddings is None:
            return []
        try:
            return self._rerank_embeddings.embed_query(text)
        except Exception as exc:
            logger.warning("重排向量生成失败，降级纯词法打分: %s", exc)
            return []

    def _rerank_cosine(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度（与 SQLite 长期记忆打分保持一致）"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _recall_k(self, limit: int) -> int:
        """召回量放大：统一重排需要比最终 limit 更多的候选才有意义。上限 30 约束实时 embedding 成本。"""
        return min(max(limit * 3, 20), 30)

    def _rerank(
        self,
        query: str,
        candidates: List[MemoryEntry],
        limit: int,
    ) -> List[MemoryEntry]:
        """对候选记忆统一打分排序，与 SQLite 长期记忆的公式一致。

        有 query embedding：0.70*余弦 + 0.25*BM25 + 0.05*exact
        无 embedding（降级）：0.95*BM25 + 0.05*exact
        """
        if not candidates:
            return []

        query_embedding = self._rerank_embed(query)
        query_lower = query.strip().lower()
        query_tokens = tokenize_for_bm25(query)

        # BM25 词法分（先转 float 列表：rank_bm25 返回 numpy 数组）
        corpus_tokens = [
            tokenize_for_bm25(content_to_text(e.content))
            for e in candidates
        ]
        bm25_scores = [0.0] * len(candidates)
        if query_tokens and any(corpus_tokens):
            bm25 = BM25Okapi(corpus_tokens)
            raw = [float(s) for s in bm25.get_scores(query_tokens)]
            max_raw = max(raw) if raw else 0.0
            if max_raw > 0:
                bm25_scores = [max(0.0, s) / max_raw for s in raw]

        scored: List[tuple] = []
        for idx, entry in enumerate(candidates):
            # 向量分：优先用已存 embedding，否则实时生成（保证 PG/Milvus 候选与 SQLite 同口径）
            vec_score = 0.0
            entry_embedding = getattr(entry, "embedding", None)
            if query_embedding:
                if not entry_embedding:
                    entry_embedding = self._rerank_embed(content_to_text(entry.content))
                if entry_embedding:
                    vec_score = self._rerank_cosine(entry_embedding, query_embedding)
                    vec_score = max(0.0, min(1.0, vec_score))

            bm25_score = bm25_scores[idx]

            content_str = content_to_text(entry.content).lower()
            exact_score = 1.0 if query_lower and query_lower in content_str else 0.0

            if query_embedding:
                total = vec_score * 0.70 + bm25_score * 0.25 + exact_score * 0.05
            else:
                total = bm25_score * 0.95 + exact_score * 0.05

            scored.append((entry, total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    def _fetch_postgres_by_ids(
        self,
        tenant_id: str,
        user_id: str,
        ids: List[str],
    ) -> Dict[str, MemoryEntry]:
        """按 memory_id 批量从 PostgreSQL 取回正式记录，避免 N+1 查询。"""
        if not ids or not self._postgres_dsn or psycopg is None:
            return {}
        result: Dict[str, MemoryEntry] = {}
        try:
            with psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, memory_type, namespace, content, metadata, created_at
                        FROM memory_entries
                        WHERE tenant_id = %s AND user_id = %s AND id = ANY(%s)
                        """,
                        (tenant_id, user_id, ids),
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            logger.warning("PostgreSQL 批量取回记忆失败: %s", exc)
            return {}
        for row in rows:
            result[row[0]] = MemoryEntry(
                id=row[0],
                content=row[3],
                memory_type=MemoryType(row[1]),
                user_id=user_id,
                namespace=row[2],
                metadata=row[4] or {},
                created_at=row[5],
            )
        return result

    def _merge_and_hydrate(
        self,
        primary: str,
        lex: List[MemoryEntry],
        vec: List[MemoryEntry],
        tenant_id: str,
        user_id: str,
        memory_type: MemoryType,
    ) -> List[MemoryEntry]:
        """合并 lexical + vector 两路候选，去重并回主存储 hydrate。

        - 去重按 memory_id；主存储条目（lex）优先
        - vec 只贡献 memory_id：同 id 已存在则补 provenance，否则从主存储批量 hydrate
        - 返回的每条 MemoryEntry.content 均为主存储正式记录
        - provenance 记入 metadata["recall_sources"]
        - memory_type 决定 SQLite hydrate 用 semantic 还是 episodic store（显式传参，避免全局可变状态）
        """
        merged: List[MemoryEntry] = []
        seen: set = set()
        id_to_entry: Dict[str, MemoryEntry] = {}

        for entry in lex:
            if entry.id in seen:
                continue
            seen.add(entry.id)
            entry.metadata["recall_sources"] = ["lexical"]
            id_to_entry[entry.id] = entry
            merged.append(entry)

        # 仅由向量召回、主存储未出现的 id，批量 hydrate
        vec_ids = [e.id for e in vec if e.id not in seen]
        hyd = {}
        if vec_ids:
            if primary == "postgres":
                hyd = self._fetch_postgres_by_ids(tenant_id, user_id, vec_ids)
            else:
                store = self.semantic if memory_type == MemoryType.SEMANTIC else self.episodic
                for vid in vec_ids:
                    item = store.get(vid)
                    if item is not None:
                        hyd[vid] = item

        for e in vec:
            if e.id in seen:
                continue
            hydrated = hyd.get(e.id)
            if hydrated is None:
                continue  # 陈旧索引：主存储查不到，丢弃
            hydrated.metadata["recall_sources"] = ["vector"]
            seen.add(e.id)
            merged.append(hydrated)

        # 已被 lexical 召回的同一 id，补充 vector provenance
        for e in vec:
            if e.id in seen and e.id in id_to_entry:
                existing = id_to_entry[e.id]
                sources = existing.metadata.get("recall_sources") or ["lexical"]
                if "vector" not in sources:
                    sources.append("vector")
                existing.metadata["recall_sources"] = sources

        return merged

    def add_short_term_message(
        self,
        thread_id: str,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "default_user",
        tenant_id: Optional[str] = None,
    ) -> None:
        tenant = tenant_id or self.default_tenant_id
        metadata = metadata or {}
        metadata.update({"tenant_id": tenant, "user_id": user_id})
        payload = self._serialize_message(message)
        if self.short_term_backend == "redis" and self._redis_client is not None:
            key = self._redis_thread_key(tenant, user_id, thread_id)
            self._redis_client.rpush(key, json.dumps(payload))
            self._redis_client.expire(key, self.short_term_ttl)
            self._compress_redis_thread(tenant, user_id, thread_id)
            return
        if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
            self._save_pg_short_term_message(tenant, user_id, thread_id, payload)
            self._compress_pg_thread(tenant, user_id, thread_id)
            return
        if self._redis_client is None:
            self.short_term.add_message(thread_id, message, metadata)
            return
        self.short_term.add_message(thread_id, message, metadata)

    def add_short_term_messages(
        self,
        thread_id: str,
        messages: List[BaseMessage],
        user_id: str = "default_user",
        tenant_id: Optional[str] = None,
    ) -> None:
        for message in messages:
            self.add_short_term_message(
                thread_id=thread_id,
                message=message,
                user_id=user_id,
                tenant_id=tenant_id,
            )

    def get_short_term_summary(
        self,
        thread_id: str,
        user_id: str = "default_user",
        tenant_id: Optional[str] = None,
    ) -> str:
        tenant = tenant_id or self.default_tenant_id
        if self.short_term_backend == "redis" and self._redis_client is not None:
            key = self._redis_summary_key(tenant, user_id, thread_id)
            return self._redis_client.get(key) or ""
        if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
            return self._get_pg_short_term_summary(tenant, user_id, thread_id)
        if self._redis_client is None:
            messages = self.short_term.get_messages(thread_id, include_summary=True, last_n=0)
            if messages and isinstance(messages[0], SystemMessage):
                return str(messages[0].content)
            return ""
        return ""

    def get_short_term_messages(
        self,
        thread_id: str,
        include_summary: bool = True,
        last_n: Optional[int] = None,
        user_id: str = "default_user",
        tenant_id: Optional[str] = None,
    ) -> List[BaseMessage]:
        tenant = tenant_id or self.default_tenant_id
        if self.short_term_backend == "redis" and self._redis_client is not None:
            key = self._redis_thread_key(tenant, user_id, thread_id)
            raw = self._redis_client.lrange(key, 0, -1) or []
            if last_n:
                raw = raw[-last_n:]
            messages = [self._deserialize_message(json.loads(item)) for item in raw]
            if include_summary:
                summary = self.get_short_term_summary(thread_id, user_id=user_id, tenant_id=tenant)
                if summary:
                    return [SystemMessage(content=f"历史对话摘要：{summary}"), *messages]
            return messages
        if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
            raw = self._get_pg_short_term_messages(tenant, user_id, thread_id)
            if last_n:
                raw = raw[-last_n:]
            messages = [self._deserialize_message(item) for item in raw]
            if include_summary:
                summary = self.get_short_term_summary(thread_id, user_id=user_id, tenant_id=tenant)
                if summary:
                    return [SystemMessage(content=f"历史对话摘要：{summary}"), *messages]
            return messages
        if self._redis_client is None:
            return self.short_term.get_messages(thread_id, include_summary, last_n)
        return self.short_term.get_messages(thread_id, include_summary, last_n)

    def should_inject_long_term(
        self,
        user_id: str,
        thread_id: str,
        tenant_id: Optional[str] = None,
    ) -> bool:
        tenant = tenant_id or self.default_tenant_id
        if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
            return len(self._get_pg_short_term_messages(tenant, user_id, thread_id)) == 0
        if self.short_term_backend == "redis" and self._redis_client is not None:
            key = self._redis_thread_key(tenant, user_id, thread_id)
            return int(self._redis_client.llen(key) or 0) == 0
        return len(self.short_term.get_messages(thread_id, include_summary=False)) == 0

    def mark_injection_skipped(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        query: str,
        reason: str,
    ) -> None:
        self._last_trace = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "query": query,
            "skipped": True,
            "skip_reason": reason,
            "profile_injected": False,
            "summary_chars": 0,
            "memory_count": 0,
            "source_count": {},
            "injected_chars": 0,
            "items": [],
            "milvus_raw_hits": [],
        }

    def update_short_term_metadata(
        self,
        thread_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        self.short_term.update_thread_metadata(thread_id, metadata)

    def get_short_term_metadata(self, thread_id: str) -> Dict[str, Any]:
        return self.short_term.get_thread_metadata(thread_id)

    def clear_short_term(self, thread_id: str) -> bool:
        if self.short_term_backend == "redis" and self._redis_client is not None:
            keys = self._redis_client.keys(f"ma:short:*:*:{thread_id}")
            if keys:
                self._redis_client.delete(*keys)
            summary_keys = self._redis_client.keys(f"ma:short:summary:*:*:{thread_id}")
            if summary_keys:
                self._redis_client.delete(*summary_keys)
        if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
            with psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM short_term_messages WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM short_term_summaries WHERE thread_id = %s", (thread_id,))
                    conn.commit()
        return self.short_term.clear_thread(thread_id)

    def list_active_threads(self) -> List[str]:
        if self.short_term_backend == "redis" and self._redis_client is not None:
            keys = self._redis_client.keys("ma:short:*:*:*")
            threads = {item.rsplit(":", 1)[-1] for item in keys if "summary" not in item}
            if threads:
                return sorted(threads)
        if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
            with psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT thread_id FROM short_term_messages")
                    rows = cur.fetchall()
            threads = [row[0] for row in rows if row and row[0]]
            if threads:
                return sorted(set(threads))
        return self.short_term.list_active_threads()

    def save_user_profile(
        self,
        user_id: str,
        profile: Dict[str, Any],
        merge: bool = True,
        tenant_id: Optional[str] = None,
    ) -> str:
        if not self.enable_long_term:
            return str(uuid4())
        tenant = tenant_id or self.default_tenant_id
        existing = self.get_user_profile(user_id, tenant_id=tenant)
        merged_profile = merge_user_profile(existing, profile) if merge and existing else profile
        if self.long_term_backend == "postgres" and self._postgres_dsn and psycopg:
            memory_id = str(uuid4())
            self._upsert_profile_pg(tenant, user_id, merged_profile)
            self._index_memory_milvus(
                text=json.dumps(merged_profile),
                metadata={
                    "tenant_id": tenant,
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "memory_type": MemoryType.SEMANTIC.value,
                    "namespace": "user_profile",
                    "created_at": datetime.now().isoformat(),
                },
            )
            return memory_id
        memory_id = self.semantic.save_profile(user_id, profile, merge)
        if self.enable_milvus:
            self._index_memory_milvus(
                text=json.dumps(merged_profile),
                metadata={
                    "tenant_id": tenant,
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "memory_type": MemoryType.SEMANTIC.value,
                    "namespace": "user_profile",
                    "created_at": datetime.now().isoformat(),
                },
            )
        return memory_id

    def get_user_profile(self, user_id: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.enable_long_term:
            return None
        tenant = tenant_id or self.default_tenant_id
        if self.long_term_backend == "postgres" and self._postgres_dsn and psycopg:
            with psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT profile FROM user_profiles WHERE tenant_id = %s AND user_id = %s",
                        (tenant, user_id),
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0]
        return self.semantic.get_profile(user_id)

    def save_fact(
        self,
        user_id: str,
        fact: str,
        category: Optional[str] = None,
        tenant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        if not self.enable_long_term:
            return str(uuid4())
        tenant = tenant_id or self.default_tenant_id
        memory_id = str(uuid4()) if self.long_term_backend == "postgres" else self.semantic.save_fact(user_id, fact, category)
        entry = MemoryEntry(
            id=memory_id,
            content={"text": fact, "category": category or "general"},
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            thread_id=thread_id,
            namespace=f"facts/{category or 'general'}",
            metadata={"tenant_id": tenant, "category": category or "general"},
        )
        if self.long_term_backend == "postgres" and self._postgres_dsn and psycopg:
            self._insert_memory_pg(entry, summary=fact[:500])
        if self.enable_milvus:
            self._index_memory_milvus(
                text=fact,
                metadata={
                    "tenant_id": tenant,
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "memory_type": MemoryType.SEMANTIC.value,
                    "namespace": f"facts/{category or 'general'}",
                    "thread_id": thread_id,
                    "created_at": datetime.now().isoformat(),
                },
            )
        return memory_id

    def _search_milvus(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        memory_type: Optional[str] = None,
        namespace: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        if not self._milvus_store:
            return []
        try:
            # 保留 oversampling：当前仍是在 Python 侧按 tenant/user/type/thread 过滤，
            # 需要多取一些以保证过滤后仍有足够候选。设置安全上限。
            # （等以后 metadata filter 下推 Milvus 后再收紧）
            docs = self._milvus_store.similarity_search(query, k=min(max(limit * 4, 20), 120))
            logger.info(
                "[memory] milvus search raw | tenant=%s user=%s thread=%s query=%s raw_hits=%d",
                tenant_id,
                user_id,
                thread_id or "",
                query[:120],
                len(docs),
            )
        except Exception as exc:
            logger.warning("Milvus 检索失败，降级 PostgreSQL: %s", exc)
            return []
        entries: List[MemoryEntry] = []
        current_raw_hits: List[Dict[str, Any]] = []
        for doc in docs:
            metadata = doc.metadata or {}
            snippet = doc.page_content if len(doc.page_content) <= 160 else doc.page_content[:160] + "..."
            rejected_by = ""
            if metadata.get("tenant_id") != tenant_id or metadata.get("user_id") != user_id:
                rejected_by = "tenant_or_user_mismatch"
            elif memory_type and metadata.get("memory_type") != memory_type:
                rejected_by = "memory_type_mismatch"
            elif namespace and metadata.get("namespace") != namespace:
                rejected_by = "namespace_mismatch"
            elif thread_id and metadata.get("thread_id") != thread_id:
                rejected_by = "thread_mismatch"
            current_raw_hits.append(
                {
                    "query": query,
                    "memory_type_filter": memory_type,
                    "namespace_filter": namespace,
                    "thread_filter": thread_id,
                    "accepted": rejected_by == "",
                    "rejected_by": rejected_by or None,
                    "metadata": {
                        "tenant_id": metadata.get("tenant_id"),
                        "user_id": metadata.get("user_id"),
                        "thread_id": metadata.get("thread_id"),
                        "memory_type": metadata.get("memory_type"),
                        "namespace": metadata.get("namespace"),
                        "memory_id": metadata.get("memory_id"),
                    },
                    "snippet": snippet,
                }
            )
            if rejected_by:
                continue
            created_at = metadata.get("created_at")
            created_dt = datetime.fromisoformat(created_at) if created_at else datetime.now()
            entries.append(
                MemoryEntry(
                    id=str(metadata.get("memory_id", uuid4())),
                    content=doc.page_content,
                    memory_type=MemoryType(metadata.get("memory_type", MemoryType.SEMANTIC.value)),
                    user_id=user_id,
                    namespace=metadata.get("namespace"),
                    metadata=metadata,
                    created_at=created_dt,
                )
            )
            if len(entries) >= limit:
                break
        self._last_milvus_raw_hits.extend(current_raw_hits)
        accepted_hits = [item for item in current_raw_hits if item.get("accepted")]
        rejected_hits = [item for item in current_raw_hits if not item.get("accepted")]
        accepted_preview = [
            {
                "memory_id": item["metadata"].get("memory_id"),
                "type": item["metadata"].get("memory_type"),
                "namespace": item["metadata"].get("namespace"),
                "thread_id": item["metadata"].get("thread_id"),
                "snippet": item.get("snippet"),
            }
            for item in accepted_hits[: min(3, len(accepted_hits))]
        ]
        rejected_reason_count: Dict[str, int] = {}
        for item in rejected_hits:
            reason = item.get("rejected_by") or "unknown"
            rejected_reason_count[reason] = rejected_reason_count.get(reason, 0) + 1
        logger.info(
            "[memory] milvus search filtered | accepted=%d rejected=%d accepted_preview=%s rejected_reason_count=%s",
            len(accepted_hits),
            len(rejected_hits),
            json.dumps(accepted_preview),
            json.dumps(rejected_reason_count),
        )
        return entries

    def _search_postgres(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        memory_type: str,
        namespace: Optional[str],
        thread_id: Optional[str],
        limit: int,
    ) -> List[MemoryEntry]:
        if not self._postgres_dsn or psycopg is None:
            return []
        sql = """
            SELECT id, memory_type, namespace, content, metadata, created_at
            FROM memory_entries
            WHERE tenant_id = %s
              AND user_id = %s
              AND memory_type = %s
        """
        params: List[Any] = [tenant_id, user_id, memory_type]
        if query:
            pattern = f"%{query}%"
            sql += " AND (summary ILIKE %s OR content::text ILIKE %s)"
            params.extend([pattern, pattern])
        if namespace:
            sql += " AND namespace = %s"
            params.append(namespace)
        if thread_id:
            sql += " AND thread_id = %s"
            params.append(thread_id)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with psycopg.connect(self._postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        entries: List[MemoryEntry] = []
        for row in rows:
            entries.append(
                MemoryEntry(
                    id=row[0],
                    content=row[3],
                    memory_type=MemoryType(row[1]),
                    user_id=user_id,
                    namespace=row[2],
                    metadata=row[4] or {},
                    created_at=row[5],
                )
            )
        return entries

    def search_semantic(
        self,
        user_id: str,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
        tenant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if not self.enable_long_term:
            logger.info(
                "[memory] semantic search skipped | long_term=disabled | tenant=%s user=%s",
                tenant_id or self.default_tenant_id,
                user_id,
            )
            return []
        tenant = tenant_id or self.default_tenant_id
        scoped_thread_id = thread_id if self.long_term_scope == "thread" else None
        recall_k = self._recall_k(limit)

        # 主存储分支：PG（authoritative）或 SQLite，二选一。
        # backend=postgres 时 primary 恒为 postgres；PG 不可用（无 dsn/psycopg/失败）
        # 由 _search_postgres 内部返回空，绝不进入 SQLite 分支。
        lex: List[MemoryEntry] = []
        if self.long_term_backend == "postgres":
            primary = "postgres"
            lex = self._search_postgres(
                tenant_id=tenant,
                user_id=user_id,
                query=query,
                memory_type=MemoryType.SEMANTIC.value,
                namespace=namespace,
                thread_id=scoped_thread_id,
                limit=recall_k,
            )
        else:
            primary = "sqlite"
            lex = self.semantic.search(query, user_id=user_id, namespace=namespace, limit=recall_k)

        if primary == "postgres" and not lex:
            # PG lexical 0-hit 不代表 vector 没命中，继续 Milvus 召回。
            logger.info(
                "[memory] semantic search | primary=postgres lexical no hit | tenant=%s user=%s query=%s",
                tenant,
                user_id,
                query[:120],
            )

        # Milvus 向量召回（索引，只贡献 memory_id，最终内容回主存储 hydrate）
        vec: List[MemoryEntry] = []
        if self.enable_milvus and self._milvus_store:
            vec = self._search_milvus(
                tenant_id=tenant,
                user_id=user_id,
                query=query,
                memory_type=MemoryType.SEMANTIC.value,
                namespace=namespace,
                thread_id=scoped_thread_id,
                limit=recall_k,
            )

        merged = self._merge_and_hydrate(primary, lex, vec, tenant, user_id, MemoryType.SEMANTIC)
        if not merged:
            return []

        ranked = self._rerank(query, merged, limit)
        self._annotate_entries_with_source(ranked, primary)
        hit_preview = [str(item.content)[:120] for item in ranked[:3]]
        logger.info(
            "[memory] semantic search | primary=%s candidates=%d ranked=%d query=%s hit_preview=%s",
            primary,
            len(merged),
            len(ranked),
            query[:120],
            json.dumps(hit_preview),
        )
        return ranked

    def save_task(
        self,
        user_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        outcome: Optional[str] = None,
        tenant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        if not self.enable_long_term:
            return str(uuid4())
        tenant = tenant_id or self.default_tenant_id
        memory_id = (
            str(uuid4())
            if self.long_term_backend == "postgres"
            else self.episodic.save_task_record(user_id, task_type, task_data, outcome)
        )
        content = {"task_type": task_type, "task_data": task_data, "outcome": outcome or ""}
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=MemoryType.EPISODIC,
            user_id=user_id,
            thread_id=thread_id,
            namespace=f"tasks/{task_type}",
            metadata={"tenant_id": tenant, "task_type": task_type},
        )
        summary = outcome or json.dumps(task_data)[:500]
        if self.long_term_backend == "postgres" and self._postgres_dsn and psycopg:
            self._insert_memory_pg(entry, summary=summary)
        if self.enable_milvus:
            self._index_memory_milvus(
                text=f"{task_type}\n{summary}",
                metadata={
                    "tenant_id": tenant,
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "memory_type": MemoryType.EPISODIC.value,
                    "namespace": f"tasks/{task_type}",
                    "thread_id": thread_id,
                    "created_at": datetime.now().isoformat(),
                },
            )
        return memory_id

    def get_task_history(
        self,
        user_id: str,
        task_type: Optional[str] = None,
        limit: int = 10,
        tenant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if not self.enable_long_term:
            return []
        tenant = tenant_id or self.default_tenant_id
        if self.long_term_backend == "postgres" and self._postgres_dsn and psycopg:
            namespace = f"tasks/{task_type}" if task_type else None
            entries = self._search_postgres(
                tenant_id=tenant,
                user_id=user_id,
                query="",
                memory_type=MemoryType.EPISODIC.value,
                namespace=namespace,
                thread_id=thread_id if self.long_term_scope == "thread" else None,
                limit=limit,
            )
            if entries:
                return entries
        return self.episodic.get_task_history(user_id, task_type, limit)

    def search_similar_tasks(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        tenant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if not self.enable_long_term:
            logger.info(
                "[memory] episodic search skipped | long_term=disabled | tenant=%s user=%s",
                tenant_id or self.default_tenant_id,
                user_id,
            )
            return []
        tenant = tenant_id or self.default_tenant_id
        scoped_thread_id = thread_id if self.long_term_scope == "thread" else None
        recall_k = self._recall_k(limit)

        # 主存储分支：PG（authoritative）或 SQLite，二选一。
        # backend=postgres 时 primary 恒为 postgres；PG 不可用由 _search_postgres 内部返回空，绝不进入 SQLite 分支。
        lex: List[MemoryEntry] = []
        if self.long_term_backend == "postgres":
            primary = "postgres"
            lex = self._search_postgres(
                tenant_id=tenant,
                user_id=user_id,
                query=query,
                memory_type=MemoryType.EPISODIC.value,
                namespace=None,
                thread_id=scoped_thread_id,
                limit=recall_k,
            )
        else:
            primary = "sqlite"
            lex = self.episodic.get_similar_tasks(user_id, query, recall_k)

        if primary == "postgres" and not lex:
            # PG lexical 0-hit 不代表 vector 没命中，继续 Milvus 召回。
            logger.info(
                "[memory] episodic search | primary=postgres lexical no hit | tenant=%s user=%s query=%s",
                tenant,
                user_id,
                query[:120],
            )

        # Milvus 向量召回（索引，只贡献 memory_id，最终内容回主存储 hydrate）
        vec: List[MemoryEntry] = []
        if self.enable_milvus and self._milvus_store:
            vec = self._search_milvus(
                tenant_id=tenant,
                user_id=user_id,
                query=query,
                memory_type=MemoryType.EPISODIC.value,
                namespace=None,
                thread_id=scoped_thread_id,
                limit=recall_k,
            )

        merged = self._merge_and_hydrate(primary, lex, vec, tenant, user_id, MemoryType.EPISODIC)
        if not merged:
            return []

        ranked = self._rerank(query, merged, limit)
        self._annotate_entries_with_source(ranked, primary)
        logger.info(
            "[memory] episodic search | primary=%s candidates=%d ranked=%d query=%s",
            primary,
            len(merged),
            len(ranked),
            query[:120],
        )
        return ranked

    def search_all(
        self,
        user_id: str,
        query: str,
        include_short_term: bool = False,
        short_term_thread_id: Optional[str] = None,
        limit_per_type: int = 5,
        tenant_id: Optional[str] = None,
        long_term_thread_id: Optional[str] = None,
    ) -> Dict[str, List[MemoryEntry]]:
        tenant = tenant_id or self.default_tenant_id
        results = {
            "semantic": self.search_semantic(
                query=query,
                user_id=user_id,
                limit=limit_per_type,
                tenant_id=tenant,
                thread_id=long_term_thread_id,
            ),
            "episodic": self.search_similar_tasks(
                query=query,
                user_id=user_id,
                limit=limit_per_type,
                tenant_id=tenant,
                thread_id=long_term_thread_id,
            ),
        }
        if include_short_term and short_term_thread_id:
            messages = self.get_short_term_messages(
                thread_id=short_term_thread_id,
                include_summary=True,
                last_n=limit_per_type,
                user_id=user_id,
                tenant_id=tenant,
            )
            results["short_term"] = [
                MemoryEntry(
                    content=str(message.content),
                    memory_type=MemoryType.SHORT_TERM,
                    user_id=user_id,
                    thread_id=short_term_thread_id,
                    metadata={"tenant_id": tenant},
                )
                for message in messages
            ]
        return results

    def get_context_for_agent(
        self,
        user_id: str,
        thread_id: str,
        query: Optional[str] = None,
        max_memories: int = 10,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        tenant = tenant_id or self.default_tenant_id
        context = {}
        context["user_profile"] = (
            self.get_user_profile(user_id, tenant_id=tenant)
            if self.long_term_scope == "user"
            else None
        )
        recent_messages = self.get_short_term_messages(
            thread_id=thread_id,
            last_n=5,
            user_id=user_id,
            tenant_id=tenant,
        )
        context["recent_messages"] = recent_messages
        if query:
            all_memories = self.search_all(
                user_id=user_id,
                query=query,
                limit_per_type=max_memories // 2,
                tenant_id=tenant,
                long_term_thread_id=thread_id,
            )
            # 保持各类型内部 rerank 顺序，不再按 created_at 覆盖相关性排序
            combined = []
            for mem_type, entries in all_memories.items():
                for entry in entries:
                    combined.append((entry, mem_type))
            context["relevant_memories"] = combined[:max_memories]
        context["recent_tasks"] = self.get_task_history(
            user_id,
            limit=3,
            tenant_id=tenant,
            thread_id=thread_id,
        )
        context["conversation_summary"] = self.get_short_term_summary(
            thread_id=thread_id,
            user_id=user_id,
            tenant_id=tenant,
        )
        logger.info(
            "[memory] context ready | tenant=%s user=%s thread=%s recent_msgs=%d relevant=%d tasks=%d summary_chars=%d",
            tenant,
            user_id,
            thread_id,
            len(context.get("recent_messages", [])),
            len(context.get("relevant_memories", [])),
            len(context.get("recent_tasks", [])),
            len(context.get("conversation_summary", "")),
        )
        return context

    def build_personalized_prompt_context(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        tenant_id: Optional[str] = None,
        max_memories: int = 8,
    ) -> str:
        """
        构建用于注入到大语言模型 (LLM) 的个性化 Prompt 上下文。

        主要功能：
        1. 上下文聚合：获取用户画像（长期记忆）、最近对话及摘要（短期记忆），并根据当前 query 检索最相关的记忆片段。
        2. Prompt 格式化：将聚合的上下文信息转换为结构化的 Markdown 文本，供 LLM 理解和使用，使其具备个性化和记忆能力。
        3. 链路追踪：记录详细的注入日志（如记忆来源分布、Milvus 检索命中详情、注入字符数等）至 `self._last_trace`，便于调试与监控。

        Args:
            user_id (str): 用户唯一标识。
            thread_id (str): 当前会话/线程 ID。
            query (str): 用户的当前查询。
            tenant_id (Optional[str]): 租户 ID，用于多租户数据隔离，默认使用系统默认租户。
            max_memories (int): 注入的相关记忆片段的最大数量。

        Returns:
            str: 格式化后的个性化 Prompt 上下文字符串。
        """
        self._last_milvus_raw_hits = []
        context = self.get_context_for_agent(
            user_id=user_id,
            thread_id=thread_id,
            query=query,
            max_memories=max_memories,
            tenant_id=tenant_id,
        )
        memory_entries = [item[0] for item in context.get("relevant_memories", [])]
        memory_text = format_memories_for_prompt(memory_entries, max_length=1800)
        profile_text = json.dumps(context.get("user_profile", {})) if context.get("user_profile") else ""
        summary_text = context.get("conversation_summary", "")
        recent_messages = context.get("recent_messages", [])
        recent_lines: List[str] = []
        for msg in recent_messages[-8:]:
            role = "用户"
            msg_type = getattr(msg, "type", "")
            if msg_type == "ai":
                role = "助手"
            text = str(getattr(msg, "content", "")).strip()
            if not text:
                continue
            if len(text) > 120:
                text = text[:120] + "..."
            recent_lines.append(f"- {role}: {text}")
        recent_text = "\n".join(recent_lines)
        sections = []
        if profile_text:
            sections.append(f"## 用户画像\n{profile_text}")
        if recent_text:
            sections.append(f"## 最近对话\n{recent_text}")
        if summary_text:
            sections.append(f"## 对话摘要\n{summary_text}")
        if memory_text:
            sections.append(memory_text)
        injected = "\n\n".join(sections).strip()
        trace_items = []
        source_count: Dict[str, int] = {}
        for item in memory_entries:
            source = item.metadata.get("retrieval_source", "unknown")
            source_count[source] = source_count.get(source, 0) + 1
            snippet = str(item.content)
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            trace_items.append(
                {
                    "id": item.id,
                    "type": item.memory_type.value,
                    "source": source,
                    "namespace": item.namespace,
                    "thread_id": item.thread_id,
                    "snippet": snippet,
                    "created_at": item.created_at.isoformat(),
                }
            )
        self._last_trace = {
            "tenant_id": tenant_id or self.default_tenant_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "query": query,
            "profile_injected": bool(profile_text),
            "summary_chars": len(summary_text),
            "memory_count": len(memory_entries),
            "source_count": source_count,
            "injected_chars": len(injected),
            "items": trace_items,
            "milvus_raw_hits": self._last_milvus_raw_hits,
            "injected_preview": {
                "profile": profile_text[:240] + ("..." if len(profile_text) > 240 else ""),
                "memory_text": memory_text[:480] + ("..." if len(memory_text) > 480 else ""),
            },
        }
        logger.info(
            "[memory] prompt injection | tenant=%s user=%s thread=%s profile=%s summary_chars=%d memories=%d injected_chars=%d injected_preview=%s",
            tenant_id or self.default_tenant_id,
            user_id,
            thread_id,
            bool(profile_text),
            len(summary_text),
            len(memory_entries),
            len(injected),
            json.dumps(
                {
                    "profile": profile_text[:120] + ("..." if len(profile_text) > 120 else ""),
                    "memory_text": memory_text[:180] + ("..." if len(memory_text) > 180 else ""),
                },
                ensure_ascii=False,
            ),
        )
        return injected

    def get_last_trace(self) -> Dict[str, Any]:
        return self._last_trace.copy()

    def persist_turn(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        query: str,
        answer: str,
    ) -> None:
        user_message = HumanMessage(content=query)
        ai_message = AIMessage(content=answer)
        self.add_short_term_messages(
            thread_id=thread_id,
            messages=[user_message, ai_message],
            user_id=user_id,
            tenant_id=tenant_id,
        )
        lower_query = query.lower()
        remember_markers = [
            "记住", "请记住", "记一下", "我叫", "我是", "叫我", "我的名字", "叫做",
            "我的偏好", "我偏好", "我喜欢", "我不喜欢", "我希望你", "回答偏好",
            "以后你", "你叫做", "你的偏好", "你要",
            "remember", "please remember", "my name is", "i am", "i'm",
            "call me", "i prefer", "i like", "my preference",
        ]
        should_extract_long_term = any(marker in lower_query for marker in remember_markers)
        extracted = extract_memory_from_messages([user_message]) if should_extract_long_term else {"facts": [], "preferences": []}

        def is_valid_candidate(text: str, allow_second_person: bool = False) -> bool:
            normalized = text.strip()
            if not normalized:
                return False
            lowered = normalized.lower()
            question_signals = ["?", "？", "什么", "吗", "how", "what", "why", "which"]
            if any(token in lowered for token in question_signals):
                return False
            if (normalized.startswith("你") and not allow_second_person) or normalized.startswith("请问"):
                return False
            return True

        allow_second_person = any(token in lower_query for token in ["以后你", "你叫做", "你的偏好", "回答偏好", "你要"])
        facts = [item for item in extracted.get("facts", []) if is_valid_candidate(item, allow_second_person=allow_second_person)]
        preferences = [item for item in extracted.get("preferences", []) if is_valid_candidate(item, allow_second_person=allow_second_person)]
        if should_extract_long_term and not facts and not preferences and is_valid_candidate(query, allow_second_person=allow_second_person):
            preferences = [query.strip()]

        for fact in facts:
            self.save_fact(
                user_id=user_id,
                fact=fact,
                category="user_fact",
                tenant_id=tenant_id,
                thread_id=thread_id,
            )
        if preferences:
            if self.long_term_scope == "user":
                self.save_user_profile(
                    user_id=user_id,
                    profile={"preferences": preferences},
                    merge=True,
                    tenant_id=tenant_id,
                )
            else:
                for pref in preferences:
                    self.save_fact(
                        user_id=user_id,
                        fact=pref,
                        category="user_preference",
                        tenant_id=tenant_id,
                        thread_id=thread_id,
                    )
        if self.save_conversation_task:
            self.save_task(
                user_id=user_id,
                task_type="conversation",
                task_data={"query": query},
                outcome=answer[:1200],
                tenant_id=tenant_id,
                thread_id=thread_id,
            )
        logger.info(
            "[memory] turn persisted | tenant=%s user=%s thread=%s short_backend=%s long_backend=%s scope=%s remember_mode=%s facts=%d prefs=%d save_task=%s",
            tenant_id,
            user_id,
            thread_id,
            self.short_term_backend,
            self.long_term_backend,
            self.long_term_scope,
            should_extract_long_term,
            len(facts),
            len(preferences),
            self.save_conversation_task,
        )

    def clear_user_memory(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, int]:
        tenant = tenant_id or self.default_tenant_id
        if memory_types is None:
            memory_types = ["semantic", "episodic", "short_term"]
        results = {}
        if self.enable_long_term and "semantic" in memory_types and self.long_term_backend != "postgres":
            results["semantic"] = self.semantic.clear(user_id=user_id)
        if self.enable_long_term and "episodic" in memory_types and self.long_term_backend != "postgres":
            results["episodic"] = self.episodic.clear(user_id=user_id)
        if "short_term" in memory_types:
            keys = []
            if self.short_term_backend == "redis" and self._redis_client is not None:
                keys.extend(self._redis_client.keys(f"ma:short:{tenant}:{user_id}:*"))
                keys.extend(self._redis_client.keys(f"ma:short:summary:{tenant}:{user_id}:*"))
                if keys:
                    self._redis_client.delete(*keys)
            if self.short_term_backend == "postgres" and self._postgres_dsn and psycopg:
                with psycopg.connect(self._postgres_dsn) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM short_term_messages WHERE tenant_id = %s AND user_id = %s",
                            (tenant, user_id),
                        )
                        cur.execute(
                            "DELETE FROM short_term_summaries WHERE tenant_id = %s AND user_id = %s",
                            (tenant, user_id),
                        )
                        conn.commit()
            results["short_term"] = len(keys)
        if self.long_term_backend == "postgres" and self._postgres_dsn and psycopg:
            with psycopg.connect(self._postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM memory_entries WHERE tenant_id = %s AND user_id = %s",
                        (tenant, user_id),
                    )
                    cur.execute(
                        "DELETE FROM user_profiles WHERE tenant_id = %s AND user_id = %s",
                        (tenant, user_id),
                    )
                    conn.commit()
        return results

    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        active_threads = len(self.list_active_threads())
        stats = {
            "short_term": {
                "active_threads": active_threads,
                "backend": self.short_term_backend,
            },
            "semantic": {
                "namespaces": [] if (not self.enable_long_term or self.long_term_backend == "postgres") else self.semantic.list_namespaces(user_id),
            },
            "episodic": {
                "namespaces": [] if (not self.enable_long_term or self.long_term_backend == "postgres") else self.episodic.list_namespaces(user_id),
            },
            "backends": {
                "postgres": bool(self._postgres_dsn and psycopg and self.long_term_backend == "postgres"),
                "milvus": bool(self._milvus_store and self.enable_milvus and self.enable_long_term),
            },
            "modes": {
                "short_term": self.short_term_backend,
                "long_term": self.long_term_backend,
                "long_term_scope": self.long_term_scope,
                "save_conversation_task": self.save_conversation_task,
            },
        }
        return stats
