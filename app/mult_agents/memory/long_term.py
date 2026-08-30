"""
长期记忆模块

提供语义记忆（Semantic）和情景记忆（Episodic）的持久化存储
支持 PostgreSQL 和内存两种后端
"""

import json
import logging
import re
import sqlite3
from abc import ABC
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jieba
from rank_bm25 import BM25Okapi

from langchain_huggingface import HuggingFaceEmbeddings

from .base import BaseMemory, MemoryEntry, MemoryType

logger = logging.getLogger("mult_agents.memory")


def content_to_text(content: Any) -> str:
    """将记忆内容统一转换为可检索文本（供记忆检索/重排复用）"""
    if isinstance(content, str):
        return content

    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def tokenize_for_bm25(text: str) -> List[str]:
    """中英文混合文本的 BM25 分词（供记忆检索/重排复用）"""
    if not text:
        return []

    tokens = jieba.lcut(text.lower(), cut_all=False)

    return [
        token.strip()
        for token in tokens
        if token.strip()
        and re.search(r"[\w一-鿿]", token)
    ]


class BaseLongTermMemory(BaseMemory, ABC):
    """长期记忆基类"""


    def __init__(self, memory_type: MemoryType, embedding_model_path: str = ""):
        super().__init__(memory_type)
        self._embedding_dim: Optional[int] = None
        self._embeddings = None
        if embedding_model_path:
            try:
                self._embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
                logger.info("长期记忆嵌入模型已加载: %s", embedding_model_path)
            except Exception as exc:
                logger.warning("长期记忆嵌入模型加载失败，降级关键词匹配: %s", exc)
    
    def _generate_embedding(self, text: str) -> List[float]:
        if self._embeddings is not None:
            try:
                return self._embeddings.embed_query(text)
            except Exception as exc:
                logger.warning("嵌入生成失败，降级关键词匹配: %s", exc)
        return []
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _content_to_text(self, content: Any) -> str:
        """将记忆内容统一转换为可检索文本"""
        return content_to_text(content)

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """中英文混合文本的 BM25 分词"""
        return tokenize_for_bm25(text)


class SQLiteLongTermMemory(BaseLongTermMemory):
    """
    基于 SQLite 的长期记忆实现
    
    适合开发和测试环境，生产环境建议使用 PostgreSQL
    """
    
    def __init__(
        self,
        memory_type: MemoryType,
        db_path: Optional[str] = None,
        embedding_model_path: str = "",
    ):
        super().__init__(memory_type, embedding_model_path)
        
        if db_path is None:
            db_path = str(Path(__file__).resolve().parents[2] / "data" / "memory.db")
        
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_tables()
    
    def _ensure_db_directory(self) -> None:
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self) -> None:
        """初始化数据库表"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    user_id TEXT,
                    namespace TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user 
                ON memories(user_id, memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_namespace 
                ON memories(namespace)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created 
                ON memories(created_at)
            """)
            
            conn.commit()
    
    def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目"""
        # 如果没有 embedding，自动生成
        if entry.embedding is None and isinstance(entry.content, str):
            entry.embedding = self._generate_embedding(entry.content)
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, memory_type, user_id, namespace, metadata, embedding,
                 created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                json.dumps(entry.content) if isinstance(entry.content, dict) else entry.content,
                entry.memory_type.value,
                entry.user_id,
                entry.namespace,
                json.dumps(entry.metadata),
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.created_at.isoformat(),
                datetime.now().isoformat(),
                entry.access_count,
            ))
            conn.commit()
        
        logger.debug(f"长期记忆已保存: {entry.id}")
        return entry.id
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """获取指定 ID 的记忆"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return self._row_to_entry(row)
    
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 5,
        **kwargs
    ) -> List[MemoryEntry]:
        """
        搜索记忆
        
        使用向量相似度 + 关键词匹配
        """
        query_embedding = self._generate_embedding(query)
        
        # 构建查询条件
        conditions = ["memory_type = ?"]
        params = [self.memory_type.value]
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        
        where_clause = " AND ".join(conditions)
        
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where_clause}",
                params
            ).fetchall()
        
        # 将数据库行统一转为 MemoryEntry，便于后续打分
        entries = [self._row_to_entry(row) for row in rows]

        if not entries:
            return []

        # --------------------------------------------------
        # 1. BM25 词法相关性
        # --------------------------------------------------
        query_tokens = self._tokenize_for_bm25(query)

        corpus_tokens = [
            self._tokenize_for_bm25(self._content_to_text(entry.content))
            for entry in entries
        ]

        bm25_scores = [0.0] * len(entries)

        if query_tokens and any(corpus_tokens):
            bm25 = BM25Okapi(corpus_tokens)
            raw_bm25_scores = bm25.get_scores(query_tokens)

            # BM25 分数归一化到 0~1，便于与向量分数融合
            # 注意：rank_bm25.get_scores() 返回 numpy 数组，需先转成 float 列表
            raw_bm25_list = [float(s) for s in raw_bm25_scores]

            max_bm25 = max(raw_bm25_list) if raw_bm25_list else 0.0

            if max_bm25 > 0:
                bm25_scores = [
                    max(0.0, score) / max_bm25
                    for score in raw_bm25_list
                ]

        # --------------------------------------------------
        # 2. Embedding + BM25 + Exact Match 综合评分
        # --------------------------------------------------
        entries_with_scores = []

        query_lower = query.strip().lower()

        for index, entry in enumerate(entries):

            # 向量语义相似度（余弦相似度，夹到 [0,1]）
            vec_score = 0.0
            if entry.embedding and query_embedding:
                vec_score = self._calculate_similarity(entry.embedding, query_embedding)
                vec_score = max(0.0, min(1.0, vec_score))

            # BM25 词法分数
            bm25_score = bm25_scores[index]

            # 整句完全命中加分（保留原关键词逻辑，降级为小 bonus）
            content_str = self._content_to_text(entry.content).lower()
            exact_score = 1.0 if query_lower and query_lower in content_str else 0.0

            if query_embedding:
                # 正常情况：语义检索为主，BM25 为辅
                total_score = (
                    vec_score * 0.70
                    + bm25_score * 0.25
                    + exact_score * 0.05
                )
            else:
                # Embedding 不可用时自动退化为 BM25 关键词检索
                total_score = bm25_score * 0.95 + exact_score * 0.05

            entries_with_scores.append((entry, total_score))

        # 按分数排序并返回
        entries_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in entries_with_scores[:limit]]
    
    def delete(self, memory_id: str) -> bool:
        """删除指定记忆"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def clear(
        self,
        user_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> int:
        """清除记忆"""
        conditions = ["memory_type = ?"]
        params = [self.memory_type.value]
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        
        where_clause = " AND ".join(conditions)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE {where_clause}",
                params
            )
            conn.commit()
            return cursor.rowcount
    
    def list_namespaces(self, user_id: Optional[str] = None) -> List[str]:
        """列出所有命名空间"""
        query = "SELECT DISTINCT namespace FROM memories WHERE memory_type = ?"
        params = [self.memory_type.value]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [row["namespace"] for row in rows if row["namespace"]]
    
    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """将数据库行转换为 MemoryEntry"""
        content = row["content"]
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass
        
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except json.JSONDecodeError:
                pass
        
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass
        
        return MemoryEntry(
            id=row["id"],
            content=content,
            memory_type=MemoryType(row["memory_type"]),
            user_id=row["user_id"],
            namespace=row["namespace"],
            metadata=metadata,
            embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"],
        )


class SemanticMemoryStore(SQLiteLongTermMemory):
    """
    语义记忆存储
    
    存储事实知识、用户画像、偏好设置等
    使用 namespace 区分不同类型的语义记忆：
    - "user_profile": 用户画像
    - "preferences": 用户偏好
    - "facts": 事实知识
    """
    
    def __init__(self, db_path: Optional[str] = None, embedding_model_path: str = ""):
        super().__init__(MemoryType.SEMANTIC, db_path, embedding_model_path)
    
    def save_profile(
        self,
        user_id: str,
        profile_data: Dict[str, Any],
        merge: bool = True
    ) -> str:
        """
        保存用户画像
        
        Args:
            user_id: 用户标识
            profile_data: 画像数据
            merge: 是否合并现有画像
            
        Returns:
            记忆条目 ID
        """
        if merge:
            # 获取现有画像
            existing = self.get_profile(user_id)
            if existing:
                existing.update(profile_data)
                profile_data = existing
        
        entry = MemoryEntry(
            content=profile_data,
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            namespace="user_profile",
            metadata={"type": "profile", "version": "1.0"},
        )
        
        return self.save(entry)
    
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像"""
        results = self.search(
            query="user_profile",
            user_id=user_id,
            namespace="user_profile",
            limit=1
        )
        
        if results:
            content = results[0].content
            if isinstance(content, dict):
                return content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return None
    
    def save_fact(
        self,
        user_id: str,
        fact: str,
        category: Optional[str] = None
    ) -> str:
        """
        保存事实知识
        
        Args:
            user_id: 用户标识
            fact: 事实内容
            category: 事实类别
            
        Returns:
            记忆条目 ID
        """
        entry = MemoryEntry(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            namespace=f"facts/{category}" if category else "facts",
            metadata={"category": category},
        )
        
        return self.save(entry)


class EpisodicMemoryStore(SQLiteLongTermMemory):
    """
    情景记忆存储
    
    存储历史任务、执行轨迹、交互记录
    使用 namespace 区分不同类型的情景记忆：
    - "tasks/{task_type}": 任务执行记录
    - "conversations": 对话历史，没有独立方法，合并到 tasks/conversation
    - "actions": 操作记录                 ❌ 完全没实现

    """
    
    def __init__(self, db_path: Optional[str] = None, embedding_model_path: str = ""):
        super().__init__(MemoryType.EPISODIC, db_path, embedding_model_path)
    
    def save_task_record(
        self,
        user_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        outcome: Optional[str] = None
    ) -> str:
        """
        保存任务执行记录
        
        Args:
            user_id: 用户标识
            task_type: 任务类型
            task_data: 任务数据
            outcome: 任务结果
            
        Returns:
            记忆条目 ID
        """
        content = {
            "task_type": task_type,
            "data": task_data,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        }
        
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.EPISODIC,
            user_id=user_id,
            namespace=f"tasks/{task_type}",
            metadata={"task_type": task_type, "has_outcome": outcome is not None},
        )
        
        return self.save(entry)
    
    def get_similar_tasks(
        self,
        user_id: str,
        task_description: str,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """
        获取相似的历史任务
        
        Args:
            user_id: 用户标识
            task_description: 任务描述
            limit: 返回数量
            
        Returns:
            相似任务列表
        """
        return self.search(
            query=task_description,
            user_id=user_id,
            limit=limit
        )
    
    def get_task_history(
        self,
        user_id: str,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        获取任务历史
        
        Args:
            user_id: 用户标识
            task_type: 任务类型过滤
            limit: 返回数量
            
        Returns:
            任务记录列表
        """
        namespace = f"tasks/{task_type}" if task_type else None
        
        # 获取所有匹配的任务
        with self._get_connection() as conn:
            query = """
                SELECT * FROM memories 
                WHERE memory_type = ? AND user_id = ?
            """
            params = [self.memory_type.value, user_id]
            
            if namespace:
                query += " AND namespace = ?"
                params.append(namespace)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_entry(row) for row in rows]


# 别名，便于使用
LongTermMemory = SQLiteLongTermMemory
