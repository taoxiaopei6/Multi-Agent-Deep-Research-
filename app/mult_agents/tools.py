"""工具模块：封装 Web 检索、本地 RAG 查询与通用辅助工具函数。"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .rag.core import RAGSystem, RAGConfig
from .providers import get_search_provider

logger = logging.getLogger("mult_agents")

# 全局 RAG 系统实例
_RAG_SYSTEM: Optional[RAGSystem] = None


def init_rag_system(api_key: str, config: Optional[RAGConfig] = None):
    """初始化全局 RAG 系统"""
    global _RAG_SYSTEM
    if _RAG_SYSTEM is None:
        try:
            _RAG_SYSTEM = RAGSystem(api_key, config)
        except Exception as e:
            logger.warning("RAG 系统初始化失败: %s", e)


def search_knowledge_base_records(query: str, limit: int = 5) -> list[dict]:
    if _RAG_SYSTEM is None:
        return []
    try:
        return _RAG_SYSTEM.search_records(query, k=limit)
    except Exception:
        return []


def bocha_web_search_records(query: str, count: int = 8) -> list[dict]:
    """网络检索（通过 SearchProvider 接口）。"""
    provider = get_search_provider()
    return provider.search(query, count=count)
