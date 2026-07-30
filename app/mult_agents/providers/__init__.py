"""搜索提供者工厂。"""

from typing import Optional

from .base import SearchProvider
from .bocha import BochaSearchProvider


_PROVIDER: Optional[SearchProvider] = None


def get_search_provider(name: str = "bocha", api_key: Optional[str] = None) -> SearchProvider:
    """获取搜索提供者实例（单例）。"""
    global _PROVIDER
    if _PROVIDER is None:
        if name == "bocha":
            _PROVIDER = BochaSearchProvider(api_key)
        else:
            raise ValueError(f"未知的搜索提供者: {name}")
    return _PROVIDER


def init_search_provider(name: str = "bocha", api_key: Optional[str] = None) -> SearchProvider:
    """初始化并返回搜索提供者（强制重新初始化）。"""
    global _PROVIDER
    if name == "bocha":
        _PROVIDER = BochaSearchProvider(api_key)
    else:
        raise ValueError(f"未知的搜索提供者: {name}")
    return _PROVIDER


__all__ = ["SearchProvider", "get_search_provider", "init_search_provider"]
