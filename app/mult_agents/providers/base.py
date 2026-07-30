"""搜索提供者抽象基类。定义统一的搜索接口，支持切换不同的搜索引擎实现。"""

from abc import ABC, abstractmethod


class SearchProvider(ABC):
    """搜索提供者接口。实现此接口以接入不同的搜索引擎。"""

    @abstractmethod
    def search(self, query: str, count: int = 8, **kwargs) -> list[dict]:
        """执行搜索并返回结构化结果列表。

        返回的每个 dict 应包含:
            - source_id (str): 来源ID前缀，例如 "WEB-1"
            - title (str): 结果标题
            - url (str): 结果链接
            - snippet (str): 结果摘要
            - domain (str): 来源域名
            - published_at (str): 发布日期（可选）
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称，用于日志和配置标识。"""
        ...
