"""Bocha 搜索引擎实现。"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

from .base import SearchProvider


logger = logging.getLogger("mult_agents")


class BochaSearchProvider(SearchProvider):
    """Bocha Web Search API 实现。"""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("BOCHA_API_KEY", "").strip()

    @property
    def name(self) -> str:
        return "bocha"

    def search(self, query: str, count: int = 8, **kwargs) -> list[dict]:
        logger.info("[bocha_search] 开始搜索 | query=%s | count=%s", query, count)
        logger.info("[bocha_search] API Key 状态 | 是否配置=%s | Key前缀=%s",
                    bool(self._api_key), self._api_key[:8] + "..." if self._api_key else "None")
        if not self._api_key:
            logger.warning("[bocha_search] 未配置 BOCHA_API_KEY，跳过搜索")
            return []

        payload = {
            "query": query,
            "summary": True,
            "freshness": kwargs.get("freshness", "noLimit"),
            "count": count,
        }
        request = urllib.request.Request(
            url="https://api.bocha.cn/v1/web-search",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            logger.info("[bocha_search] 发送请求 | url=%s", request.full_url)
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                logger.info("[bocha_search] 收到响应 | status=%s | content_length=%s", response.status, len(raw))
            result = json.loads(raw)
            logger.info("[bocha_search] 解析响应成功 | data字段存在=%s", "data" in result)
        except urllib.error.HTTPError as e:
            logger.error("[bocha_search] HTTP 错误 | code=%s | reason=%s", e.code, e.reason)
            return []
        except urllib.error.URLError as e:
            logger.error("[bocha_search] URL 错误 | reason=%s", e.reason)
            return []
        except json.JSONDecodeError as e:
            logger.error("[bocha_search] JSON 解析错误 | error=%s", e)
            return []
        except Exception as e:
            logger.error("[bocha_search] 未知错误 | error=%s | type=%s", e, type(e).__name__)
            return []

        data = result.get("data", {})
        pages = data.get("webPages", [])
        logger.info("[bocha_search] 解析数据 | webPages类型=%s", type(pages).__name__)
        if isinstance(pages, dict):
            if isinstance(pages.get("value"), list):
                pages = pages.get("value", [])
            elif isinstance(pages.get("items"), list):
                pages = pages.get("items", [])
            else:
                pages = []
        if not isinstance(pages, list):
            logger.warning("[bocha_search] webPages 格式异常 | type=%s", type(pages).__name__)
            return []

        logger.info("[bocha_search] 获取网页数量 | total=%s", len(pages))
        records: list[dict] = []
        for idx, page in enumerate(pages[:count], 1):
            if not isinstance(page, dict):
                logger.warning("[bocha_search] 第 %s 条记录格式异常 | type=%s", idx, type(page).__name__)
                continue
            url = str(page.get("url") or "").strip()
            domain = ""
            if "://" in url:
                domain = url.split("://", 1)[1].split("/", 1)[0]
            title = page.get("name") or f"web_result_{idx}"
            snippet = page.get("summary") or ""
            records.append({
                "source_id": f"WEB-{idx}",
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "source_type": "web",
                "published_at": page.get("datePublished") or page.get("dateLastCrawled") or "",
            })
        logger.info("[bocha_search] 搜索完成 | 返回记录数=%s", len(records))
        return records
