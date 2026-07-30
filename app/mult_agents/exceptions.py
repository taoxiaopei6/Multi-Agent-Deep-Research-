"""领域异常定义：统一项目中所有组件抛出的异常类型。

使错误处理可预测、可捕获，避免裸 Exception 满天飞。
"""


class DeepResearchError(Exception):
    """所有领域异常的基类。"""
    def __init__(self, message: str = "", cause: Exception | None = None):
        self.message = message
        self.cause = cause
        super().__init__(message)


class ConfigError(DeepResearchError):
    """配置错误：API Key 缺失、配置文件格式错误等。"""
    pass


class SearchProviderError(DeepResearchError):
    """搜索服务异常：Bocha 不可用、网络错误、限流等。"""
    def __init__(self, message: str = "", provider: str = "", cause: Exception | None = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}", cause)


class MemoryStorageError(DeepResearchError):
    """记忆存储异常：Redis/PostgreSQL/Milvus 连接失败等。"""
    def __init__(self, message: str = "", backend: str = "", cause: Exception | None = None):
        self.backend = backend
        super().__init__(f"[{backend}] {message}", cause)


class WorkflowExecutionError(DeepResearchError):
    """工作流执行异常：Agent 节点执行失败、状态异常等。"""
    def __init__(self, message: str = "", node: str = "", cause: Exception | None = None):
        self.node = node
        super().__init__(f"[{node}] {message}", cause)


class LLMServiceError(DeepResearchError):
    """LLM 服务异常：API 调用失败、响应格式异常等。"""
    pass


class JSONParseError(DeepResearchError):
    """JSON 解析异常：LLM 返回的结构无法解析为合法 JSON。"""
    def __init__(self, message: str = "", raw_output: str = "", node: str = ""):
        self.raw_output = raw_output[:500]
        self.node = node
        super().__init__(f"[{node}] {message}")
