# Milvus 兼容性 Issue：langchain-milvus 0.3.3 构造失败

## 状态

- 类型：既有 bug（非本轮引入）
- 影响：`MemoryManager._init_milvus` 构造失败 → `_milvus_store=None` → **Milvus 向量召回从未真正生效**（一直静默降级到 PG/SQLite）；RAG 模块（`app/mult_agents/rag/core.py`）的 `_MilvusVectorStore` 构造同样受影响。
- 当前配置下 `config.json` 为 `enable_milvus=false`、`milvus_host=""`，Milvus 本就未启用，因此问题被掩盖。

## 当前实际版本（`F:\cloud_agent_env`）

| 包 | 版本 |
|---|---|
| langchain-milvus | **0.3.3** |
| pymilvus | **2.6.15** |

## Bug 根因

`langchain_milvus.Milvus.__init__`（0.3.3）内部逻辑：

```python
self._milvus_client = MilvusClient(**connection_args)   # 独立 MilvusClient
self.alias = self.client._using                          # "cm-<id>" 动态 alias
# 后续访问 self.col 时：
self._col_cache = Collection(self.collection_name, using=self.alias)
```

- `MilvusClient` 每次实例化生成动态 alias（`cm-<id>`），该 alias **从不注册到 pymilvus 全局 `connections`**。
- `Collection(using=<该 alias>)` 走全局 connections，找不到 → 抛 `ConnectionNotExistException: should create connection first.`
- `MemoryManager._init_milvus` 捕获异常后 `self._milvus_store=None`，静默降级。

**为什么 RAG 也受影响**：`rag/core.py` 虽先 `connections.connect(alias="default", ...)`，但 langchain 内部用的是自己的 `cm-<id>` alias，`default` 注册无济于事 → 同样 `ConnectFirst`。

**实测证据**（隔离 venv 验证）：
- 0.3.3 构造 `MilvusVectorStore` → 抛 `ConnectionNotExistException`（已复现）
- 手动 `connections.connect(alias="default")` 后再构造 0.3.3 → 仍失败（alias 不匹配）

## 修复方案（按优先级）

### 优先级 1：升级 langchain-milvus 0.3.3 → 0.4.0 ✅ 已实测解决

**验证方式**：独立 venv（`C:\Users\ian\AppData\Local\Temp\milvus_probe_venv`），装 `langchain-milvus==0.4.0`（依赖 **pymilvus 3.0.1**），连真实 Milvus（docker 19530），用独立测试 collection `milvus_probe_test`：

```
venv: langchain-milvus 0.4.0 | pymilvus 3.0.1
CONSTRUCT OK (no ConnectionNotExist)
write OK
expr search returned: 1
other-tenant search returned: 0
dropped test collection
```

**结论：0.4.0 修复了构造 bug**，expr 下推（Phase 1）在 0.4.0 下可真正生效。

**升级影响评估**：
- 0.4.0 强制依赖 pymilvus **3.0.1**（不是 2.x）。
- 项目里 pymilvus 直接使用点仅 `rag/core.py:9` 的 `from pymilvus import connections, utility`：
  - 3.0.1 仍提供 `connections`/`utility.has_collection`（已实测 import OK）
  - `connections.connect` 签名 2.6 与 3.0.1 **相同**（均无 `host`/`port`/`uri` 形参，走 `**kwargs`），RAG 的 `connect(alias="default", host=..., port=...)` 行为一致 → **不破坏 RAG**
- 其余 Milvus 调用：`manager.py` 用 `langchain_milvus.Milvus`，RAG 用同一类；无其他直接 pymilvus 调用（`MilvusClient` 仅测试脚本用，非生产代码）。
- 需回归验证：`pytest tests/`、RAG 检索、memory 检索。

### 优先级 2：若 0.4.0 仍有问题 → 兼容 pymilvus 2.5.x 组合

（未执行——0.4.0 已解决，此优先级暂不需要。若后续发现 0.4.0 其他不兼容，再测 `langchain-milvus` 某版本 + `pymilvus 2.5.x`。）

### 优先级 3（兜底，不作为默认）：alias 预注册 / subclass / monkeypatch

- **alias 预注册**：在构造前用 `connections.connect(alias=预测的 cm-xxx, ...)`——但 alias 是 `id(self._handler)` 动态生成，无法预测，不可行。
- **subclass**：继承 `Milvus` 覆写 `alias`/`col`，让 ORM Collection 复用全局 `default` 连接。可行但脆弱，随 langchain 版本迭代易碎。
- **monkeypatch**：patch `Collection` 或 `connections._fetch_handler`。最后手段，不推荐长期。

## 建议行动

1. 升级 `cloud_agent_env`：`pip install --upgrade langchain-milvus==0.4.0`（会连带 pymilvus → 3.0.1）
2. 同步更新 `requirements.txt` / `pyproject.toml` 版本约束
3. 回归验证：`pytest tests/`、真实 PG/Milvus 集成、RAG 检索
4. 若启用 Milvus，更新 `config.json`：`enable_milvus=true` + `milvus_host=127.0.0.1` + `milvus_collection=mult_agent_memory`

## 待办

- [ ] 在共享环境执行升级前，先在隔离 venv 完整回归（含 RAG 式连接、expr 查询）
- [ ] 确认 `requirements.txt` 的 `pymilvus>=2.4.0` 约束需更新为兼容 3.x
