# WABI — Next Steps

本文档基于当前代码库的工程审查，按优先级列出待完成事项。
每项标注了涉及文件、问题性质和预估工作量。

---

## Priority 1 — 小修（< 30 分钟，风险低）

### 1.1 删除无用导入与死变量

| 文件 | 问题 | 操作 |
|------|------|------|
| `agents/chitchat/agent.py:5` | `from langgraph.types import interrupt` 从未调用 | 删除 |
| `agents/goalplanning/agent.py:5` | 同上 | 删除 |
| `agents/goalplanning/agent.py:54` | `final_response = ""` 赋值后从未使用 | 删除 |
| `agents/chitchat/agent.py:83` | `msg: AnyMessage = ai_message` 多余转型，直接 return ai_message | 删除中间变量 |
| `agents/goalplanning/agent.py:85` | 同上 | 删除中间变量 |
| `agents/food_recognition/agent.py` | `finally: pass` 空块 | 删除整个 finally |

### 1.2 统一 Redis 默认 URL

`_get_redis()` 在三处使用了不同的 fallback URL：
- `chat_manager.py:30` → `redis://localhost:6379/0`
- `router.py:25` → `redis://redis:6379/0`（Docker 服务名）
- `ip_location.py:19` → `redis://localhost:6379/0`

**操作**：在 `config.py` 加 `REDIS_URL = os.getenv("WABI_REDIS_URL", "redis://redis:6379/0")`，
三处统一引用 `from langgraph_app.config import config; config.REDIS_URL`。

### 1.3 将局部 import 提升至模块顶部

| 文件 | 问题 |
|------|------|
| `router.py:93-97` | `from zoneinfo import ZoneInfo` / `from datetime import ...` 在函数体内 |
| `llm_factory.py:56` | `import copy` 在条件分支内 |
| `chat_manager.py` | `import json` 在 try 块内（多次） |

---

## Priority 2 — 中型重构（1-3 小时，影响可维护性）

### 2.1 提取共享重试逻辑

chitchat、goalplanning、food_recognition、food_recommendation 四个 agent 各自有完全相同的：
```python
sleep_times = [0.2, 0.5]
for attempt in range(3):
    try:
        ...
        break
    except Exception as e:
        last_error = e
        if attempt < 2:
            await asyncio.sleep(sleep_times[attempt])
```

**方案**：新建 `utils/agent_utils.py`，提供：
```python
async def invoke_with_retry(llm, messages, config, node_name, max_attempts=3) -> AIMessage
```
内部封装重试、日志、sleep 逻辑。每个 agent 调用一行代替 15 行。

同时将 `sleep_times = [0.2, 0.5]` 提升为 `config.py: RETRY_DELAYS = [0.2, 0.5]`。

### 2.2 提取 profile_context 构建

以下代码出现在 5 个文件中（chitchat、goalplanning、router、recognition、recommendation）：
```python
profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
    f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_profile.items() if v
)
```
**方案**：提取到 `utils/agent_utils.py:build_profile_context(user_profile) -> str`。

### 2.3 统一 logger 初始化

`guardrails/nodes.py` 使用 `setup_logger()`，其余所有节点使用 `get_logger()`。
两者功能相同。统一为 `get_logger()`。

### 2.4 Pub/Sub 轮询 → 事件驱动

`chat_manager.py` 仍使用 `get_message(timeout=1.0)` 轮询，最坏情况每条消息延迟 1 秒。

**方案**：
```python
try:
    msg = await asyncio.wait_for(
        pubsub.get_message(ignore_subscribe_messages=True, timeout=None),
        timeout=120.0
    )
except asyncio.TimeoutError:
    break
```
消除空转，失败立即响应。

---

## Priority 3 — BaseAgent 抽象

chitchat 和 goalplanning 的 agent.py 除 system prompt 外代码 100% 相同。
食物相关的两个 agent 也有大量重复的 profile_context 和错误处理结构。

**方案**：
```
agents/
  _base.py          # 共享: lang 检测、profile_context、retry wrapper、fallback
  chitchat/agent.py  # 只剩 system prompt (~15 行)
  goalplanning/agent.py  # 只剩 system prompt + 未来的历史解析逻辑
```

新增 agent 时只需继承基类并实现 `build_system_prompt(lang, profile_context) -> str`。

---

## Priority 4 — message_timestamps 清理

`state.py` 维护 `messages` 和 `message_timestamps` 两个并行数组，靠各 agent 手动同步。
任何一个节点只返回 `messages` 而忘记 `message_timestamps` 就会永久错位。

**方案**：时间戳存入 `BaseMessage.additional_kwargs["timestamp"]`，
`message_timestamps` 字段从 `GraphState` 和 `NodeOutput` 中删除。

---

## Priority 5 — Goalplanning 重设计

当前 goalplanning agent 只是一个带完整历史的 chitchat 变体。
计划中的功能包括：
- 判断同一时间段内上传多个食物，用户实际吃了哪一个
- 基于历史识别结果计算累计营养摄入
- 与用户健康目标对比，生成干预建议

**需要先设计：**
1. 如何在 message history 中标记"食物已确认/未确认"的状态
2. recognition_result 是否应该持久化到 DB（目前只存 AIMessage 文本）
3. goalplanning 的 state schema 扩展（累计热量、目标进度等）

此项在架构讨论后再动手。

---

## Priority 6 — 低优先级

- **WebSocket 多标签页**：`active_connections` 改为 `Dict[str, Set[WebSocket]]`，
  目前仅影响删除用户时的断连，实际业务影响极小。
- **增加沉默 catch 块的日志**：`ip_location.py` 的 Redis 写入失败、
  `food_recommendation/agent.py` 的 JSON 解析失败目前静默吞掉，建议加 `logger.debug`。
- **food_recognition finally 块**：已是 `pass`，可整体删除。

---

## 工作量估算

| 优先级 | 内容 | 预估时间 |
|--------|------|----------|
| P1 | 小修 6 项 | 20 分钟 |
| P2 | 重试抽象 + profile 提取 + logger + Pub/Sub | 3-4 小时 |
| P3 | BaseAgent 抽象 | 2 小时 |
| P4 | message_timestamps 清理 | 1 小时 |
| P5 | Goalplanning 重设计 | 需讨论后估算 |
| P6 | 低优先级杂项 | 1 小时 |
