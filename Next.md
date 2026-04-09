# WABI — Next Steps

本文档基于当前代码库的工程审查，按优先级列出待完成事项。

---

## 已完成

| 项目 | 内容 |
|------|------|
| P1.1 | 删除无用导入与死变量 |
| P1.2 | 统一 Redis 默认 URL → `config.REDIS_URL` |
| P1.3 | 局部 import 提升至模块顶部 |
| 2.4 | Pub/Sub 轮询 → `pubsub.listen()` + `asyncio.timeout` |
| P4 | `message_timestamps` → `additional_kwargs["timestamp"]` |

---

## Priority 1 — 重试机制与优雅降级

### 1.1 当前问题全景

对所有可失败子单元的审计结果：

| 位置 | 子单元 | 重试 | 优雅降级 |
|------|--------|------|----------|
| `guardrails/nodes.py` | LLM safety check | 3 次，但 `sleep(1)` 不统一 | ✓ 失败默认 safe |
| `router.py` | LLM stream | ✓ 3 次，固定间隔 | ✓ fallback chitchat |
| `recognition` Step 2 | LLM 目标检测 | ✗ 单次 | △ 直接跳全图，无重试 |
| `recognition` Step 3 | 本地模型预测 | ✗ | ✗ 直接返回错误消息 |
| `recognition` Step 4 | LLM 汇总 | ✗ | ✗ 直接返回错误消息 |
| `recommendation` | LLM 参数提取 | ✓ 3 次，固定间隔 | — |
| `recommendation` | Google Maps HTTP | ✗ | ✗ 静默返回 [] |
| `recommendation` | LLM 格式化 | ✓ 3 次，固定间隔 | — |
| `chitchat` | LLM | ✓ 3 次，固定间隔 | ✗ hardcoded 假响应 |
| `goalplanning` | LLM | ✓ 3 次，固定间隔 | ✗ hardcoded 假响应 |

### 1.2 重试：指数退避 + Full Jitter

当前所有节点用固定延迟 `[0.2, 0.5]`。固定延迟在高并发下有 thundering herd 问题——多个请求同时失败后在完全相同的时刻重试，第二波冲击与第一波等强。

**工业标准（AWS/Google/Netflix）**：指数退避 + full jitter：

```python
# utils/retry.py
import random, asyncio

async def with_retry(coro_fn, attempts=3, base=0.5, cap=10.0, fallback=_RAISE):
    """
    指数退避 + full jitter。
    sleep = random(0, min(cap, base * 2 ** attempt))
    fallback: 耗尽后返回该值；不传则重新抛出最后一个异常。
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as e:
            last_err = e
            if attempt < attempts - 1:
                sleep = random.uniform(0, min(cap, base * (2 ** attempt)))
                await asyncio.sleep(sleep)
    if fallback is _RAISE:
        raise last_err
    return fallback
```

不同类别的 `base` 参数：

| 类别 | base | cap | 说明 |
|------|------|-----|------|
| 轻量 LLM（chitchat、goalplanning、router、guardrails） | 0.3 | 5.0 | 请求小，限速恢复快 |
| 重量 LLM（recognition Step 2/4，含图片） | 0.8 | 15.0 | 图片请求 token 量大，429 恢复慢 |
| 外部 HTTP（Google Maps） | 1.0 | 20.0 | 外部服务 SLA 不可控 |

本地模型（`predict_nutrition`）不适用退避——它是 CPU 同步调用，失败原因是内存/模型问题，不是瞬时限速，见 §1.3。

### 1.3 降级：模型 Cascade，而非写死字符串

当前 chitchat/goalplanning 的最终 fallback 是：
```python
"您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助"
```

这是一个**假响应**——完全无视了用户刚才说了什么，用一个通用问候掩盖服务失败。

**正确设计**：主力模型 → 降级模型 → 诚实告知

```
Primary (Gemini 2.5 Flash / configured)
    ↓ 重试耗尽
Fallback (llamacpp 本地模型)
    ↓ 也失败 or 不可用
Honest error: "服务暂时不可用，请稍后再试"
```

**llamacpp 的上下文限制处理**：

llamacpp 上下文窗口短。降级时不能直接传全量历史，需截断：

```python
# config.py 新增
FALLBACK_LLM_MAX_HISTORY_TURNS: int = int(os.getenv("FALLBACK_LLM_MAX_HISTORY_TURNS", "5"))
```

`5 turns = 最近 5 条 HumanMessage + 5 条 AIMessage`，覆盖大部分对话上下文。

降级时应在响应中附加轻量标记，前端可决定是否显示"降级模式"角标：
```python
ai_message.additional_kwargs["degraded"] = True
```

**各节点降级设计**：

| 节点 | 主力失败后 | 降级失败后 |
|------|-----------|-----------|
| chitchat | llamacpp（截断至 k 轮） | "服务暂时不可用，请稍后重试" |
| goalplanning | llamacpp（截断至 k 轮） | "服务暂时不可用，请稍后重试" |
| recognition Step 2 | 重试 3 次 → `detected_items = []`（全图降级，现有逻辑） | — |
| recognition Step 3 | 跳过本地模型，改用 LLM 直接按食物名估算营养 | 插入零值占位，汇总时注明"无法获取精确数据" |
| recognition Step 4 | llamacpp（仅传数值摘要，无图片，上下文极短） | 直接格式化数字，不经 LLM |
| recommendation LLM | llamacpp（截断至 k 轮） | "服务暂时不可用，请稍后重试" |
| Google Maps | 重试 3 次 → "无法查询附近餐厅，请稍后再试" | — |
| guardrails | 默认 safe（现有逻辑合理） | — |

**recognition Step 3 的具体降级**：
本地模型失败不应走 zeros 路线——已知食物名（Step 2 的检测结果），完全可以让 LLM 给一个基于名称的营养估算。质量低于本地模型，但远好于 zeros 和错误消息。这一降级标记为 `additional_kwargs["nutrition_source"] = "llm_estimate"`，让前端可以显示"估算值"提示。

---

## Priority 2 — Worker 架构修正

### 2.1 200 个 async worker → 单 dispatcher + Semaphore

`langgraph_server.py` 当前启动 200 个 `worker_loop` coroutine，每个串行处理任务（`await process_task` 完成才取下一个）。这是用线程池思维写 async 代码的典型误区：

- 200 条 Redis `blpop` 长连接常驻，绝大多数时候全部闲置
- 真正控制并发上限的是各 agent 的 Semaphore，worker 数量与实际吞吐无关
- 这 200 个 coroutine 全在同一个 event loop 同一个线程里，没有真正的并行

**修改方案**：

```python
# langgraph_server.py
async def dispatcher():
    """单连接持续拉取任务，每个任务独立 create_task 异步运行。"""
    while True:
        try:
            task_data = await redis_client.blpop("wabi_ai_queue", timeout=0)
            if task_data:
                asyncio.create_task(process_task(json.loads(task_data[1])))
        except Exception as e:
            logger.error(f"Dispatcher error: {e}")
            await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(dispatcher())  # 一个 dispatcher 即可
```

实际并发上限由 Semaphore 保证，不需要 worker 数量来控制。`MAX_CONCURRENT_WORKERS = 200` 这个常量可以删掉，语义转移到 `semaphores.py` 里的各节点限制。

---

## Priority 3 — 中型重构（影响可维护性）

### 3.1 提取 profile_context 构建

5 处相同代码 → `utils/agent_utils.py:build_profile_context(user_profile) -> str`。

### 2.2 统一 logger 初始化

`guardrails/nodes.py` 用 `setup_logger()`，其余用 `get_logger()`，统一为 `get_logger()`。

---

## Priority 3 — BaseAgent 抽象

chitchat 和 goalplanning 除 system prompt 外代码 100% 相同，食物 agent 也有大量重复。

**依赖**：P1 重试工具先完成。

```
agents/
  _base.py           # lang 检测、profile_context、retry cascade、fallback
  chitchat/agent.py  # 只剩 system prompt
  goalplanning/agent.py  # system prompt + 未来的历史解析逻辑
```

---

## Priority 4 — 单元测试与压力测试

每个子单元需要：
- **功能测试**：正常路径 + 边界条件（空输入、无图片、模型失败 mock）
- **降级路径测试**：主力模型失败时验证 cascade 行为正确
- **压力测试**：并发验证 Semaphore 限流、退避散开（jitter 有效时不会出现同步冲击波）

优先级：`with_retry` 工具本身 > recognition 流水线 > agent cascade > guardrails

---

## Priority 5 — Goalplanning 重设计

当前只是带完整历史的 chitchat 变体。计划功能：
- 判断同一时间段上传多图，用户实际吃了哪一个
- 基于历史识别结果计算累计营养摄入
- 与用户健康目标对比，生成干预建议

**需先设计**：
1. message history 中如何标记"食物已确认/未确认"
2. recognition_result 是否持久化到 DB
3. state schema 扩展（累计热量、目标进度）

---

## Priority 6 — 低优先级

- **WebSocket 多标签页**：`active_connections` 改为 `Dict[str, Set[WebSocket]]`
- **静默 catch 块**：`ip_location.py`、`food_recommendation/agent.py` 吞掉的异常加 `logger.debug`
- **`recommendation` schema bug**：`Restaurant.user_ratings_total` 在 Places API v1 不存在，改为 `Optional[int] = None` 并在模板中条件渲染

---

## 工作量估算

| 优先级 | 内容 | 预估时间 |
|--------|------|----------|
| P1 | `with_retry` 工具 + 各节点接入 + llamacpp cascade | 3 小时 |
| P2 | profile_context 提取 + logger 统一 | 1 小时 |
| P3 | BaseAgent 抽象（依赖 P1） | 2 小时 |
| P4 | 单元测试 + 压力测试 | 3-4 小时 |
| P5 | Goalplanning 重设计 | 需讨论后估算 |
| P6 | 低优先级杂项 | 1 小时 |
