# WABI Backend Refactoring & Optimization Timeline

本计划基于对 WABI 后端代码的深度架构诊断，按照**从底层基础设施到上层业务逻辑**、**从高危 Bug 到代码整洁度**的优先级进行排序。

总预计耗时：**约 2-3 个工作日**（或 12-16 个专注编程小时）。

---

## 🗓️ Phase 1: 核心基础设施与高危性能修复 (Core Infrastructure)
**时间预估: 半天 (约 3-4 小时)**
**目标:** 消除可能导致服务器在高并发下崩溃的连接泄漏和 N+1 数据库查询，修复多端登录断连问题。

*   **[ ] 1.1 全局 Redis 连接池 (Redis Pool)**
    *   **问题**: `chat_manager.py` 和 `router.py` 为每条消息频繁新建/销毁 Redis 客户端。
    *   **计划**: 创建一个全局单例的异步 Redis 连接池（如 `backend/redis_client.py`），供所有模块复用。
*   **[ ] 1.2 重写 Pub/Sub 监听机制**
    *   **问题**: `chat_manager.py` 使用低效的 `while True` + `timeout=1.0` 轮询。
    *   **计划**: 改写为高效的 `async for message in pubsub.listen()` 异步事件驱动模式。
*   **[ ] 1.3 修复 DB N+1 查询风暴**
    *   **问题**: `db.py` 中 `list_users` 遍历用户列表去单独 `SELECT COUNT(*)` 查询消息数。
    *   **计划**: 使用 `LEFT JOIN messages GROUP BY users.id` 重写为单次 SQL 查询。
*   **[ ] 1.4 修复 WebSocket 多端覆盖 Bug**
    *   **问题**: `web_server.py` 中 `active_connections[user_id]` 会覆盖同一个用户的多个浏览器标签页。
    *   **计划**: 将字典结构改为 `Dict[str, Set[WebSocket]]`，支持单用户多端同时在线和独立断开。

---

## 🗓️ Phase 2: LLM 追踪与 Utils 目录重构 (LLM Tracking & Utils)
**时间预估: 半天 (约 3-4 小时)**
**目标:** 消除反模式的 Wrapper，统一且稳健的 Token 与耗时追踪。

*   **[ ] 2.1 移除重型反模式 Wrapper**
    *   **计划**: 彻底删除 `tracked_llm.py`（过度包装）和 `llm_callback.py`（未使用）。
*   **[ ] 2.2 实现标准 Callback 追踪器**
    *   **计划**: 编写轻量级的 `TokenTrackingCallback(BaseCallbackHandler)`，仅监听 `on_llm_end` 和 `on_llm_error`，统一提取 input/output/reasoning/cache tokens。
*   **[ ] 2.3 整合 `llm_factory.py`**
    *   **计划**: 在工厂实例化大模型时，直接将 Callback 绑定。对外依然提供 `get_tracked_llm` 接口以保证现有 Agent 代码的完全向后兼容。

---

## 🗓️ Phase 3: Router 重构与 Agent 样板代码消除 (Router & Agents)
**时间预估: 大半天 (约 4-5 小时)**
**目标:** 保留前端流式体验，但大幅提升路由解析的健壮性；消除三个对话 Agent 中 90% 的重复代码。

*   **[ ] 3.1 Router 健壮性与流式优化**
    *   **计划**: 将 `router.py` 的提示词从 `INTENT: xxx` 改为 XML 标签格式 (`<intent>xxx</intent><reasoning>yyy</reasoning>`)。
    *   **计划**: 更新正则解析逻辑，使其只提取 XML 标签内的内容，彻底免疫乱序和格式错误。
    *   **计划**: 接入 Phase 1 建立的全局 Redis 池进行 partial streaming 发布。
    *   **计划**: 修复硬编码的服务器时区判定 (`time.localtime()`)。
*   **[ ] 3.2 抽象 `BaseAgent` 消除重复**
    *   **计划**: `chitchat`, `tutorial`, `goalplanning` 中重复的多语言检测、Prompt 组装、重试机制 (`for attempt in range(3)`) 和降级 fallback，全部提取到一个共享的辅助函数或基类中。
    *   **计划**: 重写这三个 Agent，使其代码行数从上百行缩减至仅保留核心系统提示词定义的 30 行左右。

---

## 🗓️ Phase 4: 业务逻辑致命 Bug 修复与 State 清理 (Business Logic)
**时间预估: 几小时 (约 2-3 小时)**
**目标:** 修复浪费 3 倍 Token 的致命逻辑 Bug，确保数据状态同步。

*   **[ ] 4.1 修复 Goalplanning "三重执行" Bug**
    *   **问题**: 触发目标规划时，Worker 会跑一遍日常对话，然后再跑一遍全量历史对话。
    *   **计划**: 梳理 `chat_manager.py` 和 `langgraph_server.py` 的判断逻辑，确保意图为 `goalplanning` 时，直接加载全量历史并一次性执行正确的子图，不产生重复的 Job。
*   **[ ] 4.2 清理 `state.py` 的幽灵数组**
    *   **计划**: 移除独立的 `message_timestamps` 数组，将时间戳等元数据直接存入 LangChain `BaseMessage.additional_kwargs`，从根源上杜绝消息和时间戳数量不对齐的潜在 Bug。

---

## 🚦 审核与执行
请审核上述 Timeline 与重构计划。
如果您同意该路线图，我们可以从 **Phase 1 (核心基础设施)** 开始逐步执行，每完成一个 Phase 都可以进行独立的测试和验证，确保服务始终可用。