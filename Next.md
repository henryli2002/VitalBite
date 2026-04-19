# WABI — Supervisor Agent 迁移路线图

本文档定义�� WABI Chat 系统从静态 Router + Workflow 架构迁移至 Supervisor Agent + Tools 架构的分阶段实施计划。

当前架构：`input_guardrail → router → [4个独立Agent之一] → output_guardrail → END`
目标架构：`input_guardrail → supervisor_agent (react loop + tools) → output_guardrail → END`

---

## Phase 1: 核心大脑切换 (Basic Agentic Migration) ✅ 已完成

**目标**：彻底废除静态 Workflow，建立 Supervisor 调度机制。

### 行动 1.1: 创建 Tool 目录，封装业务 Tool

**analyze_food_image Tool** — `src/langgraph_app/tools/food_recognition_tool.py`

从 `agents/food_recognition/agent.py` 中剥离 Step 1-3（图片提取、目标检测、裁剪预测）为独立 `@tool`：
- 输入：`image_base64: str`（显式传入）
- 输出：`{"items": [...], "total_calories": N, "total_nutrition": {...}, "nutrition_source": "local_model"|"llm_estimate"}`
- **不生成自然语言总结**（交给 Supervisor 自己包装语言）
- 保留 `@with_semaphore("recognition")` 并发控制
- 复用现有函数：`predict_nutrition()`、`_estimate_nutrition_with_llm()`、`FoodDetection` schema
- 新增 `decode_base64_image(b64_str: str) -> bytes` 在 `predictor.py` 中（替代从 messages 扫描 base64 的旧逻辑）

**search_restaurants Tool** — `src/langgraph_app/tools/recommendation_tool.py`

从 `agents/food_recommendation/agent.py` 中剥离搜索逻辑：
- 输入：`query: str`, `cuisine_type: Optional[str]`, `lat: Optional[float]`, `lng: Optional[float]`, `radius_km: float = 5.0`, `max_results: int = 5`
- 输出：原始餐厅 JSON 列表
- **不做 LLM query 提取**（Supervisor 自己从对话中理解意图直接传参）
- **不做 LLM 格式化**（Supervisor 自己包装语言推荐给用户）
- 直接调用已有的 `search_restaurants_tool` (`tools/tools.py`)
- 位置回退逻辑（frontend GPS → IP geolocation）移入 Tool 内部

**涉及文件**：
- 新建：`src/langgraph_app/tools/food_recognition_tool.py`
- 新建：`src/langgraph_app/tools/recommendation_tool.py`
- 修改：`src/langgraph_app/agents/food_recognition/predictor.py` — 新增 `decode_base64_image()`
- 修改：`src/langgraph_app/tools/__init__.py` — 导出新 Tool

### 行动 1.2: 编写 Supervisor

**文件**：`src/langgraph_app/orchestrator/supervisor.py`

Supervisor 是一个 react loop（LLM 调用 → 判断是否需要 Tool → 执行 Tool → 结果喂回 → 再次 LLM 调用 → 直到无 Tool 调用或达到上限）。

**System Prompt 构建**（注入信息）：
- `build_profile_context(user_profile)` — 用户基本信息（提取公共函数到 `utils/agent_utils.py`，消除 5 处重复）
- `behavioral_notes` — 长期画像（Phase 4 才有数据，先预留占位）
- 当前时间 + 餐时判断（复用 router.py 中 meal time 检测逻辑）
- TDEE 估算（复用 `_calculate_tdee()`，移至 `utils/agent_utils.py`）
- 语言检测（复用 `get_dominant_language()`）

**Prompt 核心指令**：
```
你是 WABI，一个全局健康规划师。
用户信息：{profile_context}
长期特征：{behavioral_notes}
当前时间：{current_time}，餐时：{meal_time}
每日热量需求：{daily_cal_ref}

规则：
1. 收到食物图片 → 必须调用 analyze_food_image，然后基于结果生成营养总结表格
2. 用户找餐厅 → 调用 search_restaurants
3. 普通闲聊 / 目标规划 → 直接回复，不调用工具
4. 复合需求 → 可连续调用多个工具
5. 回复语言跟随用户
```

**安全控制**：
- `MAX_TOOL_CALLS_PER_TURN = 5` — 超过强制生成回复
- Tool 失败不崩溃，返回 `{"error": "..."}` 给 Supervisor，由它决定告知用户还是降级处理
- 总超时 60s

**Tool 获取 user_id/user_context 的方式**：通过 LangGraph `RunnableConfig.configurable` 传递，不作为 Tool 参数。

**涉及文件**：
- 新建：`src/langgraph_app/orchestrator/supervisor.py`
- 新建：`src/langgraph_app/utils/agent_utils.py` — `build_profile_context()`, `_calculate_tdee()`, `detect_meal_time()`

### 行动 1.3: 重写 graph.py

将线性 DAG 替换为 Agent Node ↔ Tool Node 循环结构：
```
input_guardrail → supervisor_agent (react loop) → output_guardrail → END
```

**Feature Flag**：`USE_SUPERVISOR` 环境变量（默认 `1`，保留老图代码可切回 `0`）。

**State 简化** — 新建 `supervisor_state.py`：
- 保留：`messages`, `user_id`, `user_name`, `user_profile`, `user_context`, `response_channel`, `analysis`, `debug_logs`
- 移除：`recognition_result`, `recommendation_result`, `meal_time`（变为 Tool 内部关注点）

**涉及文件**：
- 修改：`src/langgraph_app/orchestrator/graph.py`
- 新建：`src/langgraph_app/orchestrator/supervisor_state.py`
- 修改：`src/langgraph_app/config.py` — 添加 `USE_SUPERVISOR`、`MAX_TOOL_CALLS_PER_TURN`

### 行动 1.4: 修改 ai.py 和 chat_manager.py 适配

**ai.py**：
- 扩展 `build_thinking_partial()` 处理 Supervisor 的 tool_call / tool_result 事件
- `process_task()` 的 `initial_state` 对应新 SupervisorState

**chat_manager.py**：
- **删除 Phase 2 goalplanning 重跑逻辑**（两阶段处理在 Supervisor 架构下不再需要）
- 历史加载改为最近 N 条消息（如 50 条），取代按时间边界加载
- payload 增加 `behavioral_notes` 字段

**涉及文件**：
- 修改：`src/server/ai.py`
- 修改：`src/server/chat_manager.py`
- 修改：`src/server/db.py` — 新增 `load_recent_messages(user_id, limit)` 方法

### 验收标准

1. `USE_SUPERVISOR=1`：发 "你好" → 直接回复，无 Tool 调用
2. 发一张食物图片 → Supervisor 调用 `analyze_food_image`，基于返回数据自己生成营养总结表格
3. 发 "推荐附近餐厅" → Supervisor 调用 `search_restaurants`，自己包装语言（如"我查到了，这有一家..."）
4. `USE_SUPERVISOR=0` → 老流程完全正常（回滚保障）
5. 前端 thinking 指示器正常显示工具调用进度
6. 没有明显延迟增加

---

## Phase 2: 多模态存储变革 (Image Registry & Migration) ✅ 已完成

**目标**：解决数据库 Base64 膨胀问题，实现图文分离与按需回溯。

### 图片 Description 策略

**不单独调 LLM 生成描述**。Description 来自 `analyze_food_image` Tool 的返回值（零成本副产品）：
- Tool 返回 `{items: [{name: "汉堡"}, {name: "薯条"}], total_calories: 850}` 后
- 提取食物名 + 热量拼成描述：`"汉堡+薯条, 850kcal"`
- 回写到 messages 表占位符中
- 非食物图片（Supervisor 未调用识别工具）→ 无描述，`[图片: {uuid}]`，可接受

前端显示：检测到 `[图片: {uuid}...]` 模式 → 调 `GET /api/images/{uuid}` 加载原图渲染，不丢失图片历史。

### 行动 2.1: 图片存储服务 + WebSocket 拦截

**新建** `src/server/image_store.py`：
- `save_image(user_id, image_bytes, mime_type) -> uuid` — 存到 `data/images/{user_id}/{uuid}.jpg`
- `load_image(uuid) -> bytes` — 按 UUID 读取
- `update_description(message_id, description)` — 回写占位符描述

**修改** `src/server/web.py`：
- WebSocket 收到含图消息时：提取 base64 → `save_image()` → 生成 UUID
- messages 表存 `[图片: {uuid}]`（不存 base64）
- 图片 bytes 通过 Supervisor state 内存传递给 AI worker（不经过 DB）

**新增** REST 端点 `GET /api/images/{uuid}`：
- 前端渲染历史消息时调此接口加载图片

### 行动 2.2: 修改 analyze_food_image Tool

将 Tool 输入参数从 `image_base64: str` 改为 `image_uuid: str`，内部通过 UUID 从磁盘读取。

Tool 返回后，从结果中提取食物名+热量拼成短描述，UPDATE messages 表占位符为 `[图片: {uuid} | 汉堡+薯条, 850kcal]`。

### 行动 2.3: 历史数据迁移脚本

**文件**：`scripts/migrate_images.py`

1. 扫描 messages 表中 `has_image = 1` 且包含 base64 的记录
2. 提取 base64 → `save_image()` 存磁盘 → 生成 UUID
3. 从同一对话的 assistant 回复中尝试提取食物信息作为 description（尽力而为，无则留空）
4. 用占位符替换原记录中的 base64，UPDATE 回数据库

### 验收标准

1. 新发的图片消息，messages 表中不再有 base64，只有 `[图片: {uuid}]` 占位符
2. `analyze_food_image` 工具通过 UUID 正常读取图片并分析
3. 工具返回后，占位符自动更新为 `[图片: {uuid} | 汉堡+薯条, 850kcal]`
4. 前端通过 `GET /api/images/{uuid}` 正常渲染历史图片
5. 迁移脚本运行后，数据库体积显著缩小

---

## Phase 2.5: 前端渲染栈重构 ✅ 已完成

**目标**：消除手写 `renderMarkdown` 的脆弱正则链，统一特殊控件的表达方式。

- **vendored markdown-it**：`src/frontend/vendor/markdown-it.min.js`，CommonMark 解析器替换手写 80 行正则渲染。`html: false` 默认 XSS 安全。
- **Fence-based 控件协议**：`nutrition` / `restaurants` 两类特殊渲染改为 fenced block，LLM 只需产出 JSON，widget handler 渲染为卡片。未识别的 fence 降级为普通 `<code>`。
- **图片统一包裹**：刚上传的 data URL 与 DB 回读的 `/api/images/{uuid}` 统一走 `<figure class="chat-image">`，CSS 尺寸约束一致。
- **Output Guardrail 修正**：输出侧跳过正则检测（自生成文本不是攻击向量），只留 LLM 内容安全。修复了"列 Markdown 健康指南表被拦截"等假阳性。
- **烟雾测试**：`scripts/smoke_markdown_it.js` 在 node `vm.createContext` 中覆盖 8 个核心渲染路径。

---

## Phase 3: 复合任务拆解 (Multi-Step Intent Resolution)

**目标**：让 Agent 能在一轮对话中聪明地连续执行多步操作。

> 注：Supervisor 本身已是 react loop，具备工具串联能力。此 Phase 聚焦于 Prompt 强化与用户感知层反馈。

### 行动 3.1: 优化 Supervisor Prompt

强化多步推理指令：
- "你可以连续调用工具，直到收集齐所有信息再回答"
- "调用一个工具后，评估结果，决定是否需要调用下一个工具"
- "对于'看看这个食物健康吗，并推荐类似的'，你应该先识别图片，再用识别结果中的菜系信息搜索餐厅"

### 行动 3.2: 流式工具调用反馈 (Tool Call Streaming)

在 Supervisor react loop 的每次工具调用前后推送细粒度状态：
- 调用前："Agent 正在识别图片..."
- 调用后："识别完成，发现 3 道菜。Agent 正在搜索餐厅..."
- 最终："生成回复中..."

利用现有 Redis Pub/Sub `{"status": "partial"}` 协议，前端已支持 `thinking` 类型消息，无需前端改动。

### 验收标准

1. "看看这个健康吗，并推荐点类似的" → 自动串行 `analyze_food_image` + `search_restaurants`，完整回答
2. 前端显示分步进度："正在识别..." → "正在搜索餐厅..." → 最终回复

---

## Phase 4: 长期规划与快照记忆 (Weekly Planner & Chrono-Snapshots)

**目标**：引入周粒度的心智模型与动态目标锁定机制。

### 行动 4.1: Profile 历史快照表

在 `src/server/db.py` 中新建 `user_profiles_history` 表：
- `id`, `user_id`, `year_week` (如 "2026-W16"), `profile_snapshot` (JSON), `behavioral_notes`, `created_at`
- 唯一约束：`(user_id, year_week)` — 保证幂等性

每周日任务运行时，除了更新主表 `users.profile_json`，还插入一条历史快照。

### 行动 4.2: 实现 log_meal_confirmed Tool

**文件**：`src/langgraph_app/tools/meal_tools.py`

- 输入：`meal_data: dict`（items, total_calories, protein, carbs, fat）
- 调用已有的 `PostgresHistoryStore.save_meal_log()`（db.py 中已实现）
- **幂等性**：`meal_logs` 表增加 `meal_hash` 列（hash(user_id + timestamp + items_json)），唯一约束 + `INSERT ... ON CONFLICT DO NOTHING`
- Supervisor 只有在确认用户真实吃了某食物后（图片识别正常、用户未否认）才调用此 Tool

### 行动 4.3: Weekly Planner 离线任务

使用 APScheduler + PostgreSQL persistent job store：
1. 每周日 23:59 触发
2. 遍历活跃用户，读取本周 `meal_logs`
3. 用 LLM 提炼 behavioral_notes（如 "本周碳水偏高，蛋白质不足"）
4. 事务中同时：更新 `users.profile_json` 的 `behavioral_notes` + 插入 `user_profiles_history` 快照 + 插入 `weekly_summaries` 记录
5. 幂等：`weekly_summaries` 表用 `(user_id, year_week)` 唯一约束

**behavioral_notes 质量控制**：
- 长度上限 500 字符，超出时 LLM 压缩合并
- 每条追加时间戳（如 `[2026-W16] 本周碳水偏高`）
- 标记来源：`[auto]`（系统生成）vs `[user]`（用户手写），Supervisor 优先信任 `[user]`
- 用户可通过 `PUT /api/users/{id}/profile` 编辑/删除

**并发安全**：
- 单实例：`asyncio.Lock` per user_id
- 多实例：`pg_advisory_lock(hash(user_id))` + APScheduler `pg_try_advisory_lock` 防重复执行

### 验收标准

1. Supervisor 识别图片后自动调用 `log_meal_confirmed` 入库，重复不产生重复记���
2. Weekly Planner 生成的 behavioral_notes 出现在 Supervisor 的 system prompt 中
3. 服务重启后 APScheduler 自动补偿 missed job

---

## Phase 清理: 删除遗留代码

在 Phase 3 / Phase 4 验证通过后：
- `USE_SUPERVISOR` 默认改为 `1`，删除老图代码
- 删除：`agents/chitchat/agent.py`, `agents/goalplanning/agent.py`, `agents/food_recognition/agent.py`（保留 `predictor.py`, `schemas.py`）, `agents/food_recommendation/agent.py`, `orchestrator/nodes/router.py`
- 合并 `supervisor_state.py` → `state.py`

**已完成的小清理**（每次 PR 顺带收尾）：
- ✅ 根目录 `test_*.py` / `get_last_msg.py` 等探针脚本 (v4.0.0)
- ✅ `renderMarkdown` 手写正则链 (Phase 2.5)

---

## 依赖关系

```
Phase 1 ✅ (Supervisor + Tools + graph 重写 + chat_manager 简化)
  │
  v
Phase 2 ✅ (Image Registry)
  │
  v
Phase 2.5 ✅ (markdown-it + fence 控件 + guardrail 修正)
  │
  v
Phase 3  (复合任务 prompt + 流式反馈)   ← 下一步
  │
  v
Phase 4  (Weekly Planner + meal log + 快照)
  │
  v
清理遗留代码
```

---

## 关键文件索引

| 文件 | 作用 | Phase |
|------|------|-------|
| `src/langgraph_app/orchestrator/graph.py` | 核心图定义，需要重写 | 1 |
| `src/langgraph_app/orchestrator/supervisor.py` | **新建** Supervisor react loop | 1 |
| `src/langgraph_app/orchestrator/supervisor_state.py` | **新建** 简化 State | 1 |
| `src/langgraph_app/tools/food_recognition_tool.py` | **新建** 图片识别 Tool | 1 |
| `src/langgraph_app/tools/recommendation_tool.py` | **新建** 餐厅搜索 Tool | 1 |
| `src/langgraph_app/utils/agent_utils.py` | **新建** 公共函数 | 1 |
| `src/server/chat_manager.py` | 删除两阶段逻辑，简化历史加载 | 1 |
| `src/server/ai.py` | 适配 Supervisor 事件流 | 1 |
| `src/server/web.py` | 图片拦截 + UUID 存储 | 2 |
| `src/server/image_store.py` | **新建** 图片存储服务 | 2 |
| `src/langgraph_app/tools/meal_tools.py` | **新建** 饮食记录工具 | 4 |
| `src/server/weekly_planner.py` | **新建** 周报生成 | 4 |
| `src/server/db.py` | 新表 + 新方法 | 1, 4 |

## 可复用的现有代码

| 函数/模块 | 位置 | 复用于 |
|-----------|------|--------|
| `predict_nutrition()` | `agents/food_recognition/predictor.py` | food_recognition_tool |
| `FoodDetection` schema | `agents/food_recognition/schemas.py` | food_recognition_tool |
| `search_restaurants_tool` | `tools/tools.py` | recommendation_tool |
| `get_location_from_ip_async` | `tools/map/ip_location.py` | recommendation_tool |
| `invoke_with_cascade()` | `utils/cascade.py` | Supervisor fallback |
| `with_semaphore()` | `utils/semaphores.py` | 所有 Tool |
| `with_retry()` | `utils/retry.py` | 所有 Tool |
| `get_dominant_language()` | `utils/utils.py` | Supervisor |
| `inject_dynamic_context()` | `utils/llm_factory.py` | Supervisor |
| `save_meal_log()` / `load_meal_logs()` | `server/db.py` | meal_tools |
