# WABI — 健康饮食与营养分析

基于 LangGraph 的 Supervisor Agent 营养助手。食物识别、餐厅推荐、饮食规划，通过 Tool-calling 在一轮对话中组合执行。

---

## 架构

```
用户浏览器 (WebSocket)
      │
      ▼
wabi-web (FastAPI · 端口 8000)
  · WebSocket 长连接 · 多用户 Session · PostgreSQL 持久化
  · 图片拦截：base64 → image_store → UUID 占位符（DB 不再存原图）
  · 将 AI 任务推入 Redis 任务队列 (wabi_ai_queue)
      │ Redis Pub/Sub
      ▼
wabi-ai (LangGraph Worker · 端口 8001)
  · Supervisor Agent (react loop) + Tool Node
  · 结果通过 Redis Pub/Sub 推回对应客户端
      │
      ▼
基础设施
  · PostgreSQL  — 聊天历史 · 用户档案 · 图片 UUID 引用
  · Redis       — 任务队列 · IP 地理位置缓存 · Pub/Sub
  · 本地磁盘     — data/images/{user_id}/{uuid}.jpg 图片 Registry
```

---

## Agent 流水线

```
用户消息
  └─► input_guardrail (安全检测 · 正则 + LLM 双层)
        └─► supervisor_agent (react loop)
              │
              │  ┌────────── 决策 ──────────┐
              │  │                          │
              │  ▼                          ▼
              │  直接回复               Tool 调用
              │                         · analyze_food_image(image_uuid)
              │                         · search_restaurants(query, ...)
              │                         └─► 结果喂回 → 再次决策
              ▼
        output_guardrail (仅 LLM 内容安全，跳过正则)
              └─► END
```

**Supervisor 核心特性**
- 系统提示注入：用户档案、长期画像、当前时间/餐次、每日热量参考、语言
- `MAX_TOOL_CALLS_PER_TURN` 限流；Tool 失败返回 `{"error": ...}` 由 Supervisor 决策降级
- `user_id`/`user_context` 通过 `RunnableConfig.configurable` 隐式传递，不污染 Tool 签名
- 流式 thinking：Supervisor 内部 react loop 会在 Tool 调用前后推送细粒度状态给前端，例如“正在识别图片...”→“识别完成，发现 3 道食物。”→“正在搜索餐厅...”→“生成回复中...”

---

## 功能说明

- **多模态食物识别**：LLM 目标检测 → 图像裁剪 → 本地 fine-tuned 模型逐项预测营养 → Supervisor 生成总结
- **图片 Registry**：WebSocket 拦截 base64，落盘为 `data/images/{user_id}/{uuid}.jpg`，DB 只保留占位符 `[image: uuid | 汉堡+薯条, 850kcal]`。消除 base64 在数据库与 Redis Pub/Sub 中的膨胀。
- **结构化渲染**：前端用 vendored markdown-it 解析；特殊控件通过 fenced block 表达（` ```restaurants ` / ` ```nutrition `），widget handler 拦截渲染为卡片，未识别 fence 回退为普通代码块。
- **动态时区**：前端传 IANA 时区，IP 自动回落，影响餐次判断、日期边界（3AM）
- **多 LLM 后端**：统一接口支持 Google Gemini（默认）、OpenAI、AWS Bedrock、llama.cpp 本地模型
- **双层 Guardrail**：输入侧正则 + LLM 双层；输出侧跳过正则（避免对模型自身 Markdown/内联代码/健康建议措辞误判），仅走 LLM 内容安全
- **并发背压**：Semaphore 限流，recognition/recommendation/chitchat 独立通道

---

## 技术栈

| 层 | 技术 |
|----|------|
| 编排框架 | LangGraph 0.6+ · LangChain 0.3+ |
| Agent | Supervisor via `create_react_agent` (LLM ↔ Tool 循环) |
| Web 服务 | FastAPI · Uvicorn · WebSocket |
| AI 模型 | Google Gemini 2.5 Flash · OpenAI · AWS Bedrock · llama.cpp |
| 数据库 | PostgreSQL (asyncpg · 连接池) |
| 缓存/队列 | Redis asyncio |
| 图像处理 | Pillow · 本地磁盘 Registry |
| 前端渲染 | 原生 JS + vendored markdown-it (CommonMark) |
| 容器化 | Docker · Docker Compose |

---

## 快速开始

### 1. 环境配置

复制 `.env.example` 为 `.env`：

```env
# LLM
GOOGLE_API_KEY="AIza..."
LLM_PROVIDER="gemini"
GEMINI_MODEL_NAME="gemini-2.5-flash-lite"

# Maps
GOOGLE_MAPS_API_KEY="..."

# 基础设施 (Docker Compose 默认配置)
WABI_REDIS_URL="redis://wabi-redis:6380/0"
WABI_DB_URL="postgresql://wabi_user:wabi_password@wabi-postgres:5433/wabi_chat"
```

### 2. 启动

```bash
docker-compose up -d --build
```

| 服务 | 地址 |
|------|------|
| 前端 Web 界面 | http://localhost:8000 |
| AI Worker 健康检查 | http://localhost:8001/health |
| 独立数据库 (PostgreSQL) | localhost:5433 |
| 独立缓存/队列 (Redis) | localhost:6380 |

### 3. 历史图片迁移（一次性，仅老数据需要）

```bash
docker exec wabi-ai python scripts/migrate_images.py
```

将 messages 表里残留的 base64 抽出存盘并替换为 UUID 占位符。

---

## 目录结构

```
WABI/
├── src/
│   ├── langgraph_app/
│   │   ├── orchestrator/
│   │   │   ├── graph.py                # input_guardrail → supervisor → output_guardrail
│   │   │   ├── supervisor.py           # Supervisor react loop + system prompt
│   │   │   ├── supervisor_state.py     # 简化 State
│   │   │   └── nodes/
│   │   │       └── guardrails/         # 正则 + LLM 双层
│   │   ├── tools/
│   │   │   ├── food_recognition_tool.py  # analyze_food_image (uuid 入参)
│   │   │   ├── recommendation_tool.py    # search_restaurants
│   │   │   ├── map/                      # Google Maps / IP 定位
│   │   │   └── tools.py
│   │   ├── agents/                     # 保留 predictor.py / schemas.py；旧 agent.py 待清理
│   │   └── utils/
│   │       ├── agent_utils.py          # build_profile_context / detect_meal_time / TDEE
│   │       ├── cascade.py              # LLM 降级级联
│   │       ├── retry.py                # 指数退避 + Full Jitter
│   │       └── semaphores.py
│   ├── server/
│   │   ├── web.py                      # FastAPI + WebSocket + /api/images/{uuid}
│   │   ├── ai.py                       # AI Worker Dispatcher
│   │   ├── chat_manager.py             # 会话管理 · 最近 N 条历史加载
│   │   ├── image_store.py              # 图片 Registry（save/load/update_description）
│   │   ├── db.py                       # PostgresHistoryStore
│   │   └── models.py
│   └── frontend/
│       ├── index.html
│       ├── app.js                      # 渲染 · markdown-it · fence handlers
│       ├── styles.css
│       └── vendor/markdown-it.min.js
├── scripts/
│   ├── migrate_images.py               # 老数据 base64 → UUID 迁移
│   ├── smoke_image_refs.py             # /api/images/{uuid} 链路烟雾
│   └── smoke_markdown_it.js            # 前端渲染器烟雾 (node, 无 headless 浏览器)
├── eval/
├── tests/
├── docker/
├── data/images/                        # 图片 Registry (运行时生成)
└── Next.md                             # 系统演进路线
```

---

## 更新日志

### v4.1.0 (2026-04) — 图像预处理 + UUID 修复 + 餐次统一
- **图像预处理 Letterbox**：底层模型裁剪后强制拉伸导致估算过大。改用"裁剪图居中 + 白边填充正方形"策略（`food_recognition_tool.py:319`），保留原始宽高比，模拟训练时的拍摄视角。
- **Supervisor UUID 修复**：多轮对话中 LLM 可能把 `<attached_image description=...>` 的描述文本当作 UUID 传入工具。强化 prompt 强调必须传 32 位 hex UUID。
- **餐次判定统一**：前端使用和后端一致的判定逻辑（早餐 7-9:30, 午餐 11:30-13:30, 晚餐 17:30-19:30）。

### v4.0.0 (2026-04) — Supervisor 迁移 + 图片 Registry
- **Supervisor Agent**：Router + 4 Agent 静态 DAG 替换为 `create_react_agent` 的 LLM ↔ Tool 循环。支持复合意图（"看看健康吗，并推荐类似的"）单轮解决。
- **图片 Registry**：WebSocket 层拦截 base64，落盘 `data/images/{user_id}/{uuid}.jpg`，DB 只存占位符。消除 messages 表与 Redis Pub/Sub 上的 base64 膨胀；识别工具返回后自动回写 `汉堡+薯条, 850kcal` 描述。历史数据可通过 `scripts/migrate_images.py` 迁移。
- **渲染栈重构**：手写 `renderMarkdown` 下线，改用 vendored markdown-it。营养卡与餐厅卡统一经由 ` ```nutrition ` / ` ```restaurants ` fenced block 表达，widget handler 拦截；未识别 fence 回退普通 code block。
- **Output Guardrail 修正**：输出侧跳过正则层，避免模型自身 Markdown 措辞（inline code、"you must"、JSON 示例）被误判为 prompt injection；LLM 内容安全层仍然生效。
- **图片渲染一致性**：刚上传的图片与 DB 回读的图片统一走 `figure.chat-image` 包裹，CSS 尺寸约束一致（不再"发送时巨大 / 刷新后正常"）。

### v3.2.0 (2026-04)
- **优雅重试与降级**：Full Jitter 指数退避 (`utils/retry.py`) + 基于 llamacpp 的降级 (`utils/cascade.py`)，对外部 API 限流和超时的容错率显著提升。
- **并发调度器重构**：废弃 200 个低效常驻 async worker，改为单 Dispatcher + `semaphores.py` 动态并发控制。
- **降级 UI 标记**：LLM 降级触发时通过 `additional_kwargs["degraded"]` / `additional_kwargs["nutrition_source"]` 透传给前端。

### v2.3.0 (2026-04)
- 时区全链路：前端传 IANA 时区 → IP 回落 → 影响餐次判断、3AM 日期边界
- Router 流式修复；Goalplanning 三重执行修复；Tutorial Agent 合并入 chitchat
- Redis 单例化；`list_users` N+1 修复

### v2.2.0 (2026-03)
- 架构升级：SQLite → PostgreSQL + Redis 异步微服务
- 多模态增强：修复图像 base64 在 Redis 传输中的丢失
- 数据隔离：生产环境过滤测试账号
