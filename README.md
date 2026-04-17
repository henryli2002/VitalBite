# WABI — 健康饮食与营养分析

基于 LangGraph 的多智能体营养助手，支持食物识别、餐厅推荐和饮食目标规划。

---

## 架构

```
用户浏览器 (WebSocket)
      │
      ▼
wabi-web (FastAPI · 端口 8000)
  · WebSocket 长连接 · 多用户 Session · PostgreSQL 持久化
  · 将 AI 任务推入 Redis 任务队列 (wabi_ai_queue)
      │ Redis Pub/Sub
      ▼
wabi-ai (LangGraph Worker · 端口 8001)
  · 200 并发 Worker · Agent 编排流水线
  · 结果通过 Redis Pub/Sub 推回对应客户端
      │
      ▼
基础设施
  · PostgreSQL  — 聊天历史 · 用户档案
  · Redis       — 任务队列 · IP 地理位置缓存 · Pub/Sub
```

---

## Agent 流水线

```
用户消息
  └─► input_guardrail (安全检测)
        └─► router (意图识别 · 实时流式)
              ├─► recognition   (食物识别 · 4 步流水线)
              ├─► recommendation (餐厅推荐 · Google Maps)
              ├─► goalplanning  (目标规划 · 全量历史)
              └─► chitchat      (通用对话)
                    └─► output_guardrail
```

**意图分类（4 类）**

| 意图 | 触发条件 |
|------|----------|
| `recognition` | 消息含食物图片 |
| `recommendation` | 询问附近餐厅、饥饿信号 + 用餐时间 |
| `goalplanning` | 饮食规划、营养目标、历史摄入回顾 |
| `chitchat` | 默认：问候、模糊输入、无图识别请求 |

---

## 功能说明

- **多模态食物识别**：LLM 目标检测 → 图像裁剪 → 本地 fine-tuned 模型逐项预测营养 → LLM 汇总
- **动态时区**：前端传 IANA 时区，IP 自动回落，影响餐次判断（早/中/晚餐）、recognition 推荐评估、今日消息截断边界（3AM）
- **实时 Thinking 流**：router 节点原生 token streaming，前端气泡实时更新
- **多 LLM 后端**：统一接口支持 Google Gemini（默认）、OpenAI、AWS Bedrock、llama.cpp 本地模型
- **goalplanning 两阶段**：Phase 1 用今日消息快速检测意图，Phase 2 携带完整历史执行规划
- **并发背压**：Semaphore 限流，recognition ≤ 50 并发，chitchat ≤ 200 并发

---

## 技术栈

| 层 | 技术 |
|----|------|
| 编排框架 | LangGraph 0.3+ · LangChain 0.3+ |
| Web 服务 | FastAPI · Uvicorn · WebSocket |
| AI 模型 | Google Gemini 2.5 Flash · OpenAI · AWS Bedrock · llama.cpp |
| 数据库 | PostgreSQL (asyncpg · 连接池) |
| 缓存/队列 | Redis asyncio |
| 图像处理 | Pillow · base64 |
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

---

## 目录结构

系统经过标准重构，采用统一的 `src` 顶级包管理结构。

```
WABI/
├── src/
│   ├── langgraph_app/             # AI 核心业务逻辑 (LangGraph)
│   │   ├── agents/                # 各类 Agent (将降级为 Tools)
│   │   │   ├── chitchat/          # 通用对话
│   │   │   ├── goalplanning/      # 营养目标规划
│   │   │   ├── food_recognition/  # 多模态食物识别
│   │   │   └── food_recommendation/ # 餐厅推荐
│   │   ├── orchestrator/          # 工作流编排
│   │   │   ├── graph.py           # LangGraph 工作流定义
│   │   │   ├── state.py           # GraphState schema
│   │   │   └── nodes/             # 路由与安全网关
│   │   ├── tools/                 # 外部工具 (Maps, IP, etc.)
│   │   └── utils/                 # 通用脚手架 (重试, 并发, LLM 工厂)
│   ├── server/                    # Web / Worker 服务入口
│   │   ├── web.py                 # FastAPI API 入口 (WebSocket)
│   │   ├── ai.py                  # AI Worker 独立进程 (Dispatcher)
│   │   ├── chat_manager.py        # 会话管理与队列分发
│   │   ├── db.py                  # PostgreSQL 存储适配层
│   │   └── models.py              # Pydantic 数据模型
│   └── frontend/                  # 原生 JS/HTML 前端静态文件
├── eval/                          # 模型评估框架
├── tests/                         # 负载测试 · 准确率测试
├── docker/                        # 容器化配置
└── Next.md                        # 系统演进路线与架构蓝图
```

---

## 更新日志

### v3.2.0 (2026-04)
- **优雅重试与降级 (Retry & Cascade)**：实现了带有 Full Jitter 的指数退避重试 (`utils/retry.py`)，及基于 Llamacpp 的降级策略 (`utils/cascade.py`)，极大提升了对外部 API (LLM/Google Maps) 限流和超时的容错率。
- **并发调度器重构**：废弃了 200 个低效常驻 async worker，采用单 Dispatcher 监听队列，结合 `semaphores.py` 动态控制并发，消除了线程阻塞风险，提升了系统的垂直扩展能力。
- **降级 UI 标记**：LLM 降级触发时自动通过 `additional_kwargs["degraded"]` 和 `additional_kwargs["nutrition_source"]` 透传状态，为前端展示弱网/预估角标提供支持。

### v2.3.0 (2026-04)
- 时区全链路：前端传 IANA 时区 → IP 自动回落 → 影响餐次判断、recognition 推荐评估、3AM 日期边界
- Router 流式修复：删除 `TrackedChatModel` 包装层，恢复原生 token streaming
- Goalplanning 三重执行修复：3 次图执行 → 2 次，修复双发布 Bug
- Tutorial Agent 移除：相关功能合并入 chitchat，路由简化为 4 类意图
- Redis 单例化：chat_manager、router、food_recognition 统一模块级懒初始化
- N+1 查询修复：`list_users` 改为单次 `LEFT JOIN`
- `inject_dynamic_context` 转义 Bug 修复：`\\n\\n` → `\n\n`
- `datetime.utcnow()` 废弃警告消除：全部迁移至 `datetime.now(timezone.utc)`

### v2.2.0 (2026-03)
- 架构升级：SQLite → PostgreSQL + Redis 异步微服务
- 多模态增强：修复图像 base64 在 Redis 传输中的丢失问题
- 数据隔离：生产环境过滤测试账号
