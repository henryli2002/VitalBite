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

# 基础设施
WABI_REDIS_URL="redis://wabi-redis:6379/0"
WABI_DB_URL="postgresql://wabi_user:wabi_password@wabi-postgres:5432/wabi_chat"
```

### 2. 启动

```bash
docker-compose up -d --build
```

| 服务 | 地址 |
|------|------|
| 前端 Web 界面 | http://localhost:8000 |
| AI Worker 健康检查 | http://localhost:8001/health |

---

## 目录结构

```
WABI/
├── backend/
│   ├── langgraph_app/
│   │   ├── agents/
│   │   │   ├── chitchat/          # 通用对话
│   │   │   ├── goalplanning/      # 营养目标规划
│   │   │   ├── food_recognition/  # 多模态食物识别（4 步流水线）
│   │   │   └── food_recommendation/ # 地理位置餐厅推荐
│   │   ├── orchestrator/
│   │   │   ├── graph.py           # LangGraph 工作流定义
│   │   │   ├── state.py           # GraphState schema
│   │   │   └── nodes/
│   │   │       ├── router.py      # 意图路由（流式）
│   │   │       └── guardrails/    # 输入/输出安全检测
│   │   ├── tools/
│   │   │   └── map/               # IP 地理位置 · Google Maps
│   │   ├── utils/
│   │   │   ├── llm_factory.py     # LLM 单例缓存工厂
│   │   │   ├── llm_callback.py    # Token 使用追踪 Callback
│   │   │   └── semaphores.py      # 并发背压信号量
│   │   └── config.py              # 统一配置管理
│   ├── langgraph_server.py        # AI Worker 入口（Consumer）
│   ├── web_server.py              # FastAPI API 入口（Producer）
│   ├── chat_manager.py            # 会话流控 · Redis 调度 · 两阶段 goalplanning
│   ├── db.py                      # PostgreSQL 存储适配层
│   └── models.py                  # Pydantic 数据模型
├── frontend/                      # 原生 JS 前端
├── eval/                          # 模型评估框架
├── tests/                         # 负载测试 · 准确率测试
├── Next.md                        # 下一步工程计划
└── docker/
    ├── Dockerfile.ai
    └── Dockerfile.web
```

---

## 更新日志

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


## 归档环境恢复（2026-11-06）

原项目内未发现可导出的虚拟环境。本次使用 uv 0.13.5，根据现有依赖配置重新解析并生成 `requirements.lock`，固定直接/传递依赖及发布包哈希。这不是历史环境的 freeze。锁面向本机 macOS Apple Silicon / Python 3.12，其他操作系统或 CUDA 环境需单独解析；完整服务运行还需要模型、API 凭据及数据库。

```bash
uv venv .venv --python 3.12
uv pip sync requirements.lock --require-hashes
uv pip install --no-deps -e .
```

`requirements.lock` 是 uv pip 格式锁，恢复时使用以上命令。后续增加依赖后需重新生成锁，不能只改宽松版本声明。升级命令（先评估兼容性）：`uv pip compile pyproject.toml archive-extras.in --python-version 3.12 --generate-hashes -o requirements.lock`。
锁定覆盖运行时缺失的 mlx-lm 和食物评测使用的 timm；原 dev extra 中 langgraph-api>=0.7 与 google-generativeai 的 protobuf 约束冲突，不能合并安装，因此未纳入主环境锁。`archive-extras.in` 保存额外输入。

### 模型和参数需求

当前意图路由直接加载 `~/.omlx/models/Qwen3.5-9B-OptiQ-4bit`（源代码中是原机器的绝对路径），依赖 MLX/Apple Silicon；迁移时需修改 router.py 的 MODEL_PATH。该目录在 legacy 外，本次未清理。训练实验还引用 Qwen3.5-0.8B-OptiQ-4bit / gemma-4-e4b-it-4bit。食物模型为 EfficientNet-B0，分类头 1280→256→5，Dropout 0.3，输入 224×224，输出重量/热量/脂肪/碳水/蛋白质；必须同时保留 `src/langgraph_app/agents/food_recognition/model/best_model.pt` 和 `norm_stats.json`。这些自训练权重、eval 下的训练模型和 LoRA adapters 均保留，不能仅凭模型名重新下载。Gemini 默认 gemini-2.5-flash-lite；其他 provider 配置见 config.py 和 .env.example。

.env 和实际凭据保留在本地，不纳入提交。Python 缓存、工具缓存和可重建环境可删除；业务数据、自训练模型、评测结果及 Git 历史保留。
