# WABI Project - 智能健康饮食与营养分析系统

<div align="center">
  <h3>基于 LangGraph 的多智能体 (Multi-Agent) 营养健康管家</h3>
  <p>高性能、微服务化、多模态 AI 饮食管理解决方案</p>
</div>

---

## 📖 项目简介 (Introduction)

**WABI** 是一个先进的、由多个人工智能大语言模型 (LLMs) 驱动的健康饮食与营养分析系统。通过 **LangGraph** 实现复杂的基于状态图的工作流编排，能够精准处理用户的文本指令与多模态图片上传。

系统的核心价值：**精准识别食物营养 (RAG)、个性化餐饮推荐、健康目标规划，并内置强大的安全防御机制。**

---

## 🏗️ 系统架构 (Architecture)

WABI 采用了 **生产级微服务架构**，通过异步任务队列实现高并发处理：

1. **wabi-web (Web 接入层)**: 
   - 基于 **FastAPI** 的高性能网关。
   - 负责 WebSocket 长连接、多用户 Session 管理、身份验证及 PostgreSQL 数据持久化。
   - 将 AI 任务推入 **Redis 任务队列 (wabi_ai_queue)**。
2. **wabi-ai (AI 推理层)**: 
   - **LangGraph Worker 节点**，集群化部署（默认 200 并发 Worker）。
   - 从 Redis 队列监听任务，执行完整的 Agent 编排流水线。
   - 通过 **Redis Pub/Sub** 实时将结果推回给正确的 Web 客户端。
3. **基础设施层**:
   - **PostgreSQL**: 存储多用户聊天历史、个人健康档案及偏好。
   - **Redis**: 核心任务调度中心、语义缓存 (Semantic Cache) 及消息订阅。
   - **FAISS**: 驱动向量数据库，支持 FNDDS 的语义 RAG 检索。

---

## ✨ 核心特性与技术栈 (Features & Tech Stack)

- **核心框架**: `LangGraph` & `LangChain` (0.3.0+)
- **多模型支持**: 统一接口调用 `Google Gemini` (多模态首选)、`OpenAI`、`AWS Bedrock`。
- **视觉分析 (MLLM)**: 
  - **高保真多模态处理**: 已修复多模态图像在微服务传输中的损耗问题。
  - **重量估算**: 自动识别食物并估算克数。
- **高级 RAG**: 基于 `FAISS` 和 `sentence-transformers` 实现 14000+ FNDDS 营养数据的毫秒级检索。
- **安全性**: 内置 `Guardrails` 节点，针对 Prompt Injection 及违规内容进行双重防御。

---

## 🤖 智能体集群 (Agent Swarm)

- **Food Recognition Agent**: MLLM 识别图片 + FNDDS RAG 检索 + 营养综合计算。
- **Recommendation Agent**: 结合 **IP 地理位置** 和 Google Maps API 提供个性化健康餐饮。
- **Goal Planning Agent**: 结合历史对话，制定长效饮食干预与健康目标。
- **Tutorial & Chitchat**: 提供系统教程引导与自然语言交互。

---

## 🚀 快速开始 (Quick Start)

项目已全面容器化，推荐使用 Docker 进行一键部署。

### 1. 环境配置
将 `.env.example` 复制为 `.env`：
```env
GOOGLE_API_KEY="AIza..."
GOOGLE_MAPS_API_KEY="..."
WABI_REDIS_URL="redis://wabi-redis:6379/0"
WABI_DB_URL="postgresql://wabi_user:wabi_password@wabi-postgres:5432/wabi_chat"
```

### 2. 构建并运行 (Docker Compose)
```bash
docker-compose up -d --build
```
启动后访问：
- **前端 Web 界面**: `http://localhost:8000`
- **AI Worker 监控**: `http://localhost:8001/health`

---

## 🛠️ 运维与工具 (Maintenance)

### 1. 数据库清理 (Purging Test Data)
如果系统中存在负载测试留下的冗余数据，可以使用我们内置的维护脚本：
```bash
# 物理删除 ID 以 'loadtest_' 开头的所有测试账号及消息
docker exec wabi-web python3 scripts/cleanup_test_data.py
```
*注：系统已在 API 层级自动过滤测试账号，保证生产界面清洁。*

### 2. 构建向量库
第一次部署需处理 FNDDS 数据集：
```bash
docker exec wabi-ai python3 scripts/build_vector_store.py
```

---

## 📁 目录结构 (Structure)

```text
WABI/
├── docker/                  # Docker 配置
│   ├── Dockerfile.ai        # AI Worker 镜像
│   └── Dockerfile.web       # Web 服务镜像
├── backend/
│   ├── langgraph_app/       # LangGraph 核心逻辑
│   │   ├── agents/          # Agent 实现 (food_recognition, recommendation, goalplanning, etc.)
│   │   ├── orchestrator/    # 工作流编排
│   │   ├── tools/           # 工具集 (map, nutrition, vision)
│   │   ├── utils/           # 工具函数
│   │   └── config.py        # 配置管理
│   ├── langgraph_server.py  # AI Worker 入口 (Consumer)
│   ├── web_server.py        # FastAPI API 入口 (Producer)
│   ├── chat_manager.py      # 会话流控与 Redis 调度
│   ├── db.py                # PostgreSQL 存储适配层
│   └── scripts/             # 运维脚本 (清理数据、向量库构建)
├── frontend/                # 前端页面
├── eval/                    # 模型评估框架
└── tests/                   # 测试框架
```

---

## 📈 最近更新 (Recent Updates)
- **v2.1.0 (2026-03)**:
    - ✅ **架构升级**: 从 SQLite 迁移至 PostgreSQL + Redis 异步微服务架构。
    - ✅ **多模态增强**: 修复了图像 Base64 数据在 Redis 传输中的丢失问题，显著提升识别准确率。
    - ✅ **数据隐私**: 清理了 3900+ 冗余测试账号，并增加了生产环境数据隔离过滤。