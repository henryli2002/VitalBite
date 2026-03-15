# WABI Project - 智能健康饮食与营养分析系统

<div align="center">
  <h3>基于 LangGraph 的多智能体 (Multi-Agent) 营养健康管家</h3>
</div>

---

## 📖 项目简介 (Introduction)

**WABI** 是一个先进的、由多个人工智能大语言模型 (LLMs) 驱动的健康饮食与营养分析综合系统。
它利用 **LangGraph** 实现了复杂的基于状态图（StateGraph）的工作流编排，能够将用户的自然语言指令、图片上传等复杂意图进行安全检查、精准路由，并交由最专业的智能体（Agent）进行处理。

系统的核心价值在于：**精准识别食物营养、提供个性化餐饮推荐、规划健康目标，并在整个交互过程中保证极高的安全性。**

## ✨ 核心特性与技术栈 (Features & Tech Stack)

- **核心编排框架**: `LangGraph` & `LangChain` (版本 >= 0.3.0)
- **多模型支持 (LLM Factory)**: 统一接口无缝切换 `Google Gemini`、`OpenAI (GPT-4o等)`、`Anthropic Claude (via AWS Bedrock)`。
- **高级 RAG 技术**: 使用 `FAISS` 向量数据库和 `sentence-transformers` 进行 FNDDS (美国农业部食品与营养数据库) 的语义检索。
- **多模态视觉分析 (MLLM)**: 支持单图直接估算重量、双图空间测量等复杂的视觉任务。
- **防御机制**: 内置强大的防提示词注入 (Prompt Injection) 和内容安全护栏 (Guardrails)。

---

## 🏗️ 系统架构与工作流 (Architecture & Workflow)

整个系统由 `langgraph_app/orchestrator/graph.py` 定义为一个状态机。请求的流转路径如下：

1. **Input Guardrail Node (输入安全节点)**: 
   - 所有的用户输入（文本/图片）首先经过此处，通过 LLM 判定是否包含仇恨、暴力、非法、自残或色情内容。
2. **Intent Router Node (意图路由节点)**: 
   - 系统的“大脑”。除了检测“提示词注入攻击（如：忽略之前的指令）”外，它将用户的请求精准分类为六大意图之一。
3. **Agent Nodes (智能体处理集群)**: 
   - 根据 Router 的结果，请求被分发给特定的专业 Agent 处理（详见下一节）。
4. **Output Guardrail Node (输出安全节点)**: 
   - 在将最终结果返回给用户前，再次进行安全校验，确保系统输出合规且安全。

---

## 🤖 核心智能体集群 (Agent Swarm)

WABI 系统集成了以下高度专业化的智能体：

### 1. 🥗 Food Recognition Agent (食物识别与营养 RAG 智能体)
这是系统中最复杂的模块之一。它的工作流如下：
- **视觉分析与重量估算**: MLLM（多模态大模型）分析用户上传的美食图片，识别出盘子中的所有独立食物成分，并直接估算它们的重量（克）。支持通过单图估算或多图（同一物体的不同视角）进行空间测量。
- **FNDDS 数据库检索 (RAG)**: 将识别出的每种食物转换为查询条件，通过 `FAISS` 向量数据库在 FNDDS (Food and Nutrient Database for Dietary Studies) 中进行高维语义检索，找到最匹配的权威营养数据。
- **营养综合计算**: LLM 根据估算的重量和检索到的每 100g 营养数据，计算出整顿餐食的总热量、蛋白质、脂肪、碳水等，并生成具有营养专家口吻的最终分析报告。

### 2. 🗺️ Food Recommendation Agent (健康饮食推荐智能体)
当用户询问“附近有什么健康的素食”或“推荐低脂餐厅”时触发。
- **地理位置感知**: 结合用户的 IP 地址（通过 `ip_location` 工具）或用户提供的经纬度。
- **Google Maps 搜索**: 调用 `search_restaurants` 工具，在 Google Maps 上基于位置、菜系、半径搜索合适的餐厅。
- **个性化筛选**: LLM 会根据用户的饮食偏好和健康目标，对搜索结果进行二次过滤和推荐。

### 3. 🎯 Goal Planning Agent (目标规划智能体)
协助用户制定长期的饮食干预、减肥、增肌或其他营养健康目标计划，并提供步骤化的指导。

### 4. 📚 Tutorial Agent (系统教程智能体)
作为系统的使用向导，向用户解释如何上传图片、如何提问、系统能提供哪些功能等。

### 5. 💬 Chitchat Agent (日常闲聊智能体)
处理与核心健康任务无关的日常寒暄、问候等，提供自然、有温度的对话体验。

### 6. 🛡️ Guardrails Agent (安全拦截智能体)
当 Input/Output Guardrail 检测到违规内容，或者 Router 检测到越权攻击时，该智能体会接管对话，礼貌且坚定地拒绝响应并给出解释。

---

## 📁 详细项目目录结构 (Directory Structure)

```text
WABI/
├── langgraph_app/              # 核心应用源码目录
│   ├── agents/                 # 核心智能体集群 (识别、推荐、规划、闲聊等)
│   │   ├── food_recognition/   # 包含 rag_agent.py 等高阶识别逻辑
│   │   ├── food_recommendation/# 包含基于地理位置的推荐逻辑
│   │   └── ...                 
│   ├── orchestrator/           # LangGraph 工作流编排
│   │   ├── graph.py            # 定义节点、边和条件路由的主图
│   │   ├── state.py            # GraphState 状态定义
│   │   └── nodes/              # Router 和 Guardrail 的具体实现
│   ├── tools/                  # 智能体可调用的外部工具库
│   │   ├── map/                # Google Maps API 封装和 IP 定位
│   │   ├── nutrition/          # FNDDS 数据处理、FAISS 检索工具 (fndds.py)
│   │   └── vision/             # 空间/单图视觉测量工具
│   └── utils/                  # 基础架构工具
│       └── llm_factory.py      # LLM 客户端工厂模式，统一多模型调用
├── scripts/                    # 运维与构建脚本
│   ├── build_vector_store.py   # 用于将 FNDDS 数据构建为 FAISS 向量库的核心脚本
│   └── inspect_xlsx_keys.py    # 数据检查脚本
├── tests/                      # 丰富的自动化测试目录
│   └── router_intent/          # 针对 Router 意图识别和防注入的大规模测试集
├── pyproject.toml              # 项目包依赖管理和打包配置
├── FNDDS_GUIDE.md              # 美国农业部营养数据库的使用说明
└── plan.md                     # 项目开发路线图与历史演进记录
```

---

## 🚀 部署与运行指南 (Deployment & Usage)

### 1. 环境准备 (Prerequisites)
- Python >= 3.10
- 根据需求准备 API Keys：OpenAI, Google Gemini, Anthropic (AWS Bedrock), Google Maps API 等。

### 2. 安装依赖 (Installation)
系统使用 `pyproject.toml` 管理依赖项。在项目根目录下运行：
```bash
pip install -e .
# 或者安装开发依赖
pip install -e ".[dev]"
```

### 3. 配置环境变量 (Configuration)
将项目根目录的 `.env.example` 复制为 `.env`，并填充你的密钥：
```env
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
GOOGLE_MAPS_API_KEY="..."
# 等等...
```

### 4. 构建向量数据库 (Build Vector Store)
在第一次使用 **食物识别 (Food Recognition)** 功能前，需要处理本地的 FNDDS 营养数据集并构建 `FAISS` 索引：
```bash
python scripts/build_vector_store.py
```
这会在 `langgraph_app/tools/nutrition/` 目录下生成 `fndds_index.faiss` 和 `fndds_documents.json` 文件。

### 5. 启动应用 (Run)
可以通过多种方式运行该基于 LangGraph 的应用。你可以直接运行带有图测试的脚本，或使用 `langgraph-cli` 等进行服务器部署：
```bash
# 视具体的入口文件而定
```

---

## 📈 后续规划 (Roadmap)
当前的开发主要聚焦于优化多模态意图的解析准确率，以及通过双图比对进行更加精准的三维空间体积与重量估算。详细计划请参考 `plan.md`。