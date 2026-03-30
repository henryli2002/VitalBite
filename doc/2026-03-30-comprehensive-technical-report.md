# WABI 全面项目技术报告

**报告日期：** 2026-03-30

本报告全面梳理了 WABI（基于大语言模型与多模态能力的智能饮食助手）项目的技术架构与实现细节。项目采用了彻底的前后端分离设计，依托分布式的“网关层 - 消息队列 - AI计算节点”架构，实现了极高的并发处理能力与系统稳定性。以下为各个模块及底层实现的技术剖析。

---

## 1. 系统宏观架构与高并发设计（异步的核心作用）

项目核心架构为一个异步的分布式微服务系统，通过 Docker Compose 编排。**异步编程（`asyncio`）贯穿了项目的整个生命周期，其核心作用是打破 I/O 阻塞，使得单线程/单进程能够同时处理成千上万个并发请求，而不需要为每个请求分配昂贵的系统线程。**

### 1.1 异步在系统各环节的具体应用
- **网关层 (Web Server & WebSocket):** 采用 `FastAPI` + `Uvicorn`。用户的 WebSocket 长连接在等待消息时完全非阻塞（`await websocket.receive_json()`）。单台服务器可以轻松维持数万并发连接。
- **数据库读写 (PostgreSQL):** 使用了 `asyncpg`（基于 Python asyncio 的高性能异步驱动）。当查询或写入数据时（`await conn.execute(...)`），当前协程会让出控制权，程序可以去处理其他用户的聊天请求。
- **消息中间件 (Redis):** AI 计算节点通过 `await redis_client.blpop()` 异步阻塞式监听队列。当前队列无任务时，协程挂起；一旦有任务，立刻唤醒执行。
- **计算密集型任务的异步化 (FAISS 检索):** FAISS 的向量计算本身是阻塞的 CPU 密集型任务。系统在 `fndds.py` 中通过 `ThreadPoolExecutor` 配合 `await loop.run_in_executor(...)` 将同步操作封装为异步，避免了卡死主事件循环。
- **AI Worker 节点:** 作为独立的计算引擎（`wabi-ai` 容器），启动时直接拉起多达 200 个异步 Worker 协程（`MAX_CONCURRENT_WORKERS = 200`）。200个协程跑在单一进程内，同时并发处理不同的 AI 任务。

---

## 2. 各模块技术栈与核心库剖析

### 2.1 数据库与持久化层 (Database & Persistence)
- **技术选型:** PostgreSQL 15 配合 `asyncpg`（纯 Python 实现的超高性能异步 PostgreSQL 驱动）。
- **连接池管理:** 通过 `asyncpg.create_pool` 构建连接池（`min_size=1, max_size=50`），支持高并发下的数据库事务复用。
- **核心方法:**
    - 事务操作: `async with pool.acquire() as conn:`, `async with conn.transaction():`
    - 数据读写: `conn.execute()`（写入、更新）, `conn.fetch()`（批量查询）, `conn.fetchrow()`（单行查询）, `conn.fetchval()`（单值查询）。
- **数据结构:** 包含 `users` 表（存储用户信息及 JSON 格式的 profile）、`messages` 表（存储对话记录，含角色、内容与时间戳）。

### 2.2 服务网关层 (Backend Web Server)
- **技术选型:** `FastAPI`, `websockets`, `pydantic`.
- **核心职责:**
    - RESTful API：提供 `/api/users/*` 接口用于创建用户、查询历史、更新个人档案等。
    - WebSocket 接口：`/ws/{user_id}` 维持实时双向通信。
- **关键方法:**
    - `websocket.accept()` / `websocket.receive_json()` / `websocket.send_json()`。
    - Pydantic 模型校验请求入参，如 `UserCreate`, `UserProfile`, `WSIncoming`。

### 2.3 状态管理与大模型中间层 (Agent Orchestrator)
- **技术选型:** `langgraph`, `langchain`, `langchain-core`, `langchain-google-genai`, `openai`, `boto3` (AWS Bedrock)。
- **LLM Factory 缓存机制:** 为了避免高并发下频繁的 TLS 握手，系统设计了单例模型缓存（`_LLM_CACHE`）。根据 `(provider, model_name, module)` 对 LangChain 的 Chat Model 进行缓存复用。
- **配置与采样:** `config.py` 根据不同意图节点（如 router, clarification, food_recognition）动态配置温度值（temperature）、top_p 等采样参数（例如识别任务要求 deterministic，温度设为 0.0）。

---

## 3. LangGraph AI Agent 架构解析 (图实例复用机制)

Agent 核心逻辑由 LangGraph 的状态机（StateGraph）驱动。**需要特别说明的是：并不是每个并发请求都会实体化（重新实例化）一个新的图对象。** 
在系统启动时（模块加载阶段），`graph.py` 中的 `create_graph()` 会在全局作用域内被调用一次，将各级 Nodes 与 Router 绑定并执行 `compile()`。此后，所有的异步请求（如 `graph.ainvoke(...)`）都在**共享使用这同一个全局编译后的图实例（Singleton）**。由于 LangChain/LangGraph 对异步的支持，同一份图实例可以安全地并发处理不同任务状态，完全消除了反复编译有向图和初始化节点的高昂性能成本。

### 3.1 状态管理 (GraphState)
使用 `TypedDict` 定义图中的全局状态流转，包括上下文信息（`user_id`, `session_id`, `user_profile`）、分析结果（`analysis`）、业务数据（`recognition_result`, `recommendation_result`）、以及 LangChain 消息队列（`messages`）。通过 `Annotated` 与 `operator.add` 实现诸如 `debug_logs` 与 `messages` 的追加合并。

### 3.2 节点与路由 (Nodes & Router)
图（Graph）的执行流如下：
1. **`input_guardrail_node`**: 输入安全检查。使用条件边 `should_continue`，如果 `unsafe` 则直接熔断至 `END` 节点。
2. **`intent_router_node`**: 意图识别节点。
    - 结合当前时间（判断是否是用餐时间）与用户画像（`user_profile`）构建 System Prompt。
    - 调用 LLM 的结构化输出能力 `client.with_structured_output(IntentAnalysis)`，解析用户的真实意图（识别、推荐、闲聊、教学或目标规划）。
    - 采用带重试机制的设计（最多 3 次），并通过 `with_semaphore("intent")` 控制速率限制。
3. **功能性 Nodes**:
    - `recognition_node`: 食物图像识别。
    - `food_recommendation_node`: 基于地理位置的餐饮推荐。
    - `goalplanning_node`: 健康规划（如果是该意图，Web 网关会发起二次拦截并携带全量历史消息入队列重试）。
    - `chitchat_node` / `tutorial_node`: 兜底对话。
4. **`output_guardrail_node`**: 输出安全审核。同样包含熔断机制。

### 3.3 边缘计算与工具集成 (Tools & Integrations)
- **图像与多模态 (`Pillow`):** 处理前端传入的 Base64 图像，解析多模态消息（由文本与 `image_url` 组成）。
- **LBS 与地图 (`requests`, `google_maps`, `ip_location`):** 调用第三方 API。在 AI Worker 启动时，执行 `get_location_from_ip_async()` 预热 IP 缓存池以削减首次请求延迟。
- **向量数据库检索与 FAISS 实例化:** 用于 FNDDS（美国农业部营养数据库）相似度召回。
    - **单次实例化机制:** 与图对象类似，**在 `fndds.py` 模块初次加载时，SentenceTransformer 词嵌入模型（`all-MiniLM-L6-v2`）、`faiss.read_index(...)` 以及 JSON 元数据均会在全局仅初始化加载一次并常驻内存中。并不是每次请求都重新加载。**
    - **避免阻塞主事件循环:** `fndds.py` 内设了独立的线程池（`ThreadPoolExecutor(max_workers=8)`），由于 FAISS 的高维检索是纯 CPU 密集型任务，系统利用 `loop.run_in_executor()` 将这个阻塞任务丢进子线程执行，随后抛给主程序的 asyncio，避免了向量计算引发的全局卡顿。

---

## 4. 关键第三方库与组件详单

| 分类 | 库/组件名称 | 核心应用场景与方法 |
| :--- | :--- | :--- |
| **异步网关** | `fastapi`, `uvicorn` | 构建非阻塞 API 网关及应用挂载 (`FastAPI()`, `uvicorn.run()`) |
| **持久层** | `asyncpg` | 异步连接池操作 PostgreSQL (`asyncpg.create_pool()`, `pool.acquire()`) |
| **消息队列** | `redis.asyncio` | Pub/Sub 订阅发布，分布式任务队列 (`blpop`, `rpush`, `publish`, `subscribe`) |
| **Agent编排** | `langgraph` | 状态机组装 (`StateGraph()`, `add_node()`, `add_conditional_edges()`, `compile()`) |
| **LLM 框架** | `langchain_core`, `langchain_google_genai` | 消息实体化 (`HumanMessage`, `AIMessage`), 模型调用与结构化提取 (`with_structured_output`) |
| **视觉与多模态** | `pillow` | 读取、压缩、解析前端上报的图像数据格式 |
| **检索与科学计算**| `faiss-cpu`, `sentence-transformers`, `numpy` | 构建本地离线的高性能向量检索引擎，加速营养数据库的映射 |
| **数据校验** | `pydantic` | 提供接口入参验证与 LLM 输出结构的强约束 (`BaseModel`) |

## 5. 总结

WABI 系统在架构设计上彻底践行了异步非阻塞与微服务解耦理念：
1. **利用 Redis 桥接了 IO 密集型的前端网关与计算密集型的 AI 节点**，既避免了长连接直接阻塞模型推理，又为横向扩展 AI Worker 节点提供了极大的弹性。
2. **LangGraph 的精准控制** 确保了大语言模型在执行多步骤推理时（尤其是包含多模态图像处理、RAG检索、API外部调用）的状态清晰无状态泄露，配合细粒度的温度采样配置，保证了对话的逻辑连贯与营养识别的确定性。
3. **基于连接池（asyncpg, LLM Client Cache）的极致优化** 极大地降低了系统高并发环境下的握手及寻址开销。