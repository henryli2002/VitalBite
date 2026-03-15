我理解了您的需求，将识别流程更新为“分割工具+数据库工具+LLM整理”的最高准确率方案，并为单图和双图（同一物体的不同视角）设置不同的处理逻辑。

这是一个分解计划，旨在实现您所描述的工作流程。

### 计划概述

我们将分三个阶段进行：
1.  **搭建新工具和节点**: 创建实现新识别流程所需的核心工具和新的 Agent 节点。
2.  **更新路由和图逻辑**: 修改现有的路由逻辑，使其能够根据图片输入的类型（单图、双图等）将任务分发到正确的处理节点。
3.  **审阅和切换**: 确认计划无误后，切换到编码模式开始执行。

### 工作流程图

我为您绘制了新的工作流程图，以更直观地展示逻辑：

```mermaid
graph TD
    A[用户上传图片] --> B{Router LLM 判断图片};
    B --> C1[单张/多张不相关食物图片];
    B --> C2[两张同一物体的不同视角图片];
    B --> C3[其他意图 Chitchat/Recommendation...];

    C1 --> D[Accurate Recognition Node (新)];
    D --> E{1. LLM 分析图片生成描述};
    E --> F[2. RAG从FNDDS数据库检索];
    F --> G[3. 估算分量并计算营养];
    G --> H[4. 烹饪方法修正];
    H --> I[生成最终结果];

    C2 --> J[Spatial Measurement Node (新)];
    J --> K[1. 调用空间测量工具];
    K --> L[2. 估算体积和重量];
    L --> D;

    C3 --> M[现有流程];
    I --> Z[结束];
    M --> Z;
```

### 任务清单

- [ ] **阶段一：搭建新工具和节点**
    - [ ] 1. 创建 FNDDS 数据库检索工具 (`langgraph_app/tools/nutrition/fndds.py`)
    - [ ] 2. 创建空间测量工具的占位符 (`langgraph_app/tools/vision/spatial.py`)
    - [ ] 3. 创建新的食物识别 RAG Agent 节点 (`langgraph_app/agents/food_recognition/rag_agent.py`)
    - [ ] 4. 创建新的空间测量 Agent 节点 (`langgraph_app/agents/food_recognition/spatial_agent.py`)

- [ ] **阶段二：更新路由和图逻辑**
    - [ ] 1. 修改 `router.py` 中的 `IntentAnalysis`，增加新的 intent
    - [ ] 2. 更新 `router.py` 中的 LLM prompt，使其能够识别单图、双图（空间测量）和其他情况
    - [ ] 3. 修改 `graph.py`，注册新的 `accurate_recognition` 和 `spatial_measurement` 节点
    - [ ] 4. 修改 `graph.py` 中的 `route_by_intent`，增加到新节点的条件路由

- [ ] **阶段三：审阅和切换**
    - [ ] 1. 审阅整个计划和代码结构
    - [ ] 2. 请求切换到 `💻 Code` 模式进行实现

请问您对这个计划满意吗？如果可以，我们将切换到“编码”模式来实施这些更改。
