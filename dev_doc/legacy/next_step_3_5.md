### 1. 🔌 从 Mock 数据走向真实业务 (接入真实 API)
- **现状**：目前 `food_recommendation` 节点里使用的是 `_get_restaurants_mock` 假数据。
- **计划**：将推荐逻辑升级为接入真实的餐厅 API（如 Google Places API / Yelp API / 高德地图 API）。我们可以将原本的 `generate_structured` 提取出的位置和口味参数，直接作为参数去调用真实的 HTTP 请求，把真实餐厅返回给用户。

### 2. 🛠️ 引入真正的 Tool Calling (函数调用)
- **现状**：目前的 Agent 还是在依靠我们手写的 Prompt 强行让模型吐出 JSON，然后再解析。
- **计划**：为 LLM Clients 引入原生的 `bind_tools` 支持。LangGraph 对 Tool Node 有完美的支持，这样大模型就可以**自主决定**是否要调用“查询天气”、“查询地图”、“查询食物热量数据库 (如 Edamam API)”的外部函数，Agent 会变得真正智能。

### 3. 📊 重新跑一遍批量评测 (Evaluation & Benchmark)
- **现状**：我注意到您的项目里有一个庞大的 `tests/router_intent/` 目录，里面有各种 `grid_search.py` 和测试用例。
- **计划**：既然我们刚刚修改了 Router 的核心 Prompt，强制它参考多轮多模态上下文，那么这绝对会影响它的准确率。建议运行您的批量测试脚本，配合 LangSmith 看看新版 Prompt 在意图分类（尤其是那些带着之前图片的刁钻测试用例）上，准确率提升了多少，并根据 Benchmark 进一步微调系统 Prompt。

### 4. 💾 持久化记忆与会话管理 (Persistence)
- **现状**：目前在 LangGraph Studio 里测试没问题，但如果要把应用部署出去，需要管理多个用户的长线记忆。
- **计划**：为您的 `build_graph()` 引入 LangGraph 的 `MemorySaver`（如 SqliteSaver 或 PostgresSaver），通过 `thread_id` 或 `user_id` 实现跨设备的长期记忆，让 AI 能记住用户几天前的饮食偏好。(注意要放到数据库里，用postpresql？)

### 5. ⚡ 流式输出体验优化 (Streaming)
- **现状**：目前 `client.generate` 是阻塞的，要等模型彻底想完才吐出完整的一句话，这在处理多模态图片时可能需要等 3-5 秒，UX 体验不够好。
- **计划**：在底层 Client 增加 `generate_stream` 方法，并在 Agent 中返回迭代器，这样前端 UI 就能像 ChatGPT 一样一个个字地打字输出，大幅提升体感响应速度。

您觉得哪个方向最吸引您？如果有倾向，我们可以立刻开始规划和编码！