# WABI Next.js 重构与技术规划

## 概述

本文档记录 WABI 未来向 Next.js PWA/客户端迁移的技术规划与待办事项。

当前痛点是前端使用 Vanilla JS (app.js)，导致：
1. 消息状态与后端不同步（无法回写消息到数据库）
2. UI 组件逻辑与 Markdown 文本强耦合
3. 每个用户共享同一个 WebSocket 连接，无法做真正的离线推送

---

## Phase 0: 前端体验与接入（优先于 Phase 1）

这两项排在 Phase 1 之前。Phase 1 的"今日摄入"依赖 0.1 把每一项的 multiplier 写回数据库；0.2 则把当前的 admin 面板和用户端分离，为后续任何"用户自己打开就能用"的场景打底。

### 0.1 食物份额滑块（先做，~1–2 天，难度：低）

**需求**
- nutrition-card 每一项加一个 50% – 150% 的横向滑条，默认 100%，与 `weight_g` 绑定
- 实时拖动 → 前端本地重新计算并渲染 cal / mass / macro（不等后端）
- 防抖 ~500ms → PATCH 回写到 `meal_logs`，只传百分比，不传重算后的数值
- 未来 LLM 汇总（`get_today_nutrition`）读 `multiplier` 计算"真实摄入"

**落点**
- 前端：复用现有 `buildNutritionViz`，每个 `nutrition-card` 底部加 `<input type="range">` + debounce；总览卡 (`total-card`) 与饼图/柱图跟着滑块重算
- 后端：新增 `PATCH /api/meal_logs/{id}` 接受 `{items: [{idx, multiplier}, ...]}`，写入 `meal_logs.items_json`
- DB：`items_json` 每一项扩展 `"multiplier": 1.0`（缺省视为 1.0，向后兼容）

**依赖**
- 必须同时激活 `save_meal_log`（`food_recognition_tool.py` 识别完调用一次），否则没有 `meal_log_id` 可 PATCH。原 Phase 1 的第一项顺手落在这里。

**风险点**
- 滑块是 per-item，但营养卡里的饼图/柱图/总览是聚合的 —— 重算逻辑别遗漏
- 乐观更新：若 PATCH 失败要回滚本地状态并提示（不要静默偏差）

### 0.2 前端拆分 + 用户登录（后做，~2–3 天，难度：中）

**需求**
- 现有前端保留为**控制面板**（admin），移到 `/admin`
- 新增**用户端**前端：只保留右侧聊天栏，入口 `/`，无用户切换控件
- 登录：`username` 唯一，`password` 允许为空（测试期占位，后续上 argon2/bcrypt）
- 登录后建立 WebSocket，`user_id` 从 session cookie 解析，禁止 URL 里冒名他人

**落点**
- DB（迁移）：
  - `users` 加 `username TEXT UNIQUE NOT NULL`、`password_hash TEXT NULL`
  - 老数据：以 `user_id` 作默认 `username`，`password_hash = NULL`
- 后端：
  - `POST /api/auth/signup` / `POST /api/auth/login` → 种 HttpOnly session cookie（走 `itsdangerous` 签名串，不引入 JWT）
  - `/ws/{user_id}` 握手时校验 cookie 里的 user_id 与 path 一致
- 前端：
  - 新 `chat.html` + `chat.js`：从 `app.js` 抽出聊天渲染部分（`updateThinkingIndicator` / `appendMessage` / 营养卡 / 餐厅卡 / WebSocket 生命周期）
  - 拆分策略：先抽成 `chat-core.js` 模块，admin 和用户端都 import，短期可以保留少量重复代码，后续再收敛

**依赖**
- 无强依赖 0.1，但强烈建议 0.1 先落地：如果用户端先上线再改滑块，交互抖动暴露面太大。

**风险点**
- Session cookie 要指定 `SameSite=Lax` + `Secure`（生产），本地开发放宽
- WebSocket 握手鉴权：FastAPI 下需要在 `websocket.accept()` 前读 cookie
- 老 user_id 形式（由 admin 面板生成的字符串）与新 username 的映射要做幂等迁移脚本

---

## Phase 1: 后端数据结构解耦

### 待完成

- [ ] **激活 meal_logs 表**: 当前 `save_meal_log` 已实现但无调用方
  - 位置：`src/server/db.py`（`save_meal_log`）
  - 任务：改造 `food_recognition_tool.py`，识别完成后调用 `save_meal_log`，返回 `meal_log_id` 一起给前端
  - 与 0.1 一起做（滑块 PATCH 需要这个 id）

- [ ] **创建 get_today_nutrition 工具**: 供 Supervisor 查询今日营养
  - DB: `SELECT sum(calories * multiplier), sum(protein * multiplier), ... FROM meal_logs WHERE user_id = ... AND timestamp >= today`
  - 用途：当用户问"我今天吃了多少"时，不依赖聊天记录；必须套用 0.1 的 multiplier

### 已完成 ✅

- [x] 统一用餐时间判定（早餐 7-9:30, 午餐 11:30-13:30, 晚餐 17:30-19:30）
- [x] 修复 Supervisor UUID 引用错误（强调必须传 32 位 hex UUID）
- [x] Supervisor 流式 thinking：react loop 内每个 tool call / result 推送细粒度思考步
- [x] Supervisor 超时与递归上限：`SUPERVISOR_TOTAL_TIMEOUT_S` + `MAX_TOOL_CALLS_PER_TURN`
- [x] Transport 解耦：`publish_thinking` 通过 `RunnableConfig.configurable` 注入，`supervisor_state.py` 不再携带 `response_channel`
- [x] Supervisor prompt 的"FRESHNESS PRINCIPLE"：禁止复述上轮工具输出，每次命中工具语义必须重新调用
- [x] 思考面板最后一步（`supervisor_reply`）保留展示，不再自动删除

---

## Phase 2: Next.js 客户端重构（未来）

### 目标架构

```
┌─────────────┐     ┌─────────────┐
│ Next.js    │────▶│ API Routes │
│ PWA/小程序  │     │ (FastAPI)  │
└─────────────┘     └─────────────┘
      │                   │
      ▼                   ▼
┌─────────────┐     ┌─────────────┘
│ React      │◀────
│ Component │
└─────────────┘
```

### 关键变化

| 当前 (Vanilla JS) | 未来 (Next.js) |
|----------------|--------------|
| 所有用户共享 WebSocket | 每个用户独立连接/离线推送 |
| Markdown 代码块解析 UI | 结构化 JSON 消息 API |
| 营养卡片滑块无法回写 DB | React 状态 + Debounce PATCH API |
| 硬编码 JS 组件 | `<NutritionCard multipler={0.8} />` |

### 消息协议设计（Draft）

```json
{
  "type": "nutrition_card",
  "meal_log_id": 123,
  "items": [
    {"name": "虾仁豌豆", "weight_g": 150, "calories": 200, ...}
  ],
  "multiplier": 1.0,
  "created_at": "2026-04-19T12:00:00Z"
}
```

### API 设计（Draft）

| 方法 | 路径 | 用途 |
|-----|------|------|
| GET | `/api/chat/history` | 加载最近 N 条消息 |
| POST | `/api/chat/message` | 发送新消息 |
| PATCH | `/api/meal_logs/{id}` | 调整营养 multiplier |
| GET | `/api/nutrition/today` | 获取今日营养汇总 |

### 食物滑块交互（React 设计）

```jsx
function NutritionCard({ items, multiplier, onChange }) {
  const [localMult, setLocalMult] = useState(multiplier);

  // 防抖回写
  useDebouncedCallback((val) => {
    fetch(`/api/meal_logs/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ multiplier: val })
    });
  }, 500);

  return (
    <input
      type="range"
      min="0.5" max="1.5" step="0.05"
      value={localMult}
      onChange={(e) => setLocalMult(e.target.value)}
    />
  );
}
```

---

## 技术债务（当前可处理）

### 1. 图片 UUID 引用问题

**问题**: Supervisor 在多轮对话中可能把 `<attached_image description=...>` 的描述文本当作 UUID 传入工具。

**修复**: 已在 supervisor.py 中强调 UUID 为 32 位 hex。

**根本方案**: 消息协议改用 JSON 而非 Markdown 标记。

### 2. 裁剪图像预处理

**问题**: 底层 fine-tuned 模型会把裁剪后的小图强制拉伸到 224x224，导致估算过大。

**修复**: Letterbox 居中 + 白边填充（已完成 ✅）

- 位置：`src/langgraph_app/tools/food_recognition_tool.py:319`
- 逻辑：以裁剪图长边为基准，创建白色正方形，居中贴图后送模型

### 3. 意图分类失效

**问题**: `USE_SUPERVISOR=1` 模式下，intent_router_node 已不再被调用。

**现状**: Supervisor 自己决定调用什么工具，不需要提前分类 intent。

**保留**: Legacy 代码保留在 `graph.py:_create_legacy_graph()` 中，供回滚使用。

---

## 待讨论

1. 是否需要离线推送？（Firebase / APNs / 极光）
2. meal_logs 是否需要关联 messages 表？（当前独立）
3. 用户 Profile 是否需要迁移到独立表？（目前在 users.profile_json）