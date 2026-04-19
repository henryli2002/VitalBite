# WABI Next.js 重构与技术规划

## 概述

本文档记录 WABI 未来向 Next.js PWA/客户端迁移的技术规划与待办事项。

当前痛点是前端使用 Vanilla JS (app.js)，导致：
1. 消息状态与后端不同步（无法回写消息到数据库）
2. UI 组件逻辑与 Markdown 文本强耦合
3. 每个用户共享同一个 WebSocket 连接，无法做真正的离线推送

---

## Phase 1: 后端数据结构解耦（当前可做）

### 待完成

- [ ] **激活 meal_logs 表**: 当前 `save_meal_log` 已实现但无调用方
  - 位置：`src/server/db.py:377`
  - 任务：改造 `food_recognition_tool.py`，在识别完成后调用 `save_meal_log`
  - 目的：为 AI 提供结构化营养数据源（不从 Markdown 读）

- [ ] **创建 get_today_nutrition 工具**: 供 Goal Planning Agent 查询今日营养
  - DB: `SELECT sum(calories), sum(protein), ... FROM meal_logs WHERE user_id = ... AND timestamp >= today`
  - 用途：当用户问"我今天吃了多少"时，不依赖聊天记录

### 已完成 ✅

- [x] 统一用餐时间判定（早餐 7-9:30, 午餐 11:30-13:30, 晚餐 17:30-19:30）
- [x] 修复 Supervisor UUID 引用错误（强调必须传 32 位 hex UUID）

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