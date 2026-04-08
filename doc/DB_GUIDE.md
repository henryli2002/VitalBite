# WABI 工业级数据库与缓存架构指南 (DB & Caching Guide)

本指南详细记录了 WABI 系统在经历“核弹级”高并发压测后，从单机原型（SQLite）全面进化到**工业级分布式数据栈（PostgreSQL + Redis）**的完整设计思想、表结构以及代码示例。

---

## 1. 架构选型：为什么抛弃 SQLite？

在早期的开发中，由于我们需要快速 MVP，所以采用了 `aiosqlite` 搭配本地 `.db` 文件。但在真正的工业多用户场景（例如 150 人同时发几百条聊天记录）下，它暴露出了致命弱点：

1. **并发锁死 (Database is Locked)**：SQLite 只要有一个人在执行写入（Write），整个库就会**全局锁死**。150个人并发聊天时，后方队列疯狂撞击导致 SQLite 直接崩溃罢工。
2. **约束松散 (Loose Constraints)**：SQLite 默认是不强制检查外键依赖的（这也是压测早期出现幽灵用户的原罪）。
3. **微服务隔离**：Docker 容器架构下，把本地文件在容器之间挂载传递极其危险且容易损坏。

**✅ 解决方案**：引入企业级关系型数据库 **PostgreSQL 15**，结合 `asyncpg` 异步高并发连接池；同时引入内存级数据库 **Redis 7** 作为 API 阻击手和限流中枢。

---

## 2. PostgreSQL：结构化主库 (Persistence Layer)

PostgreSQL 主要负责一切**不可丢失**的用户资产档案：也就是用户注册信息、基础健康档案（Profile）、以及多模态长文本聊天大纲（Messages）。

### 2.1 数据库结构 (Schema)

核心表有两张，它们之间被强外键（Foreign Key）死死锚定：

```sql
-- 1. 用户档案表 (Users)
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    profile_json TEXT DEFAULT '{}'
);

-- 2. 聊天记录表 (Messages)
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE, -- 强关联外键！
    role TEXT NOT NULL,            -- 'user' | 'assistant'
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    has_image INTEGER DEFAULT 0
);

CREATE INDEX idx_messages_user_ts ON messages(user_id, timestamp);
```

### 2.2 Python CRUD 代码示范 (使用 asyncpg)

我们在 `db.py` 中封装了 `PostgresHistoryStore`。最大的变化是引入了**异步连接池 (Connection Pool)**，极大减少了 TCP 握手开销。

#### [Create] 增：如何处理高并发并发涌入的新老用户？
这就是你压测中途那次报错的原因！工业上，为了防止高并发下多个请求同时尝试创建同一个用户引发主键冲突，我们必须使出杀手锏：`ON CONFLICT ... DO UPDATE` (也叫 Upsert)。

```python
async def save_message(self, user_id: str, role: str, content: str, timestamp: str) -> None:
    pool = await self._get_pool()
    async with pool.acquire() as conn:          # 从池里抓一个空闲连接
        async with conn.transaction():          # 开启原子事务
            
            # 第一步：尝试强行注册这个用户。如果他已经注册过了（主键冲突），那就顺手更新一下活跃时间！
            await conn.execute(
                """
                INSERT INTO users (user_id, name, created_at, last_active, profile_json)
                VALUES ($1, $2, $3, $4, '{}')
                ON CONFLICT (user_id) DO UPDATE SET last_active = EXCLUDED.last_active
                """,
                user_id, f"User {user_id[-4:]}", timestamp, timestamp
            )
            
            # 第二步：插入聊天记录（此时第一步已经100%保证了 users 表里有这个人，绝对不会再报外键错）
            await conn.execute(
                "INSERT INTO messages (user_id, role, content, timestamp) VALUES ($1, $2, $3, $4)",
                user_id, role, content, timestamp
            )
```

#### [Read] 查：拉取上下历文
使用 `$1`, `$2` 作为占位符，这比 SQLite 的 `?` 更安全，能直接防御所有 SQL 注入。
```python
async def load_history(self, user_id: str):
    pool = await self._get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM messages WHERE user_id = $1 ORDER BY id ASC",
            user_id
        )
        return [dict(row) for row in rows]
```

---

## 3. Redis：救命的极速缓冲与消息队列

如果说 PostgreSQL 是承载数据的重型航母，那 Redis 就是光速穿梭在各个微服务之间的导弹驱逐舰。在这个项目里，我给 Redis 赋予了**两个救命的职责**。

### 3.1 职责一：Google API 的物理护盾 (Cache Layer)

在 150 人同时说“帮我推荐旁边好吃的餐厅”，如果没有 Redis，150个 HTTP 请求会同时砸给 Google Maps API，你的信用卡会被扣瞎，而且会等接近一分多钟。

**实现思路**：
我们在 `google_maps.py` 给 Google 套了一件拦截战衣：
1. **生成 MD5 Hash**：把用户的要求 `"日餐" + 纬度 + 经度 + "2公里"` 做成一个极难冲突的短字符串哈希，比如 `maps_cache_a1b2c3d4`。
2. **拦截**：在发给谷歌前，用 `redis.get()` 扫一眼这个哈希。如果今天有人周边搜过日餐了（命中 Redis），直接花 **0.001秒** 甚至不用调网络，就把数据拿出去！
3. **生命周期 (TTL)**：设为 `86400`（即 24 小时）。因为工业上通常认为，一家实体物理餐厅在 24 小时内评分和位置不会大变。24小时后缓存自动蒸发销毁，下一个人来搜又会拿到最新鲜的。

### 3.2 职责二：拯救大模型的背压系统 (Job Queue & Pub/Sub)

这就是你惊叹的**“给一个List自发去拿去消耗的背什什么式机制”**！
它的学名叫 **Backpressure / Message Queue (背压 / 消息队列)**。

**传统的 Web 开发 (直接断气崩溃)**：
HTTP 收到 150 个问答，发起 150 根长连接死死连着 Gemini大模型。结果触发 `429 Too Many Requests`，前端全军覆没返回“网络断开连接”。

**我们的工业级重构 (WABI 分布式流转)**：
1. **List 抛下 (Web端)**：
   WebSocket 从用户收到了照片，`chat_manager.py` **不再给 AI 发送任何直接请求**。它只是把内容打个包（JSON），往 Redis 的 `wabi_ai_queue` 列表的尾巴里狠狠一丢（`redis.rpush`），然后开启 PubSub（频道订阅模式）坐在那里等。
   
2. **无尽盲拿 (AI Worker端)**：
   `langgraph_server.py` 这个容器现在是一个彻底的**剥削工头**。它一开机就自启动了 `MAX_CONCURRENT=5` 名小奴隶 Worker！
   这 5 个人每天就做一件事：用 `redis.blpop("wabi_ai_queue", timeout=0)` 死死盯着这个列表！只要列表一进来任务，这5个人里最空闲的那个就“啪”地把任务抢走去执行 LangGraph 节点图。
   
3. **完美疏导 (防洪效应)**：
   如果同时进来了 150 个任务，这 5 个人只能一人抢一个（共同时干 5 个），剩下的 145 个就在 Redis 的内存 List 里静静排队睡觉。**因为并发量永远锁定在了 5 这根红线上，所以无论外面排了多少人，大模型永远不会启动限流封禁（Throat-cut 429）**。

4. **发回完工 (Pub/Sub 回传)**：
   一旦奴隶 AI Worker 做完了营养结算（哪怕过了 70 秒），它只需要拿大喇叭往 Redis 里大喊一声：“频道 `response_123`，做完了！”（`redis.publish()`），一直在等的 `Web` 端就会立刻接收到这一张纸条，把弹窗通过 WebSocket 悠哉悠哉地推到用户的手机屏幕上！
