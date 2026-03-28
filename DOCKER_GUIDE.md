# WABI Docker 使用指南（新手友好版）

欢迎使用 Docker 版本的 WABI！这篇指南专为 Docker 初学者准备，帮助你理解如何启动、关闭以及管理 WABI 的微服务架构。

## 1. 我们的架构长什么样？
我们将 WABI 拆分成了两个**独立运行的盒子（容器）**：
- 🌐 **wabi-web** (端口 8000)：负责展示网页界面、保持用户登录、以及记录历史聊天记录。
- 🧠 **wabi-ai** (端口 8001)：专门负责跑 LangGraph 和 LangChain 的 AI 推理大脑。没有任何历史包袱，纯计算节点。

这两个盒子通过一个叫做 `wabi-network` 的虚拟网线连在一起，Web 会自动向 AI 大脑发送请求。

---

## 2. 如何启动 WABI？

在终端（Terminal）进入 `WABI` 项目文件夹后，运行以下命令：

```bash
docker-compose up -d --build
```
- `up`: 组合启动所有的盒子。
- `-d` (detach): **后台运行**模式。启动后终端不会被卡住，你可以继续输入其他命令。
- `--build`: **强制重新构建环境**。意思是“检查一下我有没有修改代码或依赖，如果有，就重新打包封装一次”。

**启动效果：**
执行完毕后，访问浏览器 **http://localhost:8000** 即可直接使用，就像以前直接运行 `uvicorn` 一样。

### 🌟 新手必看：每次都需要重新构建吗？
**不需要！**
Docker 非常聪明，它有 **缓存（Cache）机制**：
- 如果你**只修改了 Python 代码**（比如修改了 `agent.py`），在使用 `--build` 时它只会瞬间替换代码文件，不会重新下载几十百兆的依赖库，几秒钟就能启动完成。
- 如果你**没有修改任何代码**，下次想打开 WABI 时，**直接运行 `docker-compose up -d` 即可（不需要加 `--build`）**，环境是瞬间直接启动的！

---

## 3. 怎样查看运行日志（报错信息）？

因为我们使用了 `-d` 将程序挂在后台，如果网页报错或者你好奇 AI 正在干嘛，可以随时查看日志。

查看所有容器的实时滚动日志（类似于平时一直盯着终端看）：
```bash
docker-compose logs -f
```
如果你只想看 AI 引擎的日志：
```bash
docker-compose logs -f wabi-ai
```
*(按 `Ctrl + C` 可以随时退出查看日志，这**不会**关闭程序)*

---

## 4. 如何关闭系统？

如果你今天不想用 WABI 了，想关掉服务释放电脑资源，运行：
```bash
docker-compose down
```
它会优雅地停掉 `wabi-web` 和 `wabi-ai` 两个容器，并且切断它们的虚拟网线。

### 🌟 新手必看：关闭会删掉我的聊天记录吗？
**绝对不会！**
我们在 `docker-compose.yml` 中配置了 **数据卷挂载（Volumes）**：
- PostgreSQL 数据库的数据保存在独立的 Docker Volume `wabi_pgdata` 中。
- 所以即使你执行了 `docker-compose down` 销毁了所有的容器，只要你不加上 `-v` 参数，数据库数据和你的账号、Profile 以及历史聊天记录就会自动永久保留！下次启动 `up` 时会自动恢复！

**⚠️ 危险警告**：如果你运行了 `docker-compose down -v`，Docker 将会连同挂载的数据卷一并**彻底删除**！这会导致你的 PostgreSQL 数据库被清空，所有历史数据灰飞烟灭！请永远不要在生产环境中随便加 `-v`。

---

## 5. 日常使用口诀总结

1. 💻 **第一次运行 / 修改了代码**：`docker-compose up -d --build`
2. 🚀 **平时日常打开**：`docker-compose up -d`
3. 🕵️ **看后台报错**：`docker-compose logs -f`
4. 🛑 **关掉不玩了**：`docker-compose down`
5. 🧹 **重新打包（清缓存，不会删数据）**：`docker-compose build --no-cache`
   *(当你遇到奇怪的依赖报错，想彻底从零重新安装所有 Python 库时使用。**请放心，这只会重新构建镜像代码，绝对不会删除你的 PostgreSQL 数据库记录！**)*
   
   如果想连着以前积攒的无用镜像彻底清理释放硬盘空间，可以运行：`docker system prune -a`
6. 📊 **查看服务运行状态**：`docker-compose ps` (简略版) 或 `docker ps` (详细版)
   *(检查容器是否正在运行，或者查看它们被映射到了哪个端口)*
7. 🔄 **重启单个服务**：`docker-compose restart wabi-ai`
   *(比如你只改了 AI 的配置，不想重启整个 Web，可以单独重启它)*
8. 🚪 **进入系统内部（高级）**：`docker exec -it wabi-ai /bin/bash`
   *(类似于 SSH 登录到 AI 大脑所在的那个“独立电脑”里看环境文件)*

