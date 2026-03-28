"""Advanced Industrial-grade async WebSocket load testing script for WABI.

This script simulates `N` concurrent users connecting via WebSockets
to the `wabi-web` frontend. It tests memory accumulation latency (first half vs second half)
as well as specific heavy-duty nodes (Food Recommendation and Image Recognition).

Usage:
  python tests/load_test_ws.py
"""

import os
import random

# Disable proxy for local testing to avoid python-socks requirement on macOS VPNs
for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(k, None)

import asyncio
import json
import time
import uuid
import logging
import base64
from typing import List, Dict, Any

import websockets

# Configure logging to match industrial standards
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
)
logger = logging.getLogger("wabi.loadtest")

# CONFIGURE TEST DEMOGRAPHICS HERE
WS_URL = "ws://localhost:8000/ws"
CONCURRENT_USERS = 10     # Set test scale
MESSAGES_PER_USER = 6      # Total turns per user. Must be >= 2.
DELAY_BETWEEN_MSGS = 2.0   # Seconds to wait after receiving AI response before sending next msg

# Pre-load image for recognition
IMAGE_PATH = "burger.jpg"
burger_b64 = ""
if os.path.exists(IMAGE_PATH):
    with open(IMAGE_PATH, "rb") as f:
        burger_b64 = base64.b64encode(f.read()).decode('utf-8')
else:
    logger.warning(f"Could not find {IMAGE_PATH}! Image Recognition test will fail.")


async def simulate_user(user_index: int) -> Dict[str, Any]:
    """Simulates a single user executing different LLM tasks."""
    user_id = f"loadtest_user_adv_{uuid.uuid4().hex[:6]}"
    url = f"{WS_URL}/{user_id}"
    
    stats: Dict[str, Any] = {
        "user_index": user_index,
        "is_failed_user": False,
        "messages_sent": 0,
        "messages_failed": 0,
        "times_first_half": [],
        "times_second_half": [],
        "times_recommendation": [],
        "times_recognition": []
    }
    
    regular_msgs = max(0, MESSAGES_PER_USER - 2)
    first_half_limit = regular_msgs // 2
    
    logger.info(f"User {user_index} ({user_id}) attempting connection...")
    
    start_time = time.time()
    try:
        # 1. Establish WebSocket connection
        async with websockets.connect(url, ping_interval=None) as websocket:
            logger.info(f"User {user_index} ({user_id}) Connected.")
            
            for msg_idx in range(MESSAGES_PER_USER):
                
                # Determine what kind of message it is:
                if msg_idx < regular_msgs:
                    task_type = "chitchat"
                    content = f"你好，我是用户{user_index}，我的消息编号是{msg_idx+1}，你能做个1+1的算术题吗？"
                    payload = {"type": "text", "content": content}
                elif msg_idx == regular_msgs:
                    task_type = "recommendation"
                    payload = {
                        "type": "text", 
                        "content": "我今天想吃点特别好的，能结合我的口味帮我推荐一家评分很高而且有特色的餐厅吗？"
                    }
                else:
                    task_type = "recognition"
                    payload = {
                        "type": "image",
                        "text": "帮我看看这个汉堡大概有多少卡路里？我今天吃这个合适吗？",
                        "content": burger_b64,
                        "mime_type": "image/jpeg"
                    }
                
                # 3. Send message
                msg_start_time = time.time()
                await websocket.send(json.dumps(payload))
                stats["messages_sent"] += 1
                
                # 4. Wait for AI response
                typing_received = False
                msg_success = False
                
                while True:
                    try:
                        # Recognition has 4 serial LLM calls → needs more time
                        recv_timeout = 180.0 if task_type == "recognition" else 90.0
                        response_raw = await asyncio.wait_for(websocket.recv(), timeout=recv_timeout)
                        response = json.loads(response_raw)
                        
                        if response.get("type") == "typing":
                            if not typing_received:
                                typing_received = True
                        elif response.get("type") == "message":
                            response_time = time.time() - msg_start_time
                            
                            # Distribute latency metric to the correct bucket
                            if task_type == "chitchat":
                                if msg_idx < first_half_limit:
                                    stats["times_first_half"].append(response_time)
                                else:
                                    stats["times_second_half"].append(response_time)
                            elif task_type == "recommendation":
                                stats["times_recommendation"].append(response_time)
                            elif task_type == "recognition":
                                stats["times_recognition"].append(response_time)
                                
                            msg_success = True
                            
                            # Log what happened to see progress
                            ai_text = response.get("content", "")
                            logger.info(f"User {user_index} [{task_type}] RT: {response_time:.2f}s -> {ai_text[:20]}...")
                            break
                        elif response.get("type") == "error":
                            logger.error(f"User {user_index} [{task_type}] received Server ERROR: {response.get('content')}")
                            stats["messages_failed"] += 1
                            break
                    except asyncio.TimeoutError:
                        logger.error(f"User {user_index} [{task_type}] timed out waiting for message #{msg_idx+1}.")
                        stats["messages_failed"] += 1
                        break
                        
                if not msg_success and "messages_failed" not in str(stats):
                    pass
                        
                # Wait before simulating typing the next message
                if msg_idx < MESSAGES_PER_USER - 1:
                    await asyncio.sleep(DELAY_BETWEEN_MSGS + random.uniform(-0.5, 0.5))
                    
    except Exception as e:
        logger.error(f"User {user_index} connection failed/dropped: {e}")
        stats["is_failed_user"] = True
        
    duration = time.time() - start_time
    
    if stats["messages_failed"] > 0:
        stats["is_failed_user"] = True
        
    return stats


def calculate_avg(times: List[float]) -> float:
    return (sum(times) / len(times)) if times else 0.0

async def main():
    logger.info(f"Starting ADVANCED WABI Load Test: {CONCURRENT_USERS} users, {MESSAGES_PER_USER} turns each.")
    start_time = time.time()
    
    import redis.asyncio as redis
    redis_client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
    await redis_client.delete("wabi_metrics:cache_hit")
    await redis_client.delete("wabi_metrics:cache_miss")
    logger.info("Cleared global Redis cache tracking metrics.")
    
    tasks = [simulate_user(i) for i in range(1, CONCURRENT_USERS + 1)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Aggregate statistics
    total_users_failed = sum(1 for r in results if r["is_failed_user"])
    total_messages_sent = sum(r["messages_sent"] for r in results)
    total_messages_failed = sum(r["messages_failed"] for r in results)
    
    all_first_half = []
    all_second_half = []
    all_recommendation = []
    all_recognition = []
    
    for r in results:
        all_first_half.extend(r["times_first_half"])
        all_second_half.extend(r["times_second_half"])
        all_recommendation.extend(r["times_recommendation"])
        all_recognition.extend(r["times_recognition"])
        
    user_failure_rate = (total_users_failed / CONCURRENT_USERS) * 100
    msg_failure_rate = (total_messages_failed / total_messages_sent) * 100 if total_messages_sent else 0.0
    
    print("\n" + "="*60)
    print("🚀 WABI ADVANCED MULTI-TASK LOAD TEST RESULTS 🚀")
    print("="*60)
    print(f"⏱️  Total Test Duration:      {total_time:.2f} seconds")
    print(f"👥 Total Concurrent Users:  {CONCURRENT_USERS}")
    print(f"💬 Total Messages Sent:     {total_messages_sent}")
    print("-" * 60)
    print(f"👤 单人遇到失败的比率:      {user_failure_rate:.1f}% ({total_users_failed}/{CONCURRENT_USERS} 人经历了报错)")
    print(f"❌ 单轮对话失败的比率:      {msg_failure_rate:.1f}% ({total_messages_failed}/{total_messages_sent} 次对话丢失或超限)")
    print("-" * 60)
    print("📊【延迟性能分析 (Latency Analysis)】")
    print(f"  📝 前半段闲聊平均延迟 (上下文短): {calculate_avg(all_first_half):.2f} 秒")
    print(f"  📝 后半段闲聊平均延迟 (上下文长): {calculate_avg(all_second_half):.2f} 秒  <-- (如果这个显著比前半段高，说明内存越来越大拖慢了推理)")
    print(f"  🍽️ 餐厅推荐节点专项延迟 (Search): {calculate_avg(all_recommendation):.2f} 秒")
    print(f"  🍔 视觉图像识别专项延迟 (Vision Model+RAG): {calculate_avg(all_recognition):.2f} 秒")
    print("="*60)
    
    try:
        hits = int(await redis_client.get("wabi_metrics:cache_hit") or 0)
        misses = int(await redis_client.get("wabi_metrics:cache_miss") or 0)
        total_cache_requests = hits + misses
        hit_rate = (hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        print("📈【语义缓存性能 (Semantic Cache Analytics)】")
        print(f"  🎯 Google Maps API 拦截率: {hit_rate:.1f}%")
        print(f"  ✅ 缓存命中次数 (Cache Hits): {hits} (节约了 API 额度和 {hits*3} 秒等待)")
        print(f"  ❌ 真实 API 请求 (Cache Misses): {misses}")
        print("="*60 + "\n")
    except Exception as e:
        logger.error(f"Failed to fetch Redis cache metrics: {e}")


if __name__ == "__main__":
    asyncio.run(main())

"""
# 清除测试用户数据库指令 (清理新脚本生成的 adv 前缀僵尸号)：
sqlite3 data/wabi_chat.db "DELETE FROM users WHERE user_id LIKE 'loadtest_user_adv_%'; DELETE FROM messages WHERE user_id LIKE 'loadtest_user_adv_%';"
"""