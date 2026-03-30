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
CONCURRENT_USERS = 100    # Set test scale
MESSAGES_PER_USER = 4     # Total turns per user. Must be >= 2.
DELAY_BETWEEN_MSGS = 3.0   # Seconds to wait after receiving AI response before sending next msg

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
        "times_per_turn": {i: [] for i in range(MESSAGES_PER_USER)},
        "errors_per_turn": {i: 0 for i in range(MESSAGES_PER_USER)}
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
                        response_raw = await asyncio.wait_for(websocket.recv(), timeout=90.0)
                        response = json.loads(response_raw)
                        
                        if response.get("type") == "typing":
                            if not typing_received:
                                typing_received = True
                        elif response.get("type") == "message":
                            response_time = time.time() - msg_start_time
                            
                            # Distribute latency metric to the correct bucket
                            stats["times_per_turn"][msg_idx].append(response_time)
                                
                            msg_success = True
                            
                            # Log what happened to see progress without truncation
                            ai_text = response.get("content", "")
                            ai_text_clean = ai_text.replace("\n", " ")
                            logger.info(f"User {user_index} [Turn {msg_idx+1}: {task_type}] RT: {response_time:.2f}s -> {ai_text_clean}")
                            break
                        elif response.get("type") == "error":
                            logger.error(f"User {user_index} [{task_type}] received Server ERROR: {response.get('content')}")
                            stats["messages_failed"] += 1
                            stats["errors_per_turn"][msg_idx] += 1
                            break
                    except asyncio.TimeoutError:
                        logger.error(f"User {user_index} [{task_type}] timed out waiting for message #{msg_idx+1}.")
                        stats["messages_failed"] += 1
                        stats["errors_per_turn"][msg_idx] += 1
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
    
    times_per_turn_all = {i: [] for i in range(MESSAGES_PER_USER)}
    errors_per_turn_all = {i: 0 for i in range(MESSAGES_PER_USER)}
    
    for r in results:
        for i in range(MESSAGES_PER_USER):
            times_per_turn_all[i].extend(r["times_per_turn"][i])
            errors_per_turn_all[i] += r["errors_per_turn"][i]
        
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
    print("📊【每轮对话性能统计 (Per-Turn Analytics)】")
    
    for i in range(MESSAGES_PER_USER):
        turn_times = times_per_turn_all[i]
        turn_errors = errors_per_turn_all[i]
        # In this script structure, turn content maps to msg_idx:
        # i < regular_msgs -> chitchat
        # i == regular_msgs -> recommendation
        # i > regular_msgs -> recognition
        regular_msgs = max(0, MESSAGES_PER_USER - 2)
        if i < regular_msgs:
            task_name = "闲聊 (Chitchat)"
        elif i == regular_msgs:
            task_name = "餐厅推荐 (Recommendation)"
        else:
            task_name = "食物识别 (Recognition)"
            
        avg_time = calculate_avg(turn_times)
        max_time = max(turn_times) if turn_times else 0.0
        success_count = len(turn_times)
        attempt_count = success_count + turn_errors
        error_rate = (turn_errors / attempt_count * 100) if attempt_count > 0 else 0.0
        
        print(f"  🟢 第 {i+1} 轮 [{task_name}]:")
        print(f"     - 平均延迟: {avg_time:.2f} 秒 | 最大延迟: {max_time:.2f} 秒")
        print(f"     - 错误率:   {error_rate:.1f}% ({turn_errors}/{attempt_count} 失败)")
        
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