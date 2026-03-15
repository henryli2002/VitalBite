# WABI NTU Agent Backend / 智能多模态代理后端

## Overview / 概述
- LangGraph orchestrated multi-agent workflow for food recognition, recommendation, and clarification.
- Supports multimodal (text + image) intent routing, guardrails, and structured outputs.
- Pluggable LLM providers (Gemini / OpenAI / AWS Bedrock Claude) via a unified factory.

## Features / 功能特性
- Intent routing, guardrail safety checks, clarification loop.
- Food recognition with nutrition estimation; restaurant recommendation with structured query extraction.
- Structured outputs via Pydantic schemas; multilingual responses (en/zh) following user language.

## Directory / 目录
- Core config: [langgraph_app/config.py](./langgraph_app/config.py)
- LLM clients & factory: [langgraph_app/utils/gemini_client.py](./langgraph_app/utils/gemini_client.py), [langgraph_app/utils/openai_client.py](./langgraph_app/utils/openai_client.py), [langgraph_app/utils/bedrock_claude_client.py](./langgraph_app/utils/bedrock_claude_client.py), [langgraph_app/utils/llm_factory.py](./langgraph_app/utils/llm_factory.py)
- Agents: clarification / recognition / recommendation in [langgraph_app/agents](./langgraph_app/agents)
- Orchestrator nodes (router, guardrail): [langgraph_app/orchestrator/nodes/router.py](./langgraph_app/orchestrator/nodes/router.py), [langgraph_app/orchestrator/nodes/guardrail.py](./langgraph_app/orchestrator/nodes/guardrail.py)
- Dev doc (CN): [dev_doc/level _1.md](./dev_doc/level%20_1.md)

## LLM Provider Switch / 模型供应商切换
- Config key: `LLM_PROVIDER` in `.env` (default `gemini`). Options: `gemini`, `openai`, `bedrock_claude`.
- Default model names are in [langgraph_app/config.py](./langgraph_app/config.py).
- All nodes/agents call `get_llm_client()` from [langgraph_app/utils/llm_factory.py](./langgraph_app/utils/llm_factory.py), so switching provider is configuration-only.

## Environment / 环境变量
Create `.env` at repo root (see sample [.env.example](./.env.example)):

```env
# Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash-lite

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini

# Bedrock (uses AWS credential chain; no extra API key)
AWS_REGION=ap-southeast-1
BEDROCK_CLAUDE_MODEL_NAME=anthropic.claude-3-haiku-20240307-v1:0  # default low-cost Haiku; switch to Sonnet if needed
# Optional local creds (prefer IAM role/temporary creds)
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_SESSION_TOKEN=...

# Provider selection
LLM_PROVIDER=gemini

# History config
HISTORY_MESSAGE_COUNT=6
```

## Install / 安装
```bash
pip install -e .
```

## Quick Start / 快速开始
1) Prepare `.env` as above.
2) Run your LangGraph entry (example):
```python
from langgraph_app.orchestrator.graph import graph

result = graph.invoke({
    "input": {
        "text": "这是什么菜？",
        "image_data": "<base64>",
        "source": "user"
    }
})
print(result["final_response"])
```

## Tests / 测试
Example (adjust as needed):
```bash
pytest tests
```

## Notes / 备注
- Bedrock uses AWS SigV4 credential chain; ensure environment/IAM role provides permissions.
- Multimodal vision support depends on the selected provider’s model capabilities.
- Pricing (参考价，需以官方最新定价为准；均为约/百万 tokens)：
  - Claude 3 Haiku (default, Bedrock，支持多模态): 输入 ~$0.8 / 输出 ~$4
  - Claude 3.5 / 3 Sonnet (Bedrock，支持多模态): 输入 ~$3 / 输出 ~$15（高质高价）
  - Gemini 2.5 Flash (多模态): 输入 ~$0.2–0.35 / 输出 ~$0.6–1.0（高性价比）
  - OpenAI 4o-mini (多模态): 输入 ~$0.15 / 输出 ~$0.60（低成本平衡）
  - OpenAI 4o (多模态): 输入 ~$2.5 / 输出 ~$10（中高端）
  - 以上为常见公开价位级别，具体以各官方实时定价为准。
- Cost (rough order, refer to official pricing):
  - Bedrock Claude 3 Haiku (default): lowest cost in Claude family; cheaper than Sonnet; good for high-concurrency, general tasks.
  - Bedrock Claude 3.5/3 Sonnet: higher quality, significantly higher cost than Haiku.
  - Gemini Flash: high speed & cost-effective; Pro/Ultra tiers are pricier and better quality.
  - OpenAI 4o-mini: low-cost, balanced; 4o/vision-capable models are mid-tier cost and higher quality.
