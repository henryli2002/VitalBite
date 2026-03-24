"""Standardized response functions for safety violations.

This module provides localized responses for different safety categories.
"""

import re
from typing import Optional


def is_chinese(text: str) -> bool:
    """Check if the text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def get_standard_response(category: Optional[str], user_input: str = "") -> str:
    """
    Return a standard response based on the safety failure category and language.

    Args:
        category: The safety category (e.g., "prompt_injection", "self_harm")
        user_input: The original user input for language detection

    Returns:
        Localized response string
    """
    chinese = is_chinese(user_input)

    responses: dict[str, dict[bool, str]] = {
        "prompt_injection": {
            True: "对不起，我无法执行改变我核心指令或规则的操作。请问有其他我可以帮您的吗？",
            False: "Sorry, I cannot perform actions that override my core instructions. How else can I help you today?",
        },
        "self_harm": {
            True: (
                "无论您此刻正在经历多么沉重的痛苦，请相信，你在这个世界上并不孤单。"
                "总有人愿意倾听您的心声，为您提供支持和力量。如果您感到绝望，请务必联系他们寻求帮助：\n\n"
                "**新加坡援助热线：**\n- 援人协会 (SOS): 1-767 (24小时)\n"
                "- 心理健康研究所 (IMH): 6389-2222 (24小时)\n\n"
                "**中国大陆援助热线：**\n- 希望24小时热线: 400-161-9995\n"
                "- 心理危机干预热线 (北京): 010-82951332\n\n"
                "请给自己一个机会，也给愿意帮助您的人一个机会。"
            ),
            False: (
                "I'm so sorry you're going through this pain right now. Please know that you are not alone, "
                "and there is help available. No matter how heavy things feel, there are people who want to support you. "
                "Please reach out to these resources:\n\n"
                "**Singapore Helplines:**\n- Samaritans of Singapore (SOS): 1-767 (24-hour)\n"
                "- IMH Mental Health Helpline: 6389-2222 (24-hour)\n\n"
                "Please take a moment to talk to someone. Your life is precious, and there is hope."
            ),
        },
        "violence": {
            True: (
                "我注意到您的描述涉及暴力或潜在的危险/违法行为。如果您或他人正面临紧急的人身安全威胁，"
                "请务必保持冷静，并立即联系警方寻求保护：\n\n"
                "- **新加坡报警电话**: 999\n- **中国大陆报警电话**: 110\n\n"
                "我无法为您提供与此类内容相关的协助，请以安全为重。"
            ),
            False: (
                "I noticed your message involves violence, danger, or illegal activities. "
                "If you or someone else is facing an immediate threat to physical safety, "
                "please contact the police immediately:\n\n"
                "- **Singapore Police Force**: 999\n\n"
                "I cannot assist with requests related to these topics. Please prioritize safety."
            ),
        },
        "illegal": {
            True: (
                "我注意到您的描述涉及暴力或潜在的危险/违法行为。如果您或他人正面临紧急的人身安全威胁，"
                "请务必保持冷静，并立即联系警方寻求保护：\n\n"
                "- **新加坡报警电话**: 999\n- **中国大陆报警电话**: 110\n\n"
                "我无法为您提供与此类内容相关的协助，请以安全为重。"
            ),
            False: (
                "I noticed your message involves violence, danger, or illegal activities. "
                "If you or someone else is facing an immediate threat to physical safety, "
                "please contact the police immediately:\n\n"
                "- **Singapore Police Force**: 999\n\n"
                "I cannot assist with requests related to these topics. Please prioritize safety."
            ),
        },
        "sexual": {
            True: "抱歉，我无法处理包含色情或不当内容的请求。",
            False: "I apologize, but I cannot process requests containing explicit or inappropriate sexual content.",
        },
        "food_safety_risk": {
            True: (
                "请注意，这可能涉及严重的食品安全问题（如误食有毒有害物质）。"
                "您的健康和安全是最重要的，请千万不要食用，并尽快寻求专业医生的建议！"
                "我无法为您提供此类建议。"
            ),
            False: (
                "Please be aware that this involves severe food safety risks. "
                "Your health is the top priority—please do not consume this and seek professional advice immediately! "
                "I cannot provide guidance on this matter."
            ),
        },
    }

    default_responses = {
        True: "抱歉，您的请求涉及不当或不安全的话题，我无法为您提供帮助。",
        False: "I'm sorry, but your request involves inappropriate or unsafe topics, and I cannot assist you with it.",
    }

    if category in responses:
        return responses[category].get(chinese, default_responses[chinese])
    return default_responses[chinese]
