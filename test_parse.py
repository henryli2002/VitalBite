raw_content = [
    {"type": "text", "text": "很抱歉，我尝试为您寻找更多西餐厅的推荐，但目前"},
    {"type": "tool_call", "name": "search_restaurants", "args": {"query": "西餐"}}
]

text_parts = [
    c.get("text", "") 
    for c in raw_content 
    if isinstance(c, dict) and "text" in c
]
final_msg = " ".join(text_parts).strip() if text_parts else str(raw_content)
print(final_msg)
