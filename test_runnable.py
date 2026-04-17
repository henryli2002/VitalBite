from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
import re
import ast
import uuid

def parse_tool_code(message: AIMessage) -> AIMessage:
    if not isinstance(message, AIMessage):
        return message
        
    text = message.content
    if not isinstance(text, str) or "[tool_code]" not in text:
        return message
        
    pattern = re.compile(r"\[tool_code\]\s*(.*?)\s*\[/tool_code\]", re.DOTALL)
    match = pattern.search(text)
    
    if match:
        code = match.group(1).strip()
        # Clean up text by removing the tool_code block
        clean_text = pattern.sub("", text).strip()
        
        # Parse AST
        if code.startswith("print(") and code.endswith(")"):
            code = code[6:-1]
            
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = node.func.id if isinstance(node.func, ast.Name) else None
                    if not func_name:
                        continue
                        
                    kwargs = {}
                    for kw in node.keywords:
                        try:
                            kwargs[kw.arg] = ast.literal_eval(kw.value)
                        except:
                            pass
                    
                    args = []
                    for a in node.args:
                        try:
                            args.append(ast.literal_eval(a))
                        except:
                            pass
                            
                    # Map first arg to 'query' if it's search_restaurants and 'query' isn't in kwargs
                    if func_name == "search_restaurants":
                        if args and "query" not in kwargs:
                            kwargs["query"] = args[0]
                        elif "food_type" in kwargs and "query" not in kwargs:
                            kwargs["query"] = kwargs.pop("food_type")
                            
                    elif func_name == "analyze_food_image":
                        if args and "image_uuid" not in kwargs:
                            kwargs["image_uuid"] = args[0]
                            
                    tool_call = {
                        "name": func_name,
                        "args": kwargs,
                        "id": f"call_{uuid.uuid4().hex[:8]}"
                    }
                    
                    # Return new AIMessage with tool_calls
                    return AIMessage(
                        content=clean_text,
                        tool_calls=[tool_call]
                    )
        except Exception as e:
            print("Parse error:", e)
            
    return message

msg = AIMessage(content="好的，没问题！我来为您推荐一些西餐厅。\n\n[tool_code]\nprint(search_restaurants(food_type='西餐'))\n[/tool_code]")
parsed = parse_tool_code(msg)
print("Content:", parsed.content)
print("Tool Calls:", parsed.tool_calls)
