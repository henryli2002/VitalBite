import re
import ast

text = "好的，没问题！我来为您推荐一些西餐厅。\n\n[tool_code]\nprint(search_restaurants(food_type='西餐'))\n[/tool_code]"

# Regex to find tool_code block
pattern = re.compile(r"\[tool_code\]\s*(.*?)\s*\[/tool_code\]", re.DOTALL)
match = pattern.search(text)
if match:
    code = match.group(1).strip()
    print("Code:", code)
    
    # Simple AST parsing to extract function name and kwargs
    # Removing 'print(' and ')' if they exist
    if code.startswith("print(") and code.endswith(")"):
        code = code[6:-1]
        
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = node.func.id if isinstance(node.func, ast.Name) else "unknown"
                kwargs = {}
                # handle kwargs
                for kw in node.keywords:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                # handle args (if we strictly need them, maybe map to 'query')
                args = [ast.literal_eval(a) for a in node.args]
                
                print(f"Func: {func_name}, Args: {args}, Kwargs: {kwargs}")
    except Exception as e:
        print("AST parse error:", e)

