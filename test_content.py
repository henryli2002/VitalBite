def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return " ".join([c.get("text", "") for c in content if isinstance(c, dict) and "text" in c])
    return str(content)

c1 = "hello"
c2 = [{"type": "text", "text": "hello list"}]
print(extract_text(c1))
print(extract_text(c2))
