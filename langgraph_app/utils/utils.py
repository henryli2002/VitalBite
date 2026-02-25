

def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return "Chinese"
    return "English"