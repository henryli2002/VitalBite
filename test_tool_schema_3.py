import json
from langchain_core.tools import tool

@tool(parse_docstring=True)
def test_tool(image_uuid: str) -> str:
    """Analyze a food image.

    Args:
        image_uuid: The 32-hex ID of the image.
    """
    return "ok"

print(json.dumps(test_tool.args_schema.schema(), indent=2))
