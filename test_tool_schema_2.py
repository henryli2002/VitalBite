import json
from langgraph_app.tools.recommendation_tool import search_restaurants

print(json.dumps(search_restaurants.args_schema.schema(), indent=2))
