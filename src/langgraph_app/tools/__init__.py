from langgraph_app.tools.food_recognition_tool import analyze_food_image
from langgraph_app.tools.recommendation_tool import search_restaurants

# All tools available for the Supervisor agent
supervisor_tools = [analyze_food_image, search_restaurants]
