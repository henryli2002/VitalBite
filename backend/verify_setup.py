import sys
import os
from dotenv import load_dotenv

# Load .env first
load_dotenv()

print(f"Python executable: {sys.executable}")

try:
    import langgraph
    from google import genai
    print("Key dependencies found.")
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

try:
    from langgraph_app.orchestrator.graph import graph
    print("Graph imported and compiled successfully.")
except Exception as e:
    print(f"Graph compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment variables. The agent will fail at runtime.")
else:
    print("GEMINI_API_KEY is present.")

print("Verification complete.")
