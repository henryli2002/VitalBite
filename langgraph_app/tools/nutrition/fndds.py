"""
Tool for interacting with the FNDDS nutritional database using a RAG retriever.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Global Initialization ---
# This part of the code will run once when the module is first imported.
# It loads the model, the FAISS index, and the metadata into memory.
MODEL_NAME = "all-MiniLM-L6-v2"
TOOL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(TOOL_DIRECTORY, "fndds_index.faiss")
METADATA_FILE = os.path.join(TOOL_DIRECTORY, "fndds_metadata.json")

print("Initializing FNDDS RAG Tool...")
try:
    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading FAISS index from: {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)

    print(f"Loading metadata from: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    print("FNDDS RAG Tool initialized successfully.")

except Exception as e:
    print(f"FATAL ERROR: Failed to initialize FNDDS RAG Tool. {e}")
    # Set to None to indicate failure
    model = None
    index = None
    metadata_list = None


# --- Pydantic Schema for Tool Input ---
class FnddsSearchInput(BaseModel):
    """Input for searching the FNDDS database."""

    food_description: str = Field(
        description="A description of the food to search for."
    )
    top_k: int = Field(default=3, description="The number of top results to return.")


# --- The Tool Definition ---
@tool("fndds_nutrition_search", args_schema=FnddsSearchInput)
def fndds_nutrition_search_tool(food_description: str, top_k: int = 3) -> str:
    """
    Searches the FNDDS vector database for the most relevant nutritional
    information based on a food description.
    """
    if not all([model, index, metadata_list]):
        error_message = (
            "FNDDS RAG Tool is not initialized. Check for errors on startup."
        )
        print(error_message)
        return json.dumps({"error": error_message})

    try:
        print(f"Performing RAG search for: '{food_description}', k={top_k}")

        # 1. Encode the user's query
        query_embedding = model.encode([food_description])
        query_embedding = np.array(query_embedding).astype("float32")

        # 2. Search the FAISS index
        distances, indices = index.search(query_embedding, top_k)

        # 3. Retrieve the corresponding metadata
        results = [metadata_list[i] for i in indices[0]]

        print(f"Found {len(results)} results.")

        # 4. Return the results as JSON string
        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        error_message = f"An error occurred during the FNDDS RAG search: {e}"
        print(error_message)
        return json.dumps({"error": error_message})


# Example of how to test this tool directly
if __name__ == "__main__":
    print("\\n--- Testing FNDDS RAG Tool ---")
    if not all([model, index, metadata_list]):
        print("Could not run test because the tool failed to initialize.")
    else:
        test_query = "a glass of whole milk"
        search_results_json = fndds_nutrition_search_tool.invoke(
            {"food_description": test_query, "top_k": 2}
        )

        print(f"\nResults for query: '{test_query}'")
        search_results = json.loads(search_results_json)
        print(json.dumps(search_results, indent=2, ensure_ascii=False))
