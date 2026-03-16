import faiss
import json
import os

def inspect_faiss_index():
    index_path = 'langgraph_app/tools/nutrition/fndds_index.faiss'
    docs_path = 'langgraph_app/tools/nutrition/fndds_documents.json'
    
    if not os.path.exists(index_path):
        print(f"Error: FAISS index not found at {index_path}")
        return
        
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    print("\n--- FAISS Index Structure ---")
    print(f"Is Trained: {index.is_trained}")
    print(f"Total Vectors (ntotal): {index.ntotal}")
    print(f"Dimensions (d): {index.d}")
    print(f"Metric Type: {index.metric_type}")
    
    if os.path.exists(docs_path):
        print(f"\nLoading documents from {docs_path}...")
        with open(docs_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            
        print("\n--- Documents Information ---")
        print(f"Total Documents: {len(documents)}")
        if len(documents) > 0:
            print("\nSample Document (Index 0):")
            print("-" * 40)
            print(documents[0])
            print("-" * 40)
            
        if len(documents) != index.ntotal:
            print("\nWARNING: Mismatch between number of documents and vectors in index!")
    else:
        print(f"\nWarning: Documents file not found at {docs_path}")

if __name__ == "__main__":
    inspect_faiss_index()
