import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import re
import json


def parse_nutrient_column(column_name):
    """Parses a nutrient column header like 'Protein (g)' into name and unit."""
    match = re.match(r"(.+?)\s+\((.+)\)", column_name)
    if match:
        return match.groups()
    return column_name, "N/A"


def create_food_data(row, nutrients_df, nutrient_columns, portions_df):
    """
    Creates both embedding text and metadata for a single food item.
    - embedding_text: used for semantic search (food name + portion description)
    - metadata: full nutritional data + portion weight (returned after recall)
    """
    food_code = row["Food code"]
    food_name = row["Main food description"]

    # Get standard portion info
    portion_row = portions_df[
        (portions_df["Food code"] == food_code) & (portions_df["Seq num"] == 1)
    ]
    portion_desc = None
    portion_weight = None
    if not portion_row.empty:
        portion_desc = portion_row.iloc[0]["Portion description"]
        portion_weight_val = portion_row.iloc[0]["Portion weight\n(g)"]
        if pd.notna(portion_weight_val) and portion_weight_val > 0:
            portion_weight = float(portion_weight_val)
            portion_desc = f"{portion_desc} ({portion_weight}g)"

    # embedding_text: only food name + portion description (for semantic search)
    embedding_text = food_name
    if portion_desc:
        embedding_text += f" - Standard portion: {portion_desc}"

    # Build full text for metadata (includes all info for display)
    full_text = f"Food: {food_name}\nFood Code: {food_code}\n"
    if portion_desc:
        full_text += f"Standard Portion: {portion_desc}\n"

    # Get nutrient data
    nutrient_data_row = nutrients_df[nutrients_df["Food code"] == row["Food code"]]
    nutrients = {}
    if not nutrient_data_row.empty:
        nutrient_series = nutrient_data_row.iloc[0]
        full_text += "Nutritional Information (per 100g):\n"
        for col in nutrient_columns:
            nutrient_value = nutrient_series[col]
            if pd.notna(nutrient_value):
                nutrient_name, nutrient_unit = parse_nutrient_column(col)
                nutrients[nutrient_name] = {
                    "value": float(nutrient_value),
                    "unit": nutrient_unit,
                }
                full_text += f"- {nutrient_name}: {nutrient_value} {nutrient_unit}\n"

    # Metadata to return after recall
    metadata = {
        "food_name": food_name,
        "food_code": int(food_code),
        "standard_portion_weight_g": portion_weight,
        "standard_portion_description": portion_desc,
        "nutrients_per_100g": nutrients,
        "full_text": full_text,
    }

    return embedding_text, metadata


def build_fndds_vector_store():
    """
    Reads FNDDS XLSX files, processes them, creates vector embeddings,
    and saves a FAISS index with metadata.
    """
    base_path = "langgraph_app/agents/food_recognition/databases"
    output_path = "langgraph_app/tools/nutrition"
    index_file = os.path.join(output_path, "fndds_index.faiss")
    metadata_file = os.path.join(output_path, "fndds_metadata.json")

    os.makedirs(output_path, exist_ok=True)

    print("Loading FNDDS data from XLSX files...")
    try:
        foods_df = pd.read_excel(
            os.path.join(base_path, "Foods and Beverages.xlsx"),
            header=1,
        )
        nutrients_df = pd.read_excel(
            os.path.join(base_path, "FNDDS Nutrient Values.xlsx"),
            header=1,
        )
        portions_df = pd.read_excel(
            os.path.join(base_path, "Portions and Weights.xlsx"),
            header=1,
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find XLSX files. Details: {e}")
        return

    print(
        f"Loaded {len(foods_df)} foods, {len(nutrients_df)} nutrient rows, {len(portions_df)} portion rows"
    )
    print("Processing data and creating documents...")

    nutrient_columns = nutrients_df.columns[4:]

    # Create embedding texts and metadata
    embedding_texts = []
    metadata_list = []
    for _, row in foods_df.iterrows():
        embedding_text, metadata = create_food_data(
            row, nutrients_df, nutrient_columns, portions_df
        )
        embedding_texts.append(embedding_text)
        metadata_list.append(metadata)

    print(f"Created {len(embedding_texts)} food documents.")

    if not embedding_texts:
        print("No documents were created. Aborting.")
        return

    print("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(
        "Generating embeddings for all food documents (this may take a few minutes)..."
    )
    embeddings = model.encode(embedding_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    print(f"Embeddings created with dimension: {dimension}")

    print("Building and saving FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)

    print(f"Saving metadata to {metadata_file}...")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print("\nVector store build process complete!")
    print(f"Index file saved to: {index_file}")
    print(f"Metadata file saved to: {metadata_file}")


if __name__ == "__main__":
    build_fndds_vector_store()
