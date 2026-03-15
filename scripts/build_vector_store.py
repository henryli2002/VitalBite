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

def create_food_document_from_wide_data(row, nutrients_df, nutrient_columns):
    """
    Creates a descriptive text document for a single food item from wide-format nutrient data.
    """
    description = f"Food: {row['Main food description']}\\n"
    description += f"Food Code: {row['Food code']}\\n"
    
    # Find the corresponding row in the nutrients dataframe
    nutrient_data_row = nutrients_df[nutrients_df['Food code'] == row['Food code']]
    
    if not nutrient_data_row.empty:
        description += "Nutritional Information (per 100g):\\n"
        # Access the first (and should be only) row of the filtered data
        nutrient_series = nutrient_data_row.iloc[0]
        
        # Iterate over the pre-identified nutrient columns
        for col in nutrient_columns:
            nutrient_value = nutrient_series[col]
            # Avoid adding nutrients with no value
            if pd.notna(nutrient_value):
                nutrient_name, nutrient_unit = parse_nutrient_column(col)
                description += f"- {nutrient_name}: {nutrient_value} {nutrient_unit}\\n"
            
    return description

def build_fndds_vector_store():
    """
    Reads FNDDS XLSX files, processes them, creates vector embeddings,
    and saves a FAISS index.
    """
    base_path = 'langgraph_app/agents/food_recognition/databases'
    output_path = 'langgraph_app/tools/nutrition'
    index_file = os.path.join(output_path, 'fndds_index.faiss')
    mapping_file = os.path.join(output_path, 'fndds_documents.json')
    
    os.makedirs(output_path, exist_ok=True)

    print("Loading FNDDS data from XLSX files...")
    try:
        # Use skiprows=[1] to skip the second row which is often a title row in these files
        foods_df = pd.read_excel(os.path.join(base_path, '2021-2023 FNDDS At A Glance - Foods and Beverages.xlsx'), header=1)
        nutrients_df = pd.read_excel(os.path.join(base_path, '2021-2023 FNDDS At A Glance - FNDDS Nutrient Values.xlsx'), header=1)
    except FileNotFoundError as e:
        print(f"Error: Could not find XLSX files. Details: {e}")
        return

    print("Processing data and creating documents...")
    
    # Identify nutrient columns to iterate through (all columns after the first 4)
    nutrient_columns = nutrients_df.columns[4:]
    
    # Create a text document for each food item
    documents = [
        create_food_document_from_wide_data(row, nutrients_df, nutrient_columns)
        for _, row in foods_df.iterrows()
    ]
    
    print(f"Created {len(documents)} food documents.")
    
    if not documents:
        print("No documents were created. Aborting.")
        return

    print("Loading sentence transformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings for all food documents (this may take a few minutes)...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    print(f"Embeddings created with dimension: {dimension}")

    print("Building and saving FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)

    print(f"Saving document mapping to {mapping_file}...")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

    print("\\nVector store build process complete!")
    print(f"Index file saved to: {index_file}")
    print(f"Documents file saved to: {mapping_file}")


if __name__ == '__main__':
    build_fndds_vector_store()
