import os
import random
import base64
import sys

def get_random_images_base64(directory, n=5):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in {directory}.")
        return

    selected_files = random.sample(files, min(n, len(files)))
    
    for filename in selected_files:
        path = os.path.join(directory, filename)
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            print(f"--- File: {filename} ---")
            print(encoded_string)
            print("-" * 40)

if __name__ == "__main__":
    n = 30
    
    food_dir = "tests/router_intent/images/food_img"
    get_random_images_base64(food_dir, n)
