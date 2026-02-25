import random
import os
import glob
import base64

def get_random_image(folder_name: str) -> str:
    """Retrieves a random image path from the specified folder in tests/router_intent/images/."""
    base_path = os.path.join(os.path.dirname(__file__),folder_name)
    # Support common image extensions
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(base_path, ext)))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {base_path}")
    
    return random.choice(image_files)

def random_image_to_base64(folder_name: str) -> str:
    random_image_path = get_random_image(folder_name)
    with open(random_image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
    return image_b64

if __name__ == "__main__":
    print(random_image_to_base64("food_img"))