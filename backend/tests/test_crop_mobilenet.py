import io
import base64
from PIL import Image
from backend.langgraph_app.agents.food_recognition.predictor import predict_nutrition

with open("burger.jpg", "rb") as f:
    img_bytes = f.read()

# 1. Full image
res_full = predict_nutrition(img_bytes)
print("Full image:", res_full)

# 2. Crop burger
img = Image.open(io.BytesIO(img_bytes))
w, h = img.size
ymin, xmin, ymax, xmax = 100, 170, 899, 829
box = (int(xmin * w / 1000), int(ymin * h / 1000), int(xmax * w / 1000), int(ymax * h / 1000))
burger_img = img.crop(box)
buf = io.BytesIO()
burger_img.save(buf, format='JPEG')
res_burger = predict_nutrition(buf.getvalue())
print("Burger crop:", res_burger)

# 3. Crop fries
ymin, xmin, ymax, xmax = 0, 0, 400, 499
box = (int(xmin * w / 1000), int(ymin * h / 1000), int(xmax * w / 1000), int(ymax * h / 1000))
fries_img = img.crop(box)
buf = io.BytesIO()
fries_img.save(buf, format='JPEG')
res_fries = predict_nutrition(buf.getvalue())
print("Fries crop:", res_fries)

