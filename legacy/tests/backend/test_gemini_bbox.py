import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(".env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

with open("burger.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode("utf-8")

prompt = """
Detect all distinct food items in this image.
For each item, provide its name and its bounding box in the format [ymin, xmin, ymax, xmax] where values are normalized between 0 and 1000.
Return ONLY valid JSON:
[
  {"name": "burger", "box": [ymin, xmin, ymax, xmax]}
]
"""

msg = HumanMessage(content=[
    {"type": "text", "text": prompt},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
])

response = llm.invoke([msg])
print(response.content)
