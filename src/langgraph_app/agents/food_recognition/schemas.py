from pydantic import BaseModel, Field
from typing import List

class BoundingBox(BaseModel):
    ymin: int = Field(description="Y min (0-1000)")
    xmin: int = Field(description="X min (0-1000)")
    ymax: int = Field(description="Y max (0-1000)")
    xmax: int = Field(description="X max (0-1000)")

class DetectedFood(BaseModel):
    name: str = Field(description="Name of the food item in English")
    box: BoundingBox = Field(description="Bounding box of the food item")

class FoodDetection(BaseModel):
    items: List[DetectedFood] = Field(description="List of detected food items")
