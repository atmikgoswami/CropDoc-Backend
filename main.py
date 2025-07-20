import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI()

# Load model
model = pickle.load(open('model.pkl', 'rb'))
print("Loaded model type:", type(model))

# Crop mapping
crop_mapping = {
    0: 'blackgram',
    1: 'chickpea',
    2: 'cotton',
    3: 'jute',
    4: 'kidneybeans',
    5: 'lentil',
    6: 'maize',
    7: 'mothbeans',
    8: 'muskmelon',
    9: 'mungbean',
    10: 'pigeonpeas',
    11: 'rice',
    12: 'watermelon',
}

# Input schema using Pydantic
class RecommendRequest(BaseModel):
    temp: float
    humidity: float
    ph: float
    water: float
    season: float  # You can use Literal[] if it's categorical like 0–3 for seasons

@app.get("/")
def home():
    return {"message": "Hello from Server"}

@app.post("/recommend")
def recommend_crop(data: RecommendRequest):
    input_features = np.array([[data.temp, data.humidity, data.ph, data.water, data.season]], dtype=np.float32)
    prediction = model.predict(input_features)
    predicted_crop = crop_mapping[prediction[0]]
    return {"prediction": predicted_crop}
