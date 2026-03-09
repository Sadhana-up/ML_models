from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np


knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseFeatures(BaseModel):
    location_Kathmandu: float
    location_Lalitpur: float
    location_Pokhara: float
    location_Bhaktapur: float
    land_area: float
    square_feet: float
    floors: float
    bedrooms: float
    bathrooms: float
    windows: float
    doors: float
    cement_bags: float
    RCC_structure: float
    plumbing: float
    electricity: float

@app.post("/predict")
def predict_price(features: HouseFeatures):
   
    X = np.array([[
        features.location_Kathmandu,
        features.location_Lalitpur,
        features.location_Pokhara,
        features.location_Bhaktapur,
        features.land_area,
        features.square_feet,
        features.floors,
        features.bedrooms,
        features.bathrooms,
        features.windows,
        features.doors,
        features.cement_bags,
        features.RCC_structure,
        features.plumbing,
        features.electricity
    ]])

    
    X_scaled = scaler.transform(X)

  
    pred = knn.predict(X_scaled)

    return {"predicted_total_price": float(pred[0])}