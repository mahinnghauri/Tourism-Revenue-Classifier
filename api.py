from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Tourism Revenue Prediction API",
    description="API for predicting tourism revenue using Random Forest",
    version="1.0.0"
)

model = joblib.load("tourism_rf_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

class TourismInput(BaseModel):
    Location: int
    Country: int
    Category: int
    Visitors: int
    Rating: float
    Accommodation_Available: int

class PredictionOutput(BaseModel):
    predicted_revenue: float
    formatted_revenue: str

@app.get("/")
def home():
    return {"message": "Tourism Revenue Prediction API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TourismInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        df = df[feature_columns]
        prediction = model.predict(df)[0]

        return {
            "predicted_revenue": float(prediction),
            "formatted_revenue": f"${prediction:,.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


