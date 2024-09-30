# app.py
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictIn, PredictOut
import joblib
from mangum import Mangum


def get_model():
    model = joblib.load('model_pipeline.joblib')
    return model


MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()

orgins = [
    "http://localhost",
    "http://localhost:8000",
    "https://tb43gljhbe.execute-api.ap-northeast-2.amazonaws.com/mlops"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=orgins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    pred = MODEL.predict(df).item()
    return PredictOut(iris_class=pred)

handler = Mangum(app)