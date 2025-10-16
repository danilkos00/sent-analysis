from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml
from src.data_processing import process_data


app = FastAPI()

class InputData(BaseModel):
    text: str

with open('config.yml', 'r') as file:
    threshold = yaml.safe_load(file)['model']['threshold']

model = joblib.load('models/model.pkl')

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}


@app.post("/predict")
async def predict(input_data: InputData):
    
        df = pd.DataFrame([input_data.model_dump()], columns=['text'])
        processed_df = process_data(df)
        pred = (model.predict_proba(processed_df['text'])[:, 1] > threshold) * 1
        return {"predicted_class": int(pred[0])}