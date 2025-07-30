from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/beer_similarity.joblib")

class BeerInput(BaseModel):
    beers: list[str]

@app.post("/recommend")
def recommend_beer(input_data: BeerInput):
    similarities = model[input_data.beers].mean(axis=1).sort_values(ascending=False)
    recommended = similarities.drop(input_data.beers).head(5)
    return {"recommendations": recommended.index.tolist()}
