"""
This program implements a FastAPI web service for predicting the species of iris plants.
It attempts to load a pre-trained model from 'model.pkl'. If the model file is not found,
it calls a function from the 'model_generator' module to create the model, and then loads it.
The service defines a POST /predict endpoint that accepts iris flower measurements
(sepal length, sepal width, petal length, petal width) as input and returns the predicted iris species.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import model_generator

app = FastAPI()

try:
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)
except FileNotFoundError:
    model_generator.create_model()
    with open('model.pkl', 'rb') as f:
        model = joblib.load(f)

iris_mapping = {0: 'Iris setosa', 1: 'Iris versicolor', 2: 'Iris virginica'}


class InputData(BaseModel):
    sepal_len: float
    sepal_wid: float
    petal_len: float
    petal_wid: float


@app.post("/predict")
async def predict(data: InputData):
    data_dict = data.dict()
    mapped_data = {
        "sepal length (cm)": data_dict["sepal_len"],
        "sepal width (cm)": data_dict["sepal_wid"],
        "petal length (cm)": data_dict["petal_len"],
        "petal width (cm)": data_dict["petal_wid"]
    }
    input_data = pd.DataFrame([mapped_data])
    prediction = model.predict(input_data)
    predicted_label = iris_mapping[int(prediction[0])]
    return {"prediction": predicted_label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
