import os
import pandas as pd
from fastapi import FastAPI, Field
from pydantic import BaseModel
from ml.data import apply_label, process_data
from ml.model import inference, load_model

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

try:
    model = load_model(MODEL_PATH)
    encoder = load_model(ENCODER_PATH)
    lb = load_model(LB_PATH)
except FileNotFoundError as e:
    print(
        f"ERROR: Model artifact not found at {e.filename}. "
        "Run train_model.py first."
    )

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(
        ..., example="United-States", alias="native-country"
    )

app = FastAPI(title="Census Income Classifier API")

@app.get("/")
async def get_root():
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.model_dump(by_alias=True)
    data_single_row = {k: [v] for k, v in data_dict.items()}
    input_df = pd.DataFrame.from_dict(data_single_row)
    data_processed, _, _, _ = process_data(
        X=input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    prediction = inference(model, data_processed)
    return {"prediction": apply_label(prediction)}
