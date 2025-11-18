import os
import pandas as pd
from fastapi import FastAPI, Field
from pydantic import BaseModel
from ml.data import apply_label, process_data
from ml.model import inference, load_model


# E302/E305 FIX: Ensure 2 blank lines between global statements and functions/classes

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


# E302 FIX: Needs 2 blank lines before class definition
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
    """ Say hello!"""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """
    POST endpoint for model inference.
    Takes census features and returns the predicted income category
    (>50K or <=50K).
    """
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.model_dump(by_alias=True)

    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # E501 FIX: Split long comment line
    # The data has names with hyphens and Python does not allow those as
    # variable names. Here it uses the functionality of FastAPI/Pydantic/etc
    # to deal with this.
    data_single_row = {k: [v] for k, v in data_dict.items()}
    input_df = pd.DataFrame.from_dict(data_single_row)

    # Process data for inference
    data_processed, _, _, _ = process_data(
        X=input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict the result
    prediction = inference(model, data_processed)

    # Return the decoded label
    return {"prediction": apply_label(prediction)}