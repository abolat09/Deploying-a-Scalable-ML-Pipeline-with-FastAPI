import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

data_path = os.path.join("data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
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
LABEL = "salary"

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label=LABEL,
    training=True
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=LABEL,
    training=False,
    encoder=encoder,
    lb=lb,
)

model = train_model(X_train, y_train)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
save_model(encoder, encoder_path)
lb_path = os.path.join(MODEL_DIR, "lb.pkl")
save_model(lb, lb_path)

model = load_model(
    model_path
)

preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the
# performance_on_categorical_slice function
# Clear the file first before appending new output
if os.path.exists("slice_output.txt"):
    os.remove("slice_output.txt")

# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]

        p_slice, r_slice, fb_slice = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label=LABEL,
            encoder=encoder,
            lb=lb,
            model=model
        )

        with open("slice_output.txt", "a") as f:
            f.write(
                f"--- Slice: {col}='{slicevalue}' "
                f"(Count: {count:,}) ---\n"
            )
            f.write(
                f"Precision: {p_slice:.4f} | Recall: {r_slice:.4f} | "
                f"F1: {fb_slice:.4f}\n\n"
            )
            print(
                f"Slice: {col}='{slicevalue}' | Precision: {p_slice:.4f} | "
                f"Recall: {r_slice:.4f} | F1: {fb_slice:.4f}"
            )
