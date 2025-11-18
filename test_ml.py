import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from ml.data import process_data
from ml.model import train_model, inference


# Fixture to provide a small, representative sample
# DataFrame for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'age': [30, 40, 50, 60],
        'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'Federal-gov'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Doctorate'],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married',
                           'Married-civ-spouse'],
        'occupation': ['Exec-managerial', 'Craft-repair', 'Prof-specialty',
                       'Prof-specialty'],
        'relationship': ['Husband', 'Not-in-family', 'Not-in-family', 'Wife'],
        'race': ['White', 'Black', 'Asian-Pac-Islander', 'White'],
        'sex': ['Male', 'Male', 'Female', 'Female'],
        'hours-per-week': [40, 45, 20, 50],
        'native-country': ['United-States', 'United-States', 'India',
                           'Germany'],
        'salary': ['>50K', '<=50K', '>50K', '>50K']
    })
    return data


# TODO: implement the first test. Change the function name and input as needed
def test_data_integrity(sample_data):
    """
    Test 1: Ensures process_data returns correct object types and preserves
    row count.
    """
    categorical_features = [
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Check the type of returned objects
    assert isinstance(X, np.ndarray)
    assert isinstance(lb, LabelBinarizer)

    # Check that the number of rows is preserved (4 original rows)
    assert X.shape[0] == 4


# TODO: implement the second test. Change the function name and input as needed
def test_label_binarization(sample_data):
    """
    Test 2: Ensures process_data correctly binarizes the 'salary' label
    (contains only 0s and 1s).
    """
    categorical_features = [
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Check that the label array is binary (contains only 0s and 1s)
    assert np.array_equal(y, np.array([1, 0, 1, 1]))
    assert y.ndim == 1


# TODO: implement the third test. Change the function name and input as needed
def test_model_inference_output(sample_data):
    """
    Test 3: Ensures model inference runs without errors and returns
    binary predictions.
    """
    categorical_features = [
        "workclass", "education", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country",
    ]

    # Prepare the data
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Train the model
    model = train_model(X, y)

    # Run inference
    preds = inference(model, X)

    # Check that predictions array is the correct shape and contains
    # binary values
    assert preds.shape[0] == 4
    assert np.issubdtype(preds.dtype, np.integer)
    assert np.all(np.logical_or(preds == 0, preds == 1))