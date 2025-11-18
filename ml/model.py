import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained ML model using precision, recall, and F1.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained ML model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    joblib.dump(model, path)


def load_model(path):
    """Loads pickle file from `path` and returns it."""
    return joblib.load(path)


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Computes metrics on a slice of the data specified by a column and value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing features and label. Columns in
        `categorical_features`.
    column_name : str
        Column containing the sliced feature.
    slice_value : str | int | float
        Value of the slice feature.
    categorical_features : list
        Names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn OneHotEncoder
        Trained OneHotEncoder (when training=False).
    lb : sklearn LabelBinarizer
        Trained LabelBinarizer (when training=False).
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    data_slice = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
