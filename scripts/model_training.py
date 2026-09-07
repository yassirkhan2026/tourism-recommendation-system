
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, hf_hub_download

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_REPO = "yassirkhan/tourism"
MODEL_REPO = "yassirkhan/tourism-recommendation-model"

MODEL_PATH = "model_building/tourism_recommendation_model.pkl"


# ---------------------------------------------------------
# Get Hugging Face token
# ---------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")


# ---------------------------------------------------------
# Download train and test datasets from Hugging Face
# ---------------------------------------------------------

train_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="train.csv",
    repo_type="dataset",
    token=HF_TOKEN
)

test_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="test.csv",
    repo_type="dataset",
    token=HF_TOKEN
)


# ---------------------------------------------------------
# Load train and test datasets
# ---------------------------------------------------------

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X_train = train_data.drop(columns=["ProdTaken"])
y_train = train_data["ProdTaken"]

X_test = test_data.drop(columns=["ProdTaken"])
y_test = test_data["ProdTaken"]


# ---------------------------------------------------------
# Identify categorical and numerical features
# ---------------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------------------------
# Random Forest model
# ---------------------------------------------------------

rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)


# ---------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)


# ---------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


# ---------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


# ---------------------------------------------------------
# Best model
# ---------------------------------------------------------

best_model = grid_search.best_estimator_

print("Best parameters:")
print(grid_search.best_params_)

print("Best CV F1 score:")
print(grid_search.best_score_)


# ---------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)


# ---------------------------------------------------------
# MLflow experiment
# ---------------------------------------------------------

mlflow.set_experiment("Tourism_Recommendation_Model")


with mlflow.start_run() as run:

    # Log tuned parameters
    mlflow.log_params(grid_search.best_params_)

    # Log CV score
    mlflow.log_metric(
        "cv_f1_score",
        grid_search.best_score_
    )

    # Log test metrics
    mlflow.log_metric(
        "test_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "test_precision",
        precision
    )

    mlflow.log_metric(
        "test_recall",
        recall
    )

    mlflow.log_metric(
        "test_f1_score",
        f1
    )

    # Log model using cloudpickle
    mlflow.sklearn.log_model(
        best_model,
        name="random_forest_model",
        serialization_format="cloudpickle"
    )

    run_id = run.info.run_id

    print("MLflow run completed.")
    print("Run ID:", run_id)


# ---------------------------------------------------------
# Save model locally
# ---------------------------------------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("Model saved successfully:", MODEL_PATH)


# ---------------------------------------------------------
# Upload trained model to Hugging Face Model Hub
# ---------------------------------------------------------

api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="tourism_recommendation_model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Trained model uploaded successfully to Hugging Face Model Hub.")
