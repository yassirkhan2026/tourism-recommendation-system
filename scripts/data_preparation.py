
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_REPO = "yassirkhan/tourism"
RAW_DATA_PATH = "data/tourism.csv"

TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"


# ---------------------------------------------------------
# Get Hugging Face token
# ---------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")


# ---------------------------------------------------------
# Load raw dataset
# ---------------------------------------------------------

df = pd.read_csv(RAW_DATA_PATH)

print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"])

# Standardize gender values
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Remove duplicate records
df = df.drop_duplicates()

print("Cleaned dataset shape:", df.shape)
print("Remaining duplicates:", df.duplicated().sum())


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]


# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Create train and test datasets
# ---------------------------------------------------------

train_data = X_train.copy()
train_data["ProdTaken"] = y_train

test_data = X_test.copy()
test_data["ProdTaken"] = y_test


print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)


# ---------------------------------------------------------
# Save train and test datasets
# ---------------------------------------------------------

train_data.to_csv(TRAIN_DATA_PATH, index=False)
test_data.to_csv(TEST_DATA_PATH, index=False)

print("Train and test datasets saved successfully.")


# ---------------------------------------------------------
# Upload train and test datasets to Hugging Face
# ---------------------------------------------------------

api = HfApi(token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=TRAIN_DATA_PATH,
    path_in_repo="train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj=TEST_DATA_PATH,
    path_in_repo="test.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print("Train and test datasets uploaded successfully to Hugging Face.")
