
from huggingface_hub import HfApi
import os


# Hugging Face Dataset repository
DATASET_REPO = "yassirkhan/tourism"

# Dataset file in the GitHub repository
DATASET_FILE = "data/tourism.csv"


# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")


# Authenticate with Hugging Face
api = HfApi(token=HF_TOKEN)


# Upload the raw tourism dataset to Hugging Face
api.upload_file(
    path_or_fileobj=DATASET_FILE,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print("Raw tourism dataset uploaded successfully to Hugging Face.")
