
from huggingface_hub import HfApi
import os


# Hugging Face Space details
SPACE_REPO = "yassirkhan/tourism-recommendation-space"


# Authenticate using the Hugging Face token
api = HfApi(token=os.environ.get("HF_TOKEN"))


# Upload all deployment files to the Hugging Face Space
api.upload_folder(
    folder_path=".",
    repo_id=SPACE_REPO,
    repo_type="space"
)

print("Deployment files uploaded successfully to Hugging Face Space.")
