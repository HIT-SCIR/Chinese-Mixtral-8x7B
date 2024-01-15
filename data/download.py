import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="Skywork/SkyPile-150B",
    repo_type="dataset",
    local_dir="./data/zh/SkyPile-150B-2023",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns="data/2023*",
)

# snapshot_download(
#     repo_id="DKYoon/SlimPajama-6B",
#     repo_type="dataset",
#     local_dir="./data/en/DKYoon-SlimPajama-6B",
#     local_dir_use_symlinks=False,
#     resume_download=True,
# )
