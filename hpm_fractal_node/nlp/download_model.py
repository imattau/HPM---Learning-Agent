"""
Download TinyLlama-1.1B-Chat GGUF model to data/models/.

Run once:
    PYTHONPATH=. python3 hpm_fractal_node/nlp/download_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LOCAL_DIR = Path(__file__).parents[2] / "data" / "models"


def download() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_DIR / FILENAME
    if dest.exists():
        print(f"Model already exists at {dest}")
        return dest

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
    )
    print(f"Saved to {path}")
    return Path(path)


def model_path() -> Path:
    return LOCAL_DIR / FILENAME


if __name__ == "__main__":
    download()
