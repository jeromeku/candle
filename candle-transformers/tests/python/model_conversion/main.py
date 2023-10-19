from typing import Optional

import fire
from model_checking import convert_pytorch_to_safetensors


def main(
    model_id,
    outdir: str = None,
    force: bool = False,
    upload_to_hub: bool = False,
    repo_id: str = None,
):
    """
    Converts pickled PyTorch model files to SafeTensor format

    Downloads the pytorch files from HF Hub, converts them to safetensor format
    and uploads them to HF Hub (optional).

    Args:
        model_id (str): HuggingFace Hub repo name (e.g., openai/clip-vit-base-patch32)
        outdir (str): Directory to save converted files.  Defaults to model name.
        force (bool): If True, converts files even if SafeTensor files already exist in repo.
        upload_to_hub (bool): If True, uploads converted files to HF Hub.
        repo_id (str): If upload_to_hub is True, repo_id is the name of the repo to upload to. If repo does not already exist on HF Hub, it will be created.
    """
    convert_pytorch_to_safetensors(
        model_id,
        outdir=outdir,
        force=force,
        upload_to_hub=upload_to_hub,
        repo_id=repo_id,
    )


if __name__ == "__main__":
    fire.Fire(main)
