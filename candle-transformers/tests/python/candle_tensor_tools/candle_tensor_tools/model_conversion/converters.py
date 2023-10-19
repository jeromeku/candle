import os
from typing import Callable, Optional

import torch
from huggingface_hub import hf_hub_download, login
from rich import print
from rich.progress import track
from rich.traceback import install
from safetensors.torch import save_file

from ..model_checking import find_pytorch_files, find_safetensor_files, hf_hub_download

install()


def _make_tensors_contiguous(model: dict):
    """Makes tensors contiguous in-place"""

    for k, v in model.items():
        if isinstance(v, torch.Tensor):
            model[k] = v.contiguous()


def _convert_single_file(in_path, out_path, normalization_func):
    """Converts a single PyTorch file to SafeTensor format"""

    model = torch.load(in_path)
    if normalization_func is not None:
        model = {normalization_func(k): v for k, v in model.items()}
    try:
        save_file(model, out_path)
    except ValueError as e:
        _make_tensors_contiguous(model)
        save_file(model, out_path)


def _convert_pytorch_to_safetensor_files(pt_file_paths, outdir, normalization_func):
    """Converts a list of PyTorch files to SafeTensor format"""
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for pt_file_path in track(
        pt_file_paths,
        description="Converting [yellow]pytorch pickle files[/yellow] to [green]safetensors[/green]...\n",
    ):
        out_file = os.path.splitext(os.path.basename(pt_file_path))[0] + ".safetensors"
        out_path = os.path.join(outdir, out_file)
        _convert_single_file(pt_file_path, out_path, normalization_func)


def _upload_to_hub(upload_dir, repo_id, repo_type="model", hf_token=None):
    """Uploads a directory to HuggingFace Hub"""
    from huggingface_hub import HfApi

    print(f"Uploading safetensors to [blue]{repo_id}[/blue] on 🤗 HuggingFace Hub 🤗")
    token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not token:
        login()
    else:
        login(token=token)

    api = HfApi()

    if not api.repo_exists(repo_id):
        api.create_repo(repo_id, repo_type=repo_type)

    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type=repo_type,
    )


def convert_pytorch_to_safetensors(
    repo: str,
    outdir: Optional[str] = None,
    normalization_func: Optional[Callable[[str], str]] = None,
    force: bool = False,
    upload_to_hub: bool = False,
    repo_id: Optional[str] = None,
):
    """Converts pickled PyTorch model files to SafeTensor format

    Downloads the pytorch files from HF Hub, converts them to safetensor format
    and uploads them to HF Hub (optional).

    Args:
        repo (str): HuggingFace Hub repo name
        outdir (str): Directory to save converted files.  Defaults to model name.
        normalization_func (Callable[[str],str]): Function that takes pytorch weight key and returns a normalized key to use in the resulting safetensors weight map.
        force (bool): If True, converts files even if SafeTensor files already exist in repo.
        upload_to_hub (bool): If True, uploads converted files to HF Hub.
        repo_id (str): If upload_to_hub is True, repo_id is the name of the repo to upload to. If repo does not already exist on HF Hub, it will be created.
    """
    model_name = repo.split("/")[-1]
    outdir = outdir or model_name

    safetensor_files = find_safetensor_files(repo)
    if len(safetensor_files) > 0 and not force:
        raise ValueError(
            f"Safe tensor files already found in {repo}.  Call again with `force=True` to force conversion."
        )
    pytorch_files = find_pytorch_files(repo)
    if len(pytorch_files) == 0:
        raise ValueError(f"No PyTorch files found in {repo}.")

    print(f"Found following pytorch pickle files in [blue]{repo}:")
    for pt_file in pytorch_files:
        print(f"  [italic]{pt_file}")
    print("")
    pt_paths = [hf_hub_download(repo, os.path.basename(f)) for f in pytorch_files]
    _convert_pytorch_to_safetensor_files(pt_paths, outdir, normalization_func)

    print(f"Saved safetensors to [bold]{outdir}")

    if upload_to_hub:
        assert repo_id is not None, "repo_id must be provided if upload_to_hub is True"
        _upload_to_hub(outdir, repo_id)

    url = f"https://huggingface.co/{repo_id}"
    print("")
    print("[bold]Finished processing![/bold]")
    print(f"Check [bold blue][link={url}]{url}[/link][/bold blue] for uploaded files!")
    return outdir
