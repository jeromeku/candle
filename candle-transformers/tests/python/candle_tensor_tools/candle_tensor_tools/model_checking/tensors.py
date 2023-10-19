import itertools
import json
import os
import struct
from ast import literal_eval
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import jsonlines
import requests  # pip install requests
import torch
import tqdm
from huggingface_hub import HfFileSystem, hf_hub_download, hf_hub_url
from safetensors.torch import save_file

# HFHUB_URL_TEMPLATE = "https://huggingface.co/{model_path}/resolve/main/{filename}"


def _check_for_safetensors_index(repo):
    """Returns True if repo has a safetensors_index.json file"""
    return len(find_safetensors_index(repo)) > 0


def download_hfhub_file(repo: str, filename: str):
    """Downloads a file from HuggingFace Hub"""
    save_path = hf_hub_download(repo, filename)
    return save_path


def _search_repo(repo, pat: str = "*"):
    """Returns files in repo matching pattern should be a glob pattern"""
    fs = HfFileSystem()
    matches = fs.glob(f"{repo}/{pat}")
    return matches


def find_pytorch_index(repo):
    """Returns list of pytorch_index files in repo"""
    return _search_repo(repo, "*pytorch*index.json")


def find_safetensors_index(repo, pat="*safetensors*index.json"):
    """Returns list of safetensors_index files in repo"""
    return _search_repo(repo, pat)


def find_safetensor_files(repo, pat="*.safetensors"):
    """Returns list of SafeTensor files in repo"""
    return _search_repo(repo, pat)


def find_pytorch_files(repo, pat="*pytorch*.bin"):
    """Returns list of pytorch files in repo"""
    return _search_repo(repo, pat)


# From https://huggingface.co/docs/safetensors/metadata_parsing#python
def _get_safetensor_header(url):
    # Fetch the first 8 bytes of the file
    headers = {"Range": "bytes=0-7"}
    response = requests.get(url, headers=headers)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack("<Q", response.content)[0]
    # Fetch length_of_header bytes starting from the 9th byte
    headers = {"Range": f"bytes=8-{7 + length_of_header}"}
    response = requests.get(url, headers=headers)
    # Interpret the response as a JSON object
    header = response.json()
    return header


def _check_for_pytorch_index(repo):
    """Returns True if repo has a pytorch_index.json file"""
    return len(find_pytorch_index(repo)) > 0


def get_pytorch_index(repo: str):
    """Retrieves PyTorch Index from repo if exists

    Raises ValueError if 1) more than one pytorch index file is found
    or 2) no index file is found

    Args:
        repo (str): HuggingFace Hub repo name
    Returns:
        pytorch model index (dict): mapping of tensors to pickle files

    """
    has_pytorch_index = _check_for_pytorch_index(repo)
    if has_pytorch_index:
        pytorch_index_file = find_pytorch_index(repo)
        if len(pytorch_index_file) > 1:
            raise ValueError(f"Found more than one pytorch_index.json file in {repo}.")
        filepath = download_hfhub_file(repo, os.path.basename(pytorch_index_file[0]))
        return json.load(open(filepath, "r"))
    else:
        raise ValueError(f"No PyTorch index files found in {repo}.")


def get_safetensor_index(repo: str):
    """Retrieves Safetensors Index from repo if exists

    Raises ValueError if 1) more than one safetensors_index.json file is found
    or 2) no safetensors_index.json file is found

    Args:
        repo (str): HuggingFace Hub repo name
    Returns:
        safetensors_index (dict): mapping of tensors to safetensor files
    """

    has_safetensors_index = _check_for_safetensors_index(repo)
    if has_safetensors_index:
        safetensors_file = find_safetensors_index(repo)
        if len(safetensors_file) > 1:
            raise ValueError(
                f"Found more than one safetensors_index.json file in {repo}."
            )

        filepath = download_hfhub_file(repo, os.path.basename(safetensors_file[0]))
        return json.load(open(filepath, "r"))
    else:
        raise ValueError(f"No SafeTensor files found in {repo}.")


def get_safetensor_headers(repo: str, merge: bool = True):
    """Retrieves headers for list of SafeTensor files from HuggingFace Hub
    Returns a dictionary of {url: header} where header
    is standard Safetensors [format](https://huggingface.co/docs/safetensors/metadata_parsing#example-output)

    Checks if a safetensors.index.json file exists and returns parsed json if so
    Otherwise, downloads each SafeTensor file and parses the header
    If neither safetensors.index.json nor SafeTensor files exist, returns empty dictionary

    Args:
        repo (str): HuggingFace Hub repo name
        merge (bool): If True, returns a ChainMap (aggregated dict) of headers from all SafeTensor files.
        Only applicable if safetensors.index.json does not exist.

    Returns:
        headers (dict): Dictionary of {url: header} where header is a dict.
    """
    safetensor_files = find_safetensor_files(repo)
    if len(safetensor_files) == 0:
        raise ValueError(f"No SafeTensor files found in {repo}.")
    urls = [hf_hub_url(repo, os.path.basename(f)) for f in safetensor_files]
    headers = {url: _get_safetensor_header(url) for url in urls}
    if merge:
        merged_tensor_map_iterable = itertools.chain.from_iterable(
            h.items() for h in headers.values()
        )
        headers = dict(merged_tensor_map_iterable)

    if "__metadata__" in headers:
        del headers["__metadata__"]

    return dict(sorted(headers.items()))


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
    for pt_file_path in tqdm.tqdm(
        pt_file_paths, desc="Converting pytorch files to safetensor format"
    ):
        out_file = os.path.splitext(os.path.basename(pt_file_path))[0] + ".safetensors"
        out_path = os.path.join(outdir, out_file)
        _convert_single_file(pt_file_path, out_path, normalization_func)


def _upload_to_hub(upload_dir, repo_id, repo_type="model", hf_token=None):
    """Uploads a directory to HuggingFace Hub"""
    from huggingface_hub import HfApi

    token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not token:
        from huggingface_hub import login

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
    outdir = outdir or repo.split("/")[-1]

    safetensor_files = find_safetensor_files(repo)
    if len(safetensor_files) > 0 and not force:
        raise ValueError(
            f"Safe tensor files already found in repo.  Call again with `force=True` to force conversion."
        )
    pytorch_files = find_pytorch_files(repo)
    if len(pytorch_files) == 0:
        raise ValueError(f"No PyTorch files found in {repo}.")

    pt_paths = [hf_hub_download(repo, os.path.basename(f)) for f in pytorch_files]
    _convert_pytorch_to_safetensor_files(pt_paths, outdir, normalization_func)

    print(f"Saved safetensors to {outdir}")

    if upload_to_hub:
        assert repo_id is not None, "repo_id must be provided if upload_to_hub is True"
        _upload_to_hub(outdir, repo_id)

    return outdir


_TENSOR_FIELD = "fields"
_EXPECTED_TRACE_KEYS = {"path", "shape", "dtype"}


@dataclass
class CandleTrace(dict):
    path: str
    shape: list[int]
    dtype: str

    def __post_init__(self):
        try:
            shape_lst = literal_eval(self.shape.strip())
        except:
            print(
                f"Failed to parse shape {self.shape} into list of ints, keeping as string"
            )
        finally:
            self.shape = shape_lst
        super().__init__(self, **self.__dict__)


@dataclass
class CandleModelTrace(dict):
    traces: List[CandleTrace]

    def __post_init__(self):
        tensor_map = {
            trace.path: dict(shape=trace.shape, dtype=trace.dtype)
            for trace in self.traces
        }
        super().__init__(self, **tensor_map)

    def __repr__(self):
        return super().__repr__()


def _extract_tensors(
    trace, tensor_field=_TENSOR_FIELD, expected_keys=_EXPECTED_TRACE_KEYS
):
    assert tensor_field in trace, f"Expected {tensor_field} in trace"
    tensor_info = trace[tensor_field]
    assert all(
        key in tensor_info for key in expected_keys
    ), f"Expected keys {expected_keys} in tensor info"
    return tensor_info


def parse_trace(
    trace_file, tensor_field=_TENSOR_FIELD, expected_keys=_EXPECTED_TRACE_KEYS
):
    """Parses a jsonlines trace file into a list of CandleTrace objects

    Args:
        trace_file (str): Path to jsonlines trace file
        tensor_field (str): Field in trace file containing tensor info defaults to "fields"
        expected_keys (set): Expected keys in tensor info defaults to {"path", "shape", "dtype"}
    Returns:
        trace_info (list): List of CandleTrace objects where CandleTrace is a dict / dataclass with fields
        path (str), shape (list[int] | str if cannot be parsed to list[int]), dtype (str)
    """
    reader = jsonlines.Reader(open(trace_file, "r"))
    traces = [
        _extract_tensors(trace, tensor_field=tensor_field, expected_keys=expected_keys)
        for trace in reader
    ]
    trace_info = sorted(
        [CandleTrace(**trace) for trace in traces], key=lambda x: x.path
    )

    return CandleModelTrace(trace_info)


def check_candle_trace(trace_dir, model_id):
    """Checks if traced candle model matches original tensors from HF model

    Checks whether the actual tensors requested during Candle model trace (via Varbuilder `get` or `get_with_hints` methods)
    match the original tensors from the HF model which will be loaded during `VarBuilder::from_mmaped_safetensors` call.

    See candle-transformers/tests/utilities/model_tracing.rs for more details on how the model is traced.

    Args:
        trace_dir (str): Directory containing model trace json
        model_id (str): HF model_id where model is stored

    Returns:
        extra_keys (list): List of extra tensors in traced model
        missing_keys (list): List of missing tensors in traced model
        all_mismatched_keys (list): List of mismatched tensors in traced model
    """
    # Read in trace
    traced_tensors = sorted(json.load(open(trace_dir, "r")))
    # Read in original model
    original_tensor_map = sorted(get_safetensor_headers(model_id))
    extra_keys, missing_keys = [], []

    if len(traced_tensors) > len(original_tensor_map):
        print("WARNING: Traced model has more tensors than original model.")
        extra_keys = set(traced_tensors.keys()).difference(
            set(original_tensor_map.keys())
        )

        print("Extra tensors:")
        for k in extra_keys:
            print(k)

    elif len(traced_tensors) < len(original_tensor_map):
        print("WARNING: Traced model has fewer tensors than original model.")
        missing_keys = set(original_tensor_map.keys()).difference(
            set(traced_tensors.keys())
        )

        print("Missing tensors:")
        for k in missing_keys:
            print(k)

    all_mismatched_keys = set(traced_tensors.keys()).symmetric_difference(
        original_tensor_map.keys()
    )
    if len(all_mismatched_keys) > 0:
        print("WARNING: Traced model has mismatched tensors.")
        print("Mismatched tensors:")
        for k in all_mismatched_keys:
            print(k)

    return extra_keys, missing_keys, all_mismatched_keys
