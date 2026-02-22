"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
import json
import time
from contextlib import suppress
from typing import Dict, Optional, Union

import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from hivemind.utils.logging import get_logger
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel
try:
    from transformers.utils import get_file_from_repo
except ImportError:
    try:
        from transformers.utils import cached_file
    except ImportError:
        from transformers.utils.hub import cached_file

    def get_file_from_repo(path_or_repo, filename, **kwargs):
        # get_file_from_repo export removed in transformers 4.50+,
        # function removed by ~4.52. It was a thin wrapper around
        # cached_file with exceptions suppressed.
        token = kwargs.pop("token", None)
        legacy = kwargs.pop("use_auth_token", None)
        if token is None:
            token = legacy
        return cached_file(
            path_or_repo,
            filename,
            token=token,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **kwargs,
        )

from petals.constants import DTYPE_MAP
from petals.models.mixtral import WrappedMixtralBlock
from petals.server.block_utils import get_model_block, resolve_block_dtype
from petals.utils.auto_config import AutoDistributedConfig
from petals.utils.disk_cache import DEFAULT_CACHE_DIR, allow_cache_reads, allow_cache_writes, free_disk_space_for
from petals.utils.hf_auth import always_needs_auth

logger = get_logger(__name__)

_MXFP4_EXPERT_PAIRS = [
    ("mlp.experts.gate_up_proj", "_blocks", "_scales"),
    ("mlp.experts.down_proj", "_blocks", "_scales"),
]


def _synthesize_mxfp4_state_dict(state_dict, config=None):
    """Materialize dense expert params from MXFP4-packed keys in state dict.

    GPT-OSS safetensors store MXFP4-packed keys (*_blocks/*_scales) instead of
    dense parameter names. This converts them to dense tensors so
    load_pretrained_block can find the expected keys.

    Modifies state_dict in place. Returns list of converted key names.
    """
    pairs_to_convert = []
    for base, blk_sfx, scl_sfx in _MXFP4_EXPERT_PAIRS:
        blk_key = base + blk_sfx
        scl_key = base + scl_sfx
        if blk_key in state_dict and scl_key in state_dict:
            pairs_to_convert.append((base, blk_key, scl_key))

    if not pairs_to_convert:
        return []

    gate_bases = [b for b, _, _ in pairs_to_convert if "gate_up_proj" in b]
    if not gate_bases:
        raise RuntimeError(
            "Cannot determine MXFP4 layout without gate_up_proj — "
            "file an issue with your model name and transformers version"
        )

    hidden_size = getattr(config, "hidden_size", None) if config else None
    intermediate_size = getattr(config, "intermediate_size", None) if config else None

    needs_transpose = None
    _convert_fn = None

    converted = []
    for base, blk_key, scl_key in pairs_to_convert:
        if base in state_dict:
            logger.debug("Skipping MXFP4 synthesis for %s: dense key already present", base)
            del state_dict[blk_key]
            del state_dict[scl_key]
            converted.append(base)
            continue

        blocks = state_dict[blk_key]
        scales = state_dict[scl_key]

        if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
            raise RuntimeError(
                f"MXFP4 dtype mismatch for {base}: "
                f"blocks.dtype={blocks.dtype}, scales.dtype={scales.dtype} "
                f"(expected both torch.uint8)"
            )

        if _convert_fn is None:
            try:
                from transformers.integrations.mxfp4 import convert_moe_packed_tensors
                _convert_fn = convert_moe_packed_tensors
            except (ImportError, ModuleNotFoundError):
                raise RuntimeError(
                    "MXFP4 packed keys detected in state dict but "
                    "transformers.integrations.mxfp4.convert_moe_packed_tensors not available. "
                    "This requires transformers>=4.55.1 with MXFP4 support."
                )

        try:
            dense = _convert_fn(blocks, scales)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"MXFP4 conversion OOM for {base} — try "
                f"CUDA_VISIBLE_DEVICES='' to force CPU conversion, "
                f"or free GPU memory"
            )

        if "gate_up_proj" in base and needs_transpose is None:
            if (hidden_size is not None and intermediate_size is not None
                    and dense.ndim == 3):
                expected_d1 = hidden_size
                expected_d2 = 2 * intermediate_size
                if dense.shape[1] == expected_d1 and dense.shape[2] == expected_d2:
                    needs_transpose = False
                elif dense.shape[1] == expected_d2 and dense.shape[2] == expected_d1:
                    needs_transpose = True
                else:
                    logger.warning(
                        "MXFP4 gate_up_proj shape %s doesn't match expected dims — skipping transpose",
                        dense.shape,
                    )
                    needs_transpose = False
            else:
                needs_transpose = False
        elif "gate_up_proj" not in base and needs_transpose is None:
            raise RuntimeError(
                "Cannot determine MXFP4 layout without gate_up_proj — "
                "file an issue with your model name and transformers version"
            )

        if needs_transpose and dense.ndim == 3:
            dense = dense.transpose(1, 2).contiguous()

        state_dict[base] = dense
        del state_dict[blk_key]
        del state_dict[scl_key]

        logger.info("MXFP4: synthesized %s (%s)", base.rsplit(".", 1)[-1], dense.dtype)
        converted.append(base)

    return converted


def load_pretrained_block(
    model_name: str,
    block_index: int,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    if config is None:
        config = AutoDistributedConfig.from_pretrained(model_name, use_auth_token=token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    torch_dtype = resolve_block_dtype(config, torch_dtype)

    with init_empty_weights():
        block = get_model_block(config, layer_idx=block_index)

    block_prefix = f"{config.block_prefix}.{block_index}."
    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    _synthesize_mxfp4_state_dict(state_dict, config=config)

    for param_name, _ in block.named_parameters():
        assert param_name in state_dict, f"{param_name} not in state dict"
        param = state_dict[param_name]
        if not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            param = param.to(torch_dtype)
        set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)

    logger.info(f"Loaded {model_name} block {block_index}")
    return block


StateDict = Dict[str, torch.Tensor]


def _load_state_dict_from_repo(
    model_name: str,
    block_prefix: str,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
) -> StateDict:
    if always_needs_auth(model_name) and token is None:
        token = True

    index_file = _find_index_file(model_name, revision=revision, token=token, cache_dir=cache_dir)
    if index_file.endswith(".index.json"):  # Sharded model
        path = get_file_from_repo(model_name, filename=index_file, use_auth_token=token, cache_dir=cache_dir)
        if path is None:
            # _find_index_file() told that a file exists but we can't get it (e.g., it just disappeared)
            raise ValueError(f"Failed to get file {index_file}")

        with open(path) as f:
            index = json.load(f)
        filenames = {
            filename for param_name, filename in index["weight_map"].items() if param_name.startswith(block_prefix)
        }
        if not filenames:
            raise RuntimeError(f"Block {block_prefix}* not found in the index: {index['weight_map']}")
    else:  # Non-sharded model
        filenames = {index_file}
    logger.debug(f"Loading {block_prefix}* from {filenames}")

    state_dict = {}
    for filename in filenames:
        shard_state_dict = _load_state_dict_from_repo_file(
            model_name,
            filename,
            block_prefix=block_prefix,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            max_disk_space=max_disk_space,
        )
        shard_state_dict = {
            param_name[len(block_prefix) :]: param
            for param_name, param in shard_state_dict.items()
            if param_name.startswith(block_prefix)
        }  # Remove unused parameters from memory
        state_dict.update(shard_state_dict)
    return state_dict


INDEX_FILES = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.bin.index.json", "pytorch_model.bin"]


def _find_index_file(
    model_name: str, *, revision: Optional[str] = None, token: Optional[Union[str, bool]] = None, cache_dir: str
) -> str:
    # If we have cached weights (e.g., Pickle from older Petals versions), reuse them
    for filename in INDEX_FILES:
        path = get_file_from_repo(
            model_name,
            filename,
            revision=revision,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if path is not None:
            return filename

    # If we don't, prefer Safetensors when possible
    # (we don't download files here since we can't account for max_disk_space in case of large files)
    for filename in INDEX_FILES:
        with suppress(EntryNotFoundError):
            get_hf_file_metadata(hf_hub_url(model_name, filename, revision=revision), token=token)
            return filename

    raise ValueError(
        f"Repo {model_name} does not contain weights in a supported format: files {INDEX_FILES} do not exist"
    )


def _load_state_dict_from_repo_file(
    model_name: str,
    filename: str,
    *,
    block_prefix: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
) -> StateDict:
    # First, try to find the weights locally
    try:
        with allow_cache_reads(cache_dir):
            path = get_file_from_repo(
                model_name,
                filename,
                revision=revision,
                use_auth_token=token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if path is not None:
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
    except Exception:
        logger.warning(f"Cache for file {filename} is corrupted, it will be downloaded again", exc_info=True)

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    while True:
        try:
            with allow_cache_writes(cache_dir):
                url = hf_hub_url(model_name, filename, revision=revision)
                file_size = get_hf_file_metadata(url, token=token).size
                if file_size is not None:
                    free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size of file {filename} from repo {model_name}")

                path = get_file_from_repo(
                    model_name,
                    filename,
                    revision=revision,
                    use_auth_token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                if path is None:
                    raise RuntimeError(f"File {filename} does not exist in repo {model_name}")
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
        except Exception as e:
            logger.warning(f"Failed to load file {filename} from HF Hub (retry in {delay:.0f} sec)", exc_info=True)
            time.sleep(delay)


def _load_state_dict_from_local_file(path: str, *, block_prefix: Optional[str] = None) -> StateDict:
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")

    if path.endswith(".safetensors"):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys() if block_prefix is None or key.startswith(block_prefix)}

    raise ValueError(f"Unknown weight format: {path}")
