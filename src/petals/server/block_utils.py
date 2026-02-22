import inspect
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel

from petals.models.mixtral.block import WrappedMixtralBlock
from petals.utils.convert_block import QuantType
from petals.utils.misc import get_size_in_bytes


def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it using BloomConfig. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    if config.torch_dtype not in ("auto", None, torch.float32):
        # If config specifies float32, we override it to the default dtype below
        return config.torch_dtype
    return torch.bfloat16


def get_block_size(
    config: PretrainedConfig,
    location: str,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    quant_type: QuantType = QuantType.NONE,
    eps: float = 0.01,  # eps accounts for ~1% of metainfo for tensor descriptions, quantization tables, etc.
) -> int:
    if location == "memory":
        assert (
            dtype is not None and quant_type is not None
        ), 'get_block_size(..., location="memory") requires to specify dtype and quant_type for calculations'

    with init_empty_weights(include_buffers=False):
        block = get_model_block(config)
        n_params = sum(param.numel() for param in block.parameters())

    if location == "memory":
        if quant_type == QuantType.NONE:
            dtype = resolve_block_dtype(config, dtype)
            bytes_per_value = get_size_in_bytes(dtype)
        elif quant_type == QuantType.INT8:
            bytes_per_value = 1
        elif quant_type == QuantType.NF4:
            bytes_per_value = 4.25 / 8  # Bitness of NF4 with this config (measured empirically)
        else:
            raise ValueError(f"Unsupported quant_type={quant_type}")
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)

    return round(n_params * bytes_per_value * (1 + eps))


def _accepts_layer_idx(cls):
    """Check if block class constructor accepts layer_idx."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None  # Can't determine — caller should try/except
    params = sig.parameters
    if "layer_idx" in params:
        return True
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return True
    return False


def get_model_block(config, layer_idx: int = 0):
    """
    Create a model block based on the block class, passing layer_idx
    to all block constructors that accept it.
    """
    if config.block_class == WrappedMixtralBlock:
        if hasattr(PreTrainedModel, "_autoset_attn_implementation"):
            config = PreTrainedModel._autoset_attn_implementation(config)
        elif getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"

    accepts = _accepts_layer_idx(config.block_class)
    if accepts is True:
        try:
            return config.block_class(config, layer_idx=layer_idx)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg and "layer_idx" in msg:
                return config.block_class(config)
            raise
    elif accepts is False:
        return config.block_class(config)
    else:
        # Signature inspection failed — try with layer_idx, fallback without
        try:
            return config.block_class(config, layer_idx=layer_idx)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg and "layer_idx" in msg:
                return config.block_class(config)
            raise


get_model_block._universal_layer_idx = True
