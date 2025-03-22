import ctypes
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch

from ._cdefs import Config, InferState, LayerWeights, Model, Weights


RootDir = Path(__file__).parent.parent

if platform.system() == "Darwin":
    LibPath = RootDir / "build" / "libeel.dylib"
else:
    LibPath = RootDir / "build" / "libeel.so"

lib = ctypes.CDLL(LibPath)

def _deref(ptr: ctypes.POINTER):
    """
    Utility because I hate the name "contents" here.
    """
    return ptr.contents


def load_config(config_path: str) -> Config:
    """
    Create an eel config from the HF config.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    head_size = config["hidden_size"] // config["num_attention_heads"]

    # some models apply rope only to a subset of the head vector
    if "partial_rotary_factor" in config:
        rope_size = head_size * config["partial_rotary_factor"]
    else:
        rope_size = head_size

    return Config(
        size=config["hidden_size"],
        ffn_hidden_size=config["intermediate_size"],
        head_size=head_size,
        num_q_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
        max_seq_len=config["max_position_embeddings"],
        num_layers=config["num_hidden_layers"],
        vocab_size=config["vocab_size"],
        norm_eps=config["rms_norm_eps"],
        rope_theta=config["rope_theta"],
        rope_size=rope_size,
        tie_embeddings=int(config["tie_word_embeddings"]),
    )


# Stolen from https://github.com/zeux/calm/blob/86dfb56daba5052c260a2dd86d296309cfbd4324/tools/convert.py#L223
# huggingface permutes WQ and WK, this function reverses it
# see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
def _permute_rope(w, num_heads, rope_size):
    head_size = w.shape[0] // num_heads
    assert rope_size <= head_size
    w = torch.unflatten(w, 0, (-1, head_size))
    # wr is the rotary part, wk is the part kept unrotated
    wr = w[:, :rope_size]
    wk = w[:, rope_size:]
    # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
    wr = torch.unflatten(wr, 1, (2, -1))
    wr = wr.transpose(1, 2)
    wr = wr.flatten(1, 2)
    # assemble the heads back
    w = torch.cat([wr, wk], dim=1)
    return torch.flatten(w, 0, 1)


def _permute_rope_bias(w, num_heads, rope_size):
    head_size = w.shape[0] // num_heads
    assert rope_size <= head_size
    w = torch.unflatten(w, 0, (-1, head_size))
    # wr is the rotary part, wk is the part kept unrotated
    wr = w[:rope_size]
    wk = w[rope_size:]
    # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
    wr = torch.unflatten(wr, 1, (2, -1))
    wr = wr.transpose(1, 2)
    wr = wr.flatten(1, 2)
    # assemble the heads back
    w = torch.cat([wr, wk], dim=0)
    return torch.flatten(w, 0, 1)


def load_weights(config: Config, weights_path: str, verbose: bool = False) -> Weights:
    """
    Load weights from the HF safetensor file.
    """
    st_weights = {}
    with safetensors.safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            st_weights[key] = f.get_tensor(key)

    if verbose:
        for key, tensor in st_weights.items():
            shape = tensor.shape
            packed_stride = np.cumprod(np.r_[1, shape[::-1]])[:-1][::-1]

            is_packed = np.all(packed_stride == tensor.stride())

            print(f"{key}: {tensor.shape}, {tensor.dtype}, {is_packed}")

    weights = Weights()

    owned_tensors = []

    def conv(
        tensor: torch.Tensor, dtype: torch.dtype, expect_size: tuple[int] | None = None, check_nans: bool = True,
    ) -> ctypes.POINTER:
        if expect_size is not None and tensor.shape != expect_size:
            raise ValueError(
                f"Tensor has unexpected shape: {tensor.shape}, expected: {expect_size}"
            )

        tensor = tensor.to(dtype)

        if check_nans and np.isnan(tensor).any():
            raise ValueError("Tensor contains NaN values")

        # ensure that tensor is packed
        shape = tensor.shape
        packed_stride = np.cumprod(np.r_[1, shape[::-1]])[:-1][::-1]
        is_packed = np.all(packed_stride == tensor.stride())
        assert is_packed
        # if not is_packed:
        #     tensor = torch.as_strided(tensor, shape=shape, stride=packed_stride)

        owned_tensors.append(tensor)

        return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))

    dtype = torch.float32
    weights.embedding_in_table = conv(
        st_weights.get("model.embed_tokens.weight", None),
        dtype,
        (config.vocab_size, config.size),
    )

    if config.tie_embeddings:
        weights.embedding_out_proj = weights.embedding_in_table
    else:
        weights.embedding_out_proj = conv(
            st_weights["model.output.weight"],
            dtype,
            (config.vocab_size, config.size),
        )

    weights.rms_weight = conv(
        st_weights.get("model.norm.weight", None), dtype, (config.size,)
    )
    layer_array = (LayerWeights * config.num_layers)()
    weights.layer = ctypes.cast(
        layer_array, ctypes.POINTER(ctypes.POINTER(LayerWeights))
    )

    for layer_idx in range(config.num_layers):
        layer = LayerWeights()
        weights.layer[layer_idx] = ctypes.pointer(layer)

        layer.rms_weight_attn = conv(
            st_weights[f"model.layers.{layer_idx}.input_layernorm.weight"],
            dtype,
            (config.size,),
        )
        layer.rms_weight_ffn = conv(
            st_weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
            dtype,
            (config.size,),
        )

        layer.q_proj = conv(
            _permute_rope(
                st_weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
                config.num_q_heads,
                config.rope_size,
            ),
            dtype,
            (
                config.num_q_heads * config.head_size,
                config.size,
            ),
        )
        layer.k_proj = conv(
            _permute_rope(
                st_weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
                config.num_kv_heads,
                config.rope_size,
            ),
            dtype,
            (
                config.num_kv_heads * config.head_size,
                config.size,
            ),
        )
        layer.v_proj = conv(
            st_weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            dtype,
            (
                config.num_kv_heads * config.head_size,
                config.size,
            ),
        )
        layer.o_proj = conv(
            st_weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            dtype,
            (
                config.size,
                config.num_q_heads * config.head_size,
            ),
        )

        layer.q_bias = conv(
            _permute_rope_bias(
                st_weights[f"model.layers.{layer_idx}.self_attn.q_proj.bias"],
                config.num_q_heads,
                config.rope_size,
            ),
            dtype,
            (config.num_q_heads * config.head_size,),
        )
        layer.k_bias = conv(
            _permute_rope_bias(
                st_weights[f"model.layers.{layer_idx}.self_attn.k_proj.bias"],
                config.num_kv_heads,
                config.rope_size,
            ),
            dtype,
            (config.num_kv_heads * config.head_size,),
        )
        layer.v_bias = conv(
            st_weights[f"model.layers.{layer_idx}.self_attn.v_proj.bias"],
            dtype,
            (config.num_kv_heads * config.head_size,),
        )

        layer.ffn_in_proj = conv(
            st_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            dtype,
            (
                config.ffn_hidden_size,
                config.size,
            ),
        )
        layer.ffn_gate_proj = conv(
            st_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            dtype,
            (
                config.ffn_hidden_size,
                config.size,
            ),
        )
        layer.ffn_out_proj = conv(
            st_weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            dtype,
            (
                config.size,
                config.ffn_hidden_size,
            ),
        )

    weights._owned_py_obj = owned_tensors

    return weights


def load_model(config: Config, weights: Weights) -> Model:
    model = Model(config=ctypes.pointer(config), weights=ctypes.pointer(weights))
    return model


def init_state(config: Config) -> InferState:
    lib.make_state.argtypes = (
        ctypes.POINTER(Config),
    )
    lib.make_state.restype = ctypes.POINTER(InferState)

    state_ptr = lib.make_state(ctypes.pointer(config))
    return _deref(state_ptr)


def forward(model: Model, state: InferState, token: int, pos: int) -> np.array:
    """
    Run the model one step forward and return logits.
    """
    lib.forward.argtypes = (
        ctypes.POINTER(Model),
        ctypes.POINTER(InferState),
        ctypes.c_int,
        ctypes.c_int,
    )
    lib.forward.restype = ctypes.POINTER(ctypes.c_float)

    logits_ptr = lib.forward(ctypes.pointer(model), ctypes.pointer(state), token, pos)

    arr = np.ctypeslib.as_array(
        logits_ptr, shape=(_deref(model.config).vocab_size,)
    )

    assert arr.dtype == np.float32

    return arr
