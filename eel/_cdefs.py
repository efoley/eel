import ctypes


class Config(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("ffn_hidden_size", ctypes.c_int),
        ("head_size", ctypes.c_int),
        ("num_q_heads", ctypes.c_int),
        ("num_kv_heads", ctypes.c_int),
        ("max_seq_len", ctypes.c_int),
        ("num_layers", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("norm_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("rope_size", ctypes.c_int),
    ]


class InferState(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.POINTER(ctypes.c_float)),
        ("x2", ctypes.POINTER(ctypes.c_float)),
        ("x3", ctypes.POINTER(ctypes.c_float)),
        ("h1", ctypes.POINTER(ctypes.c_float)),
        ("h2", ctypes.POINTER(ctypes.c_float)),
        ("h3", ctypes.POINTER(ctypes.c_float)),
        ("k_cache", ctypes.POINTER(ctypes.c_float)),
        ("v_cache", ctypes.POINTER(ctypes.c_float)),
        ("q", ctypes.POINTER(ctypes.c_float)),
        ("k", ctypes.POINTER(ctypes.c_float)),
        ("v", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("mha_out", ctypes.POINTER(ctypes.c_float)),
        ("logits", ctypes.POINTER(ctypes.c_float)),
    ]


class LayerWeights(ctypes.Structure):
    _fields_ = [
        ("rms_weight_attn", ctypes.POINTER(ctypes.c_float)),
        ("rms_weight_ffn", ctypes.POINTER(ctypes.c_float)),
        ("q_proj", ctypes.POINTER(ctypes.c_float)),
        ("k_proj", ctypes.POINTER(ctypes.c_float)),
        ("v_proj", ctypes.POINTER(ctypes.c_float)),
        ("o_proj", ctypes.POINTER(ctypes.c_float)),
        ("q_bias", ctypes.POINTER(ctypes.c_float)),
        ("k_bias", ctypes.POINTER(ctypes.c_float)),
        ("v_bias", ctypes.POINTER(ctypes.c_float)),
        ("ffn_in_proj", ctypes.POINTER(ctypes.c_float)),
        ("ffn_gate_proj", ctypes.POINTER(ctypes.c_float)),
        ("ffn_out_proj", ctypes.POINTER(ctypes.c_float)),
    ]


class Weights(ctypes.Structure):
    _fields_ = [
        ("token_embedding", ctypes.POINTER(ctypes.c_float)),
        ("rms_weight", ctypes.POINTER(ctypes.c_float)),
        ("layer", ctypes.POINTER(ctypes.POINTER(LayerWeights))),
        # NOTE: new fields from C go above this line
        (
            "_owned_py_obj",
            ctypes.py_object,
        ),  # not in C struct; holds reference to python object owning buffers
    ]


class Model(ctypes.Structure):
    _fields_ = [
        ("config", ctypes.POINTER(Config)),
        ("weights", ctypes.POINTER(Weights)),
    ]
