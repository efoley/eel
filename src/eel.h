#pragma once

#define MAX_LAYERS 32;

struct Config {
    int size; // transformer in/out dimension
    int ffn_hidden_size; // ffn hidden size
    int head_size; // attention head size (size/num_q_heads)
    int num_q_heads; // number of attention (query) heads
    int num_kv_heads; // number of key-value heads
    int max_seq_len; // maximum sequence length
    int num_layers; // number of layers in the transformer
    int vocab_size;
    float norm_eps; // regularization for normalization
    float rope_theta;
    int rope_size; // max rotary dimension for RoPE (<= head_size)
    int tie_embeddings; // true if in & out token embeddings are tied
};

/**
 * State within an inference run.
 */
struct InferState {
    // in/out buffers of size (size,)
    float *x1;
    float *x2;
    float *x3;

    // FFN buffers (ffn_hidden_size,)
    float *h1;
    float *h2;
    float *h3;
    
    // all-heads kv-cache of size (num_layers, max_seq_len, num_kv_heads * head_size)
    float *k_cache;
    float *v_cache;

    // all-heads qkv 
    float *q; // (num_q_heads * head_size,)
    float *k; // (num_kv_heads * head_size,)
    float *v; // (num_kv_heads * head_size,)

    // temp buffer for single-head attention scores (max_seq_len,)
    float *score;
    // multi-head attention output (num_q_heads * head_size,)
    float *mha_out;

    // output logits (vocab_size,)
    float *logits;
};

struct LayerWeights {
    // RMSNorm weights for attention and FFN (size,)
    float *rms_weight_attn;
    float *rms_weight_ffn;

    // qkv projections
    float *q_proj; // (num_q_heads * head_size, size)
    float *k_proj; // (num_kv_heads * head_size, size)
    float *v_proj; // (num_kv_heads * head_size, size)
    float *o_proj; // (size, num_q_heads * head_size)    

    // qkv bias
    float *q_bias; // (num_q_heads * head_size,)
    float *k_bias; // (num_kv_heads * head_size,)
    float *v_bias; // (num_kv_heads * head_size,)

    // FFN
    float *ffn_in_proj; // (ffn_hidden_size, size)
    float *ffn_gate_proj; // (ffn_hidden_size, size)
    float *ffn_out_proj; // (size, ffn_hidden_size)
};

struct Weights {
    float *embedding_in_table; // (vocab_size, size)
    float *embedding_out_proj; // (vocab_size, size)

    float *rms_weight; // (size,)

    struct LayerWeights **layer;
};

/**
 * A multiple layer transformer model.
 */
struct Model {
    struct Config *config;
    struct Weights *weights;
};

struct InferState *init_state(struct Config *config);

/**
 * Forward pass of the model by one token.
 * 
 * Returns logits for the next token.
 */
float *forward(const struct Model *model, struct InferState *state, int token, int pos);