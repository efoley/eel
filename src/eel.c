#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#include "debug.h"
#include "eel.h"
#include "infer.h"
#include "matmul.h"

#define EEL_DEBUG 0

#define EEL_DEBUG_STATE 0 // very slow!

struct InferState *make_state(struct Config *config) {
    struct InferState *state = malloc(sizeof(struct InferState));

    state->x1 = aligned_alloc(32, config->size * sizeof(float));
    state->x2 = aligned_alloc(32, config->size * sizeof(float));
    state->x3 = aligned_alloc(32, config->size * sizeof(float));

    state->h1 = aligned_alloc(32, config->ffn_hidden_size * sizeof(float));
    state->h2 = aligned_alloc(32, config->ffn_hidden_size * sizeof(float));
    state->h3 = aligned_alloc(32, config->ffn_hidden_size * sizeof(float));

    size_t kv_cache_size = (
        config->num_layers * config->max_seq_len * config->num_kv_heads * config->head_size
    );
    state->k_cache = aligned_alloc(32, kv_cache_size * sizeof(float));
    state->v_cache = aligned_alloc(32, kv_cache_size * sizeof(float));

    size_t q_size = config->num_q_heads * config->head_size;
    size_t kv_size = config->num_kv_heads * config->head_size;
    state->q = aligned_alloc(32, q_size * sizeof(float));
    state->k = aligned_alloc(32, kv_size * sizeof(float));
    state->v = aligned_alloc(32, kv_size * sizeof(float));

    state->score = aligned_alloc(32, config->max_seq_len * sizeof(float));
    state->mha_out = aligned_alloc(32, q_size * sizeof(float));

    state->logits = aligned_alloc(32, config->vocab_size * sizeof(float));

    return state;
}

/**
 * Forward one layer.
 *
 * Assume that the input vector (embedding or from previous layer) is already in state->x1.
 */
void forward_one_layer(const struct Config *config, const struct LayerWeights *weights, struct InferState *state, int layer, int pos)
{
    #if EEL_DEBUG_STATE
        #define CHECK_STATE_FOR_NAN() \
        do { \
            char context[100]; \
            snprintf(context, sizeof(context), "%s:%d", __FILE__, __LINE__); \
            check_state_for_nan(context, config, state); \
        } while (0)
    #else
        #define CHECK_STATE_FOR_NAN() do {} while (0)
    #endif

    CHECK_STATE_FOR_NAN();

    // apply pre-attention RMS norm
    // result in x2
    rms_norm(state->x1, weights->rms_weight_attn, state->x2, config->size, config->norm_eps);

    CHECK_STATE_FOR_NAN();

    // apply q, k, v projections
    // result in q, k, v
    const int all_heads_q_size = config->num_q_heads * config->head_size;
    const int all_heads_kv_size = config->num_kv_heads * config->head_size;
    mva(weights->q_proj, state->x2, weights->q_bias, state->q, all_heads_q_size, config->size);
    mva(weights->k_proj, state->x2, weights->k_bias, state->k, all_heads_kv_size, config->size);
    mva(weights->v_proj, state->x2, weights->v_bias, state->v, all_heads_kv_size, config->size);

    CHECK_STATE_FOR_NAN();

    // apply RoPE in-place to q and k
    rope(all_heads_q_size, config->head_size, pos, config->rope_theta, state->q);
    rope(all_heads_kv_size, config->head_size, pos, config->rope_theta, state->k);

    CHECK_STATE_FOR_NAN();

    // update kv cache for current timestep
    // TODO cache needs to be circular
    // TODO need attention sink
    // TODO need to rotate sink tokens
    int kv_len = pos >= config->max_seq_len ? config->max_seq_len : pos + 1;
    int layer_kv_cache_offset = layer * config->max_seq_len * all_heads_kv_size;

    float *layer_k_cache = state->k_cache + layer_kv_cache_offset;
    float *layer_v_cache = state->v_cache + layer_kv_cache_offset;

    float *kb = layer_k_cache + pos * all_heads_kv_size;
    float *vb = layer_v_cache + pos * all_heads_kv_size;

    memcpy(kb, state->k, all_heads_kv_size * sizeof(float));
    memcpy(vb, state->v, all_heads_kv_size * sizeof(float));

    CHECK_STATE_FOR_NAN();

    // compute multi-query attention
    assert(config->num_q_heads % config->num_kv_heads == 0);
    int q_h_per_kv_h = config->num_q_heads / config->num_kv_heads; // >1 for MQA
    for (int q_h = 0; q_h < config->num_q_heads; q_h++)
    {
        // offset k, v for corresponding head at pos 0
        int kv_h = q_h / q_h_per_kv_h;
        int kv_offset = kv_h * config->head_size;
        float *kh = layer_k_cache + kv_offset;
        float *vh = layer_v_cache + kv_offset;

        // offset q for this head
        float *q = state->q + q_h * config->head_size;

        // offset out for this head
        float *out = state->mha_out + q_h * config->head_size;

        attention(q, kh, vh, state->score, out, config->head_size, config->num_kv_heads, kv_len);
    }

    CHECK_STATE_FOR_NAN();

    // project attention output
    mv(weights->o_proj, state->mha_out, state->x2, config->size, all_heads_q_size);

    CHECK_STATE_FOR_NAN();

    // residual connection
    // result is in x1
    res(state->x1, state->x2, config->size);

    // apply FFN pre-norm
    rms_norm(state->x1, weights->rms_weight_ffn, state->x2, config->size, config->norm_eps); 

    CHECK_STATE_FOR_NAN();

    // FFN
    mv(weights->ffn_in_proj, state->x2, state->h1, config->ffn_hidden_size, config->size);
    mv(weights->ffn_gate_proj, state->x2, state->h2, config->ffn_hidden_size, config->size);
    swiglu(state->h2, state->h1, state->h3, config->ffn_hidden_size);
    mv(weights->ffn_out_proj, state->h3, state->x2, config->size, config->ffn_hidden_size);

    CHECK_STATE_FOR_NAN();

    // residual connection
    // result is in x1
    res(state->x1, state->x2, config->size);
}

float *forward(const struct Model *model, struct InferState *state, int token, int pos) {
    const struct Config *config = model->config;
    const struct Weights *weights = model->weights;

    #if EEL_DEBUG
    printf("forward(..., token=%d, pos=%d)\n", token, pos);
    #endif

    #if EEL_DEBUG
    check_weights_for_nan("", config, weights);
    #endif

    int size = config->size;
    memcpy(state->x1, weights->embedding_in_table + token * size, sizeof(float) * size);

    #if EEL_DEBUG
    printf("Embedding:");
    print_head_tail(state->x1, size, 3);
    #endif

    for (int layer=0; layer<config->num_layers; layer++) {
        forward_one_layer(config, weights->layer[layer], state, layer, pos);

        #if EEL_DEBUG
        if (layer < 1) {
            printf("Layer %d:", layer);
            print_head_tail(state->x1, size, 3);
        }
        #endif
    }

    rms_norm(state->x1, weights->rms_weight, state->x2, size, config->norm_eps);

    mv(weights->embedding_out_proj, state->x2, state->logits, config->vocab_size, size);

    return state->logits;
}