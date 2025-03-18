#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "eel.h"

void check_array(const char *context, const char *name, const float *arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (isnan(arr[i])) {
            printf("%s: NaN detected in array '%s' at index %zu\n", context, name, i);
            abort();
        }
    }
}

void check_state_for_nan(const char* context, const struct Config *config, struct InferState *state) {
    check_array(context, "x1", state->x1, config->size);
    check_array(context, "x2", state->x2, config->size);
    check_array(context, "x3", state->x3, config->size);

    check_array(context, "h1", state->h1, config->ffn_hidden_size);
    check_array(context, "h2", state->h2, config->ffn_hidden_size);
    check_array(context, "h3", state->h3, config->ffn_hidden_size);

    size_t kv_cache_size = config->num_layers * config->max_seq_len * config->num_kv_heads * config->head_size;
    check_array(context, "k_cache", state->k_cache, kv_cache_size);
    check_array(context, "v_cache", state->v_cache, kv_cache_size);

    check_array(context, "q", state->q, config->num_q_heads * config->head_size);
    check_array(context, "k", state->k, config->num_kv_heads * config->head_size);
    check_array(context, "v", state->v, config->num_kv_heads * config->head_size);

    check_array(context, "score", state->score, config->max_seq_len);
    check_array(context, "mha_out", state->mha_out, config->num_q_heads * config->head_size);

    check_array(context, "logits", state->logits, config->vocab_size);
}

void check_weights_for_nan(const char *context, const struct Config *config, const struct Weights *weights) {
    check_array(context, "embedding_in_table", weights->embedding_in_table, (size_t)config->vocab_size * config->size);
    check_array(context, "embedding_out_proj", weights->embedding_out_proj, (size_t)config->vocab_size * config->size);
    check_array(context, "rms_weight", weights->rms_weight, config->size);
  
    for (int layer_idx = 0; layer_idx < config->num_layers; ++layer_idx) {
      const struct LayerWeights *layer_weights = weights->layer[layer_idx];
      char layer_context[256];  // Fixed-size buffer, avoid potential overflow
      snprintf(layer_context, sizeof(layer_context), "%s layer %d", context, layer_idx); // Safe string formatting
  
  
      check_array(layer_context, "rms_weight_attn", layer_weights->rms_weight_attn, config->size);
      check_array(layer_context, "rms_weight_ffn", layer_weights->rms_weight_ffn, config->size);
      check_array(layer_context, "q_proj", layer_weights->q_proj, (size_t)config->num_q_heads * config->head_size * config->size);
      check_array(layer_context, "k_proj", layer_weights->k_proj, (size_t)config->num_kv_heads * config->head_size * config->size);
      check_array(layer_context, "v_proj", layer_weights->v_proj, (size_t)config->num_kv_heads * config->head_size * config->size);
      check_array(layer_context, "o_proj", layer_weights->o_proj, (size_t)config->size * config->num_q_heads * config->head_size);
      check_array(layer_context, "q_bias", layer_weights->q_bias, (size_t)config->num_q_heads * config->head_size);
      check_array(layer_context, "k_bias", layer_weights->k_bias, (size_t)config->num_kv_heads * config->head_size);
      check_array(layer_context, "v_bias", layer_weights->v_bias, (size_t)config->num_kv_heads * config->head_size);
      check_array(layer_context, "ffn_in_proj", layer_weights->ffn_in_proj, (size_t)config->ffn_hidden_size * config->size);
      check_array(layer_context, "ffn_gate_proj", layer_weights->ffn_gate_proj, (size_t)config->ffn_hidden_size * config->size);
      check_array(layer_context, "ffn_out_proj", layer_weights->ffn_out_proj, (size_t)config->size * config->ffn_hidden_size);
    }
}

void print_head_tail(const float *arr, int n, int h)
{
    for (int i = 0; i < h; i++)
    {
        if (i > 0)
        {
            printf(", ");
        }
        printf("%f ", arr[i]);
    }
    printf("...");
    for (int i = n - h; i < n; i++)
    {
        if (i > n - h)
        {
            printf(", ");
        }
        printf("%f ", arr[i]);
    }
    printf("\n");
}