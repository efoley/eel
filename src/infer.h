#pragma once

#include <math.h>

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
void softmax(const float *x, float *o, int size);

static inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

static inline float silu(float x)
{
    return x * sigmoid(x);
}

/**
 * Residual connection.
 */
static inline void res(float *x1, const float *x2, int n)
{
    for (int i = 0; i < n; i++)
    {
        x1[i] += x2[i];
    }
}

/**
 * SwiGLU FFN activation.
 */
static inline void swiglu(const float *g, const float *x, float *y, int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = silu(g[i]) * x[i];
    }
}

void rms_norm(const float *x, const float *g, float *out, int n, float eps);

void attention(
    const float *q, // (head_size)
    // k and v are pointers into the kv cache
    // they should be passed in offset to the appropriate point
    // for this head index on timestep 0
    const float *kb, // pointer into cache of size (kv_len, num_kv_heads, head_size)
    const float *vb, // pointer into cache of size (kv_len, num_kv_heads, head_size)
    float *score,    // temporary buffer for attention scores
    float *out,
    int head_size,
    int num_kv_heads,
    int kv_len); // length of the kv cache

/**
 * RoPE in-place on a vector.
 */
void rope(int d, int attention_head_size, int t, float rope_theta, float *vec);