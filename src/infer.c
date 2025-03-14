#include "infer.h"

#include <float.h>

void softmax(const float *x, float *o, int size) {
    float score_max = -FLT_MAX;
    for (int i = 0; i < size; ++i) {
      if (x[i] > score_max) {
        score_max = x[i];
      }
    }
    float score_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
      o[i] = expf(x[i] - score_max);
      score_sum += o[i];
    }
    for (int i = 0; i < size; ++i) {
      o[i] /= score_sum;
    }
}

void attention(
    const float *q,
    const float *kb,
    const float *vb,
    float* score,
    float* out,
    int head_size,
    int num_kv_heads,
    int kv_len) {

    // stride per timestep into the kv-cache
    const int kv_stride = num_kv_heads * head_size;
    for (int t = 0; t < kv_len; ++t) {
        const float *k = kb + t * kv_stride;
        float s = 0.f;
        for (int i = 0; i < head_size; ++i) {
            s += q[i] * k[i];
        }
        s /= sqrt(head_size); // TODO does Qwen use scaled dpa?
        score[t] = (float)s;
    }

    softmax(score, score, kv_len);

    for (int i=0; i<head_size; i++) {
        float s = 0.f;
        for (int t=0; t<kv_len; t++) {
            s += score[t] * vb[i + t * kv_stride];
        }
        out[i] = s;
    }
}


void rms_norm(const float *x, const float *g, float *out, int n, float eps) {
    float s = 0.f;
    for (int i=0; i<n; i++) {
        s += x[i] * x[i];
    }
    float scale = 1.f/sqrt(s/n + eps);
    for (int i=0; i<n; i++) {
        out[i] = g[i] * x[i] * scale;
    }
}


void rope(int d, int attention_head_size, int t, float rope_theta, float *vec) {
    // TODO EDF support partial rotary embeddings
    // TODO could pre-cache sin/cos of length attention_head_size, since they're reused and
    // being transcendental they're actually expensive
    for (int i=0; i<d; i+=2) {
        const int j = i % attention_head_size; // position within attention head
        const float theta = 1.f/pow(rope_theta, j/(float)attention_head_size);
        const float sin_val = sin(theta * t);
        const float cos_val = cos(theta * t);
        float x = vec[i];
        float y = vec[i+1];
        vec[i] = x*cos_val - y*sin_val;
        vec[i+1] = x*sin_val + y*cos_val;
    }
}