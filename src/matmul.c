#include "matmul.h"

void mm(const float *restrict A, const float *restrict B, float *restrict C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void mva(const float *restrict A, const float *restrict x, const float *restrict b, float *restrict y, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        y[i] = 0.0;
        for (int j = 0; j < N; ++j)
        {
            y[i] += A[i * N + j] * x[j];
        }
        y[i] += b[i];
    }
}

void mv(const float *restrict A, const float *restrict x, float *restrict y, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        y[i] = 0.0;
        for (int j = 0; j < N; ++j)
        {
            y[i] += A[i * N + j] * x[j];
        }
    }
}
