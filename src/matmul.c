#include "matmul.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
void mva(const float *restrict A, const float *restrict x, const float *restrict b, float *restrict y, int M, int N)
{
    memcpy(y, b, sizeof(float) * M); // Copy b to y
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, x, 1, 1.0f, y, 1);
}

void mv(const float *restrict A, const float *restrict x, float *restrict y, int M, int N)
{
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, x, 1, 0.0f, y, 1);
}

#else

void mva(const float *restrict A, const float *restrict x, const float *restrict b, float *restrict y, int M, int N)
{
#pragma omp parallel for
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
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        y[i] = 0.0;
        for (int j = 0; j < N; ++j)
        {
            y[i] += A[i * N + j] * x[j];
        }
    }
}

#endif