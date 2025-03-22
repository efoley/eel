#include "matmul.h"

#include <assert.h>
#include <stdint.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
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

#elif defined(__AVX2__)

// see https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
           vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}

void mva(const float *restrict A, const float *restrict x, const float *restrict b, float *restrict y, int M, int N)
{
    assert((uintptr_t)A % 32 == 0);
    assert((uintptr_t)x % 32 == 0);
    
    #pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        const float *rowA = &A[i * N];
        __m256 sum = _mm256_setzero_ps();

        for (int j = 0; j < N; j += 8)
        {
            __m256 vA = _mm256_loadu_ps(&rowA[j]);
            __m256 vx = _mm256_loadu_ps(&x[j]);

            // do the fma operation. there isn't any speed difference between
            // using FMA vs plain AVX multiply and add. not surprising since we're
            // surely just memory bound loading from A & x.
            #if 1
            sum = _mm256_fmadd_ps(vA, vx, sum); // sum += A[j] * x[j]
            #else
            __m256 vAx = _mm256_mul_ps(vA, vx);
            sum = _mm256_add_ps(sum, vAx);
            #endif
        }

        if (b) {
            y[i] = hsum256_ps_avx(sum) + b[i];
        } else {
            y[i] = hsum256_ps_avx(sum);
        }
    }
}

void mv(const float *restrict A, const float *restrict x, float *restrict y, int M, int N)
{
    mva(A, x, NULL, y, M, N);
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