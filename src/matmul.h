#pragma once

/**
 * Compute
 * y = Ax + b
 *
 * @param A matrix with dimensions M x N
 * @param x vector of length N.
 * @param b vector of length M.
 * @param y the result of length M.
 */
void mva(const float *restrict A, const float *restrict x, const float *restrict b, float *restrict y, int M, int N);

/**
 * Compute y = Ax
 *
 * @param A matrix with dimensions M x N.
 * @param x vector of length N.
 * @param y the result of length M.
 */
void mv(const float *restrict A, const float *restrict x, float *restrict y, int M, int N);
