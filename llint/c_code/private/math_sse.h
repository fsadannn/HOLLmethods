#ifndef math_sse_h
#define math_sse_h

#include "immintrin.h"
#include "matrix.h"

// general complex

void zaxbzpy_sse(mxComplexDouble *A, mxComplexDouble *B, mxComplexDouble *C, mxComplexDouble alpha, mxComplexDouble beta, size_t N)
{
    __m128d alpha_real = _mm_set1_pd(alpha.real);
    __m128d alpha_imag = _mm_set1_pd(alpha.imag);
    __m128d beta_real = _mm_set1_pd(beta.real);
    __m128d beta_imag = _mm_set1_pd(beta.imag);
    __m128d a_real, a_imag, b_real, b_imag, c_real, c_imag, result_real, result_imag;
    for (size_t i = 0; i < N; i++)
    {
        a_real = _mm_set_pd(A[i].real, A[i].real);
        a_imag = _mm_set_pd(A[i].imag, A[i].imag);
        b_real = _mm_load_pd(&B[i].real);
        b_imag = _mm_load_pd(&B[i].imag);
        c_real = _mm_load_pd(&C[i].real);
        c_imag = _mm_load_pd(&C[i].imag);
        result_real = _mm_add_pd(_mm_mul_pd(alpha_real, a_real), b_real);
        result_imag = _mm_add_pd(_mm_mul_pd(alpha_imag, a_imag), b_imag);
        _mm_store_pd(&B[i].real, result_real);
        _mm_store_pd(&B[i].imag, result_imag);
        result_real = _mm_add_pd(_mm_mul_pd(beta_real, a_real), c_real);
        result_imag = _mm_add_pd(_mm_mul_pd(beta_imag, a_imag), c_imag);
        _mm_store_pd(&C[i].real, result_real);
        _mm_store_pd(&C[i].imag, result_imag);
    }
}

void zaxpy_sse(mxComplexDouble *A, mxComplexDouble *B, mxComplexDouble alpha, size_t N)
{
    __m128d alpha_real = _mm_set1_pd(alpha.real);
    __m128d alpha_imag = _mm_set1_pd(alpha.imag);
    __m128d a_real, a_imag, b_real, b_imag, result_real, result_imag;
    for (size_t i = 0; i < N; i++)
    {
        a_real = _mm_set_pd(A[i].real, A[i].real);
        a_imag = _mm_set_pd(A[i].imag, A[i].imag);
        b_real = _mm_load_pd(&B[i].real);
        b_imag = _mm_load_pd(&B[i].imag);
        result_real = _mm_add_pd(_mm_mul_pd(alpha_real, a_real), b_real);
        result_imag = _mm_add_pd(_mm_mul_pd(alpha_imag, a_imag), b_imag);
        _mm_store_pd(&B[i].real, result_real);
        _mm_store_pd(&B[i].imag, result_imag);
    }
}

// general double

// The function `daxpy_sse2` is performing the operation `B = alpha * A + B` using SSE instructions.
void daxpy_sse(double *A, double *B, double alpha, size_t N)
{
    __m128d alpha_vec = _mm_set1_pd(alpha);
    __m128d a_vec, b_vec, result;
    size_t aligendN = N - N % 2;
    for (size_t i = 0; i < aligendN; i += 2)
    {
        a_vec = _mm_load_pd(A + i);
        b_vec = _mm_load_pd(B + i);
        result = _mm_add_pd(_mm_mul_pd(alpha_vec, a_vec), b_vec);
        _mm_store_pd(B + i, result);
    }
    if (N % 2 != 0)
    {
        B[N - 1] += alpha * A[N - 1];
    }
}

// The function `daxbzpy_sse2` is performing the operation `B = alpha * A + B` and `C = beta * A + C` using SSE instructions.
void daxbzpy_sse(double *A, double *B, double *C, double alpha, double beta, size_t N)
{
    __m128d alpha_vec = _mm_set1_pd(alpha);
    __m128d beta_vec = _mm_set1_pd(beta);
    __m128d a_vec, b_vec, c_vec, result;
    size_t aligendN = N - N % 2;
    for (size_t i = 0; i < aligendN; i += 2)
    {
        a_vec = _mm_load_pd(A + i);
        b_vec = _mm_load_pd(B + i);
        c_vec = _mm_load_pd(C + i);
        result = _mm_add_pd(_mm_mul_pd(alpha_vec, a_vec), b_vec);
        _mm_store_pd(B + i, result);
        result = _mm_add_pd(_mm_mul_pd(beta_vec, a_vec), c_vec);
        _mm_store_pd(C + i, result);
    }
    if (N % 2 != 0)
    {
        B[N - 1] += alpha * A[N - 1];
        C[N - 1] += beta * A[N - 1];
    }
}

/** The function `dsum_sse` is performing the operation `B = A + B` using SSE instructions. It takes two arrays `A` and `B` of length `N` and adds the corresponding elements of `A` and `B`, storing the result in `B`. The function uses SSE instructions to perform the addition in parallel for better performance. If `N` is not divisible by 2, the function performs a scalar addition for the last element.*/
void dsum_sse(double *A, double *B, size_t N)
{
    __m128d a_vec, b_vec, result;
    size_t aligendN = N - N % 2;
    for (size_t i = 0; i < aligendN; i += 2)
    {
        a_vec = _mm_load_pd(A + i);
        b_vec = _mm_load_pd(B + i);
        result = _mm_add_pd(a_vec, b_vec);
        _mm_store_pd(B + i, result);
    }
    if (N % 2 != 0)
    {
        B[N - 1] += A[N - 1];
    }
}

/** The function `dsub_sse` is performing the operation `B = B - A` using SSE instructions. It takes two arrays `A` and `B` of length `N` and subtracts the corresponding elements of `A` from `B`, storing the result in `B`. The function uses SSE instructions to perform the subtraction in parallel for better performance. If `N` is not divisible by 2, the function performs a scalar subtraction for the last element.*/
void dsub_sse(double *A, double *B, size_t N)
{
    __m128d a_vec, b_vec, result;
    size_t aligendN = N - N % 2;
    for (size_t i = 0; i < aligendN; i += 2)
    {
        a_vec = _mm_load_pd(A + i);
        b_vec = _mm_load_pd(B + i);
        result = _mm_sub_pd(b_vec, a_vec);
        _mm_store_pd(B + i, result);
    }
    if (N % 2 != 0)
    {
        B[N - 1] -= A[N - 1];
    }
}

void dupdate_pq_sse(double *P, double *Q, size_t n)
{
    __m128d half = _mm_set1_pd(0.5);
    __m128d p_vec, q_vec, result_p, result_q;
    size_t aligendN = n - n % 2;
    for (size_t i = 0; i < aligendN; i += 2)
    {
        p_vec = _mm_load_pd(P + i);
        q_vec = _mm_load_pd(Q + i);
        result_p = _mm_sub_pd(p_vec, _mm_mul_pd(half, q_vec));
        result_q = _mm_add_pd(q_vec, result_p);
        _mm_store_pd(P + i, result_p);
        _mm_store_pd(Q + i, result_q);
    }
    if (n % 2 != 0)
    {
        P[n - 1] -= 0.5 * Q[n - 1];
        Q[n - 1] += P[n - 1];
    }
}

#endif