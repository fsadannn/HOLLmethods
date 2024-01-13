/*
 *Mex-Function to calculate an approximation to exp(A), where A is a
 *square matrix with real coefficients.
 *
 *The approximation is the quotient of two polynomials P(A)/Q(A) of degree
 *q. Both q and A are input to the routine expm64v1. The output is the
 * approximation to exp(A).
 *
 *The calling syntax is
 * outMatrix = expm64v4(A,q,s)
 */

/*
 * This function will be called from matlab, that's why we need to include
 * mex.h
 */

#include "mex.h"
#include "matrix.h"
#include "blas.h"
#include "lapack.h"
#include "string.h"
#include "math.h"
#include "immintrin.h"
#include "math_sse.h"

// void (*sumABCptr)(const double *, double *, double *, double *, double *, int *);

void initPQ(double *P, double *Q, mwSize *n)
{
    mwSize i, total;
    total = (*n) * (*n);
    // #pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i] = 1.0;
        Q[i] = -1.0;
    }
}

void initPQ_complex(mxComplexDouble *P, mxComplexDouble *Q, mwSize *n)
{
    mwSize i, total;

    total = (*n) * (*n);
    // #pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i].real = 1.0;
        Q[i].real = -1.0;
    }
}

void setidentity2(double *P, double *Q, mwSize *n)
{
    mwSize i, total;
    total = (*n) * (*n);
    // #pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i] = 1.0;
        Q[i] = 1.0;
    }
}

void setidentity2_complex(mxComplexDouble *P, mxComplexDouble *Q, mwSize *n)
{
    mwSize i, total;
    total = (*n) * (*n);
    // #pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i].real = 1.0;
        Q[i].real = 1.0;
    }
}

void setidentity(double *P, mwSize *n)
{
    mwSize i, total;
    total = (*n) * (*n);
    // #pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i] = 1.0;
    }
}

void setidentity_complex(mxComplexDouble *P, mwSize *n)
{
    mwSize i, total;
    total = (*n) * (*n);
    // #pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i].real = 1.0;
    }
}

void complex_pade(const mxArray *prhs0, mxArray *plhs[], mwSize nrowsA, mwSize ncolumnsA, int p_poldegree, mxDouble s)
{
    /* scalar values to use in zgemm */
    mxComplexDouble zone, zzero, mzone, c;

    char *chn = "N";
    int k;
    bool isCopyofAUsed = false;

    mwSize numberofbytes, total, poldegree = p_poldegree, onei = 1;

    mxArray *Q_M, *Ak_M, *copyofA_M, *Aux_M, *mxPivot;
    mxComplexDouble *A, *P, *Q, *Ak, *copyofA, *Aux;

    mwSize dims[2];

    mxInt32 info;
    mxInt32 *iPivot;

    zone.real = 1.0;
    zone.imag = 0.0;
    mzone.real = -1.0;
    mzone.imag = 0.0;
    zzero.real = 0.0;
    zzero.imag = 0.0;
    c.imag = 0.0;

    A = mxGetComplexDoubles(prhs0);
    total = nrowsA * ncolumnsA;
    numberofbytes = total * mxGetElementSize(prhs0);

    /*creating auxiliary matrices */
    /* P and Q will store the matrix polynomials, are initialized to identity */
    plhs[0] = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    P = mxGetComplexDoubles(plhs[0]);
    Q_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    Q = mxGetComplexDoubles(Q_M);

    /* Ak will store powers of A, is initialized to be A*/
    Ak_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    Ak = mxGetComplexDoubles(Ak_M);

    /* auxiliary matrix for calls to zgemm*/
    Aux_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    Aux = mxGetComplexDoubles(Aux_M);

    /* s = 2^s; */
    c.real = 1.0 / pow(2, s);

    memcpy(Ak, A, numberofbytes);

    /* Ak = Ak*(1/s) */
    zscal(&total, &c, Ak, &onei);

    switch (poldegree)
    {
    case 2:
        /* P = I */
        setidentity_complex(P, &nrowsA);

        c.real = 0.083333333333333;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &c, Ak, &nrowsA, Ak, &nrowsA, &zone, P, &nrowsA);

        /*
         * 2*Q = -2*c1*A
         */
        zaxpy(&total, &mzone, Ak, &onei, Q, &onei);

        /*
         * P = P - Q
         */
        c.real = -0.5;
        zaxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        zaxpy(&total, &zone, P, &onei, Q, &onei);

        break;

    case 3:
        /* P = I */
        /* Aux = -2*c1*I */
        initPQ_complex(P, Aux, &nrowsA);

        /* Q = A^2; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c.real = 0.100000000000000;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        zaxpy(&total, &c, Q, &onei, P, &onei);

        c.real = (-2.0) * 0.008333333333333;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        zaxpy(&total, &c, Q, &onei, Aux, &onei);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c2*A^3
         */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        c.real = -0.5;
        zaxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        zaxpy(&total, &zone, P, &onei, Q, &onei);

        break;

    case 4:
        /* P = I */
        /* Aux = -2*c1*I */
        initPQ_complex(P, Aux, &nrowsA);

        /* Q = A^2; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c.real = 0.107142857142857;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        zaxpy(&total, &c, Q, &onei, P, &onei);

        c.real = (-2.0) * 0.011904761904762;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        zaxpy(&total, &c, Q, &onei, Aux, &onei);

        c.real = 5.952380952380952e-04;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4
         */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &c, Q, &nrowsA, Q, &nrowsA, &zone, P, &nrowsA);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c2*A^3
         */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        c.real = -0.5;
        zaxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        zaxpy(&total, &zone, P, &onei, Q, &onei);

        break;
    case 5:
        /* copy of A */
        copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
        copyofA = mxGetComplexDoubles(copyofA_M);
        /* copyofA = copyofA*(1/s) */
        memcpy(copyofA, Ak, numberofbytes);
        isCopyofAUsed = true;

        /* P = I */
        /* Aux = -2*c1*I */
        initPQ_complex(P, Aux, &nrowsA);

        /* Q = A^2; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c.real = 0.111111111111111;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        zaxpy(&total, &c, Q, &onei, P, &onei);

        c.real = (-2.0) * 0.013888888888889;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        zaxpy(&total, &c, Q, &onei, Aux, &onei);

        /* Ak = A^4 */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Q, &nrowsA, Q, &nrowsA, &zzero, Ak, &nrowsA);

        c.real = 9.920634920634920e-04;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4
         */
        zaxpy(&total, &c, Ak, &onei, P, &onei);

        c.real = (-2.0) * 3.306878306878306e-05;
        /*
         * Aux = -2*c1 - 2*c3*A^2 - 2*c5*A^4
         */
        zaxpy(&total, &c, Ak, &onei, Aux, &onei);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c3*A^3 - 2*c5*A^5
         */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        c.real = -0.5;
        zaxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        zaxpy(&total, &zone, P, &onei, Q, &onei);

        break;
    case 6:
        /* copy of A */
        copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
        copyofA = mxGetComplexDoubles(copyofA_M);
        /* copyofA = copyofA*(1/s) */
        memcpy(copyofA, Ak, numberofbytes);
        isCopyofAUsed = true;

        /* P = I */
        /* Aux = -2*c1*I */
        initPQ_complex(P, Aux, &nrowsA);

        /* Q = A^2; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c.real = 0.113636363636364;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        zaxpy(&total, &c, Q, &onei, P, &onei);

        c.real = (-2.0) * 0.015151515151515;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        zaxpy(&total, &c, Q, &onei, Aux, &onei);

        /* Ak = A^4 */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Q, &nrowsA, Q, &nrowsA, &zzero, Ak, &nrowsA);

        c.real = 0.001262626262626;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4
         */
        zaxpy(&total, &c, Ak, &onei, P, &onei);

        c.real = (-2.0) * 6.313131313131313e-05;
        /*
         * Aux = -2*c1 - 2*c3*A^2 - 2*c5*A^4
         */
        zaxpy(&total, &c, Ak, &onei, Aux, &onei);

        c.real = 1.503126503126503e-06;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4 + c6*A^6
         */

        /* Ak = A^6 */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &c, Q, &nrowsA, Ak, &nrowsA, &zone, P, &nrowsA);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c3*A^3 - 2*c5*A^5
         */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        c.real = -0.5;
        zaxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        zaxpy(&total, &zone, P, &onei, Q, &onei);

        break;
    case 1:
        /* identity matrix */
        /* initialize to identity P, Q, I */
        setidentity2_complex(P, Q, &nrowsA);
        // /* P = P + 1/2Ak, Q = Q - 1/2Ak */
        c.real = 0.5;
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        c.real = -0.5;
        zaxpy(&total, &c, Ak, &onei, Q, &onei);
        break;

    default:
        mexErrMsgTxt("poldegree must be between 2 and 6");
    }
    /* P = Q\P; */
    /* Create inputs for DGESV */
    dims[0] = nrowsA;
    dims[1] = nrowsA;
    mxPivot = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    iPivot = mxGetInt32s(mxPivot);

    /* Call LAPACK, P = Q\P */
    zgesv(&nrowsA, &nrowsA, Q, &nrowsA, iPivot, P, &nrowsA, &info);

    poldegree = (mwSize)(s / 2);
    /* for k=1:s, P = P*P; end */
    // #pragma vectorize
    for (k = 1; k <= poldegree; k++)
    {
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, P, &nrowsA, P, &nrowsA, &zzero, Aux, &nrowsA);
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Aux, &nrowsA, Aux, &nrowsA, &zzero, P, &nrowsA);
    }
    if ((int)(s) % 2 != 0)
    {
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, P, &nrowsA, P, &nrowsA, &zzero, Aux, &nrowsA);
        memcpy(P, Aux, numberofbytes);
    }

    mxDestroyArray(Ak_M);
    mxDestroyArray(Q_M);
    mxDestroyArray(Aux_M);
    mxDestroyArray(mxPivot);
    if (isCopyofAUsed)
    {
        mxDestroyArray(copyofA_M);
    }
}

void dpade(const mxArray *prhs0, mxArray *plhs[], mwSize nrowsA, mwSize ncolumnsA, int p_poldegree, mxDouble s)
{
    /* scalar values to use in dgemm */
    mxDouble zone, zzero, mzone, c, cc;

    char *chn = "N";
    int k;
    bool isCopyofAUsed = false;

    mwSize numberofbytes, total, poldegree = p_poldegree, onei = 1;

    mxArray *Q_M, *Ak_M, *copyofA_M, *Aux_M, *mxPivot;
    mxDouble *A, *P, *Q, *Ak, *copyofA, *Aux;

    mwSize dims[2];

    mxInt32 info;
    mxInt32 *iPivot;

    zone = 1.0;
    zzero = 0.0;
    mzone = -1.0;

    A = mxGetDoubles(prhs0);
    total = nrowsA * ncolumnsA;
    numberofbytes = total * mxGetElementSize(prhs0);

    /*creating auxiliary matrices */
    /* P and Q will store the matrix polynomials, are initialized to identity */
    plhs[0] = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    P = mxGetDoubles(plhs[0]);
    Q_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Q = mxGetDoubles(Q_M);

    /* Ak will store powers of A, is initialized to be A*/
    Ak_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Ak = mxGetDoubles(Ak_M);

    /* auxiliary matrix for calls to dgemm*/
    Aux_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Aux = mxGetDoubles(Aux_M);

    /* s = 2^s; */
    c = 1.0 / pow(2, s);

    memcpy(Ak, A, numberofbytes);

    /* Ak = Ak*(1/s) */
    dscal(&total, &c, Ak, &onei);

    switch (poldegree)
    {
    case 2:
        /* P = I */
        setidentity(P, &nrowsA);

        c = 0.083333333333333;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &c, Ak, &nrowsA, Ak, &nrowsA, &zone, P, &nrowsA);

        /*
         * 2*Q = -2*c1*A
         */
        // daxpy(&total, &mzone, Ak, &onei, Q, &onei);
        dsub_sse(Ak, Q, total);

        /*
         * P = P - Q
         */
        // c = -0.5;
        // daxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        // daxpy(&total, &zone, P, &onei, Q, &onei);
        // summ_pq_avx(P, Q, &total);
        dupdate_pq_sse(P, Q, total);

        break;

    case 3:
        /* P = I */
        /* Aux = -2*c1*I */
        initPQ(P, Aux, &nrowsA);

        /* Q = A^2; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c = 0.100000000000000;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        // daxpy(&total, &c, Q, &onei, P, &onei);

        cc = (-2.0) * 0.008333333333333;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        // daxpy(&total, &cc, Q, &onei, Aux, &onei);

        // summ_abc_avx(Q, P, Aux, &c, &cc, &total);
        daxbzpy_sse(Q, P, Aux, c, cc, total);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c2*A^3
         */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        // c = -0.5;
        // daxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        // daxpy(&total, &zone, P, &onei, Q, &onei);
        // summ_pq_avx(P, Q, &total);
        dupdate_pq_sse(P, Q, total);

        break;

    case 4:
        /* P = I */
        /* Aux = -2*c1*I */
        initPQ(P, Aux, &nrowsA);

        /* Q = A^2; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c = 0.107142857142857;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        // daxpy(&total, &c, Q, &onei, P, &onei);

        cc = (-2.0) * 0.011904761904762;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        // daxpy(&total, &cc, Q, &onei, Aux, &onei);
        // summ_abc_avx(Q, P, Aux, &c, &cc, &total);
        daxbzpy_sse(Q, P, Aux, c, cc, total);

        c = 5.952380952380952e-04;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4
         */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &c, Q, &nrowsA, Q, &nrowsA, &zone, P, &nrowsA);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c2*A^3
         */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        // c = -0.5;
        // daxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        // daxpy(&total, &zone, P, &onei, Q, &onei);
        // summ_pq_avx(P, Q, &total);
        dupdate_pq_sse(P, Q, total);

        break;
    case 5:
        /* copy of A */
        copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
        copyofA = mxGetDoubles(copyofA_M);
        /* copyofA = copyofA*(1/s) */
        memcpy(copyofA, Ak, numberofbytes);
        isCopyofAUsed = true;

        /* P = I */
        /* Aux = -2*c1*I */
        initPQ(P, Aux, &nrowsA);

        /* Q = A^2; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c = 0.111111111111111;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        // daxpy(&total, &c, Q, &onei, P, &onei);

        cc = (-2.0) * 0.013888888888889;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        // daxpy(&total, &cc, Q, &onei, Aux, &onei);
        // summ_abc_avx(Q, P, Aux, &c, &cc, &total);
        daxbzpy_sse(Q, P, Aux, c, cc, total);

        /* Ak = A^4 */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Q, &nrowsA, Q, &nrowsA, &zzero, Ak, &nrowsA);

        c = 9.920634920634920e-04;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4
         */
        // daxpy(&total, &c, Ak, &onei, P, &onei);

        cc = (-2.0) * 3.306878306878306e-05;
        /*
         * Aux = -2*c1 - 2*c3*A^2 - 2*c5*A^4
         */
        // daxpy(&total, &cc, Ak, &onei, Aux, &onei);
        // summ_abc_avx(Ak, P, Aux, &c, &cc, &total);
        daxbzpy_sse(Ak, P, Aux, c, cc, total);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c3*A^3 - 2*c5*A^5
         */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        // c = -0.5;
        // daxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        // daxpy(&total, &zone, P, &onei, Q, &onei);
        // summ_pq_avx(P, Q, &total);
        dupdate_pq_sse(P, Q, total);

        break;
    case 6:
        /* copy of A */
        copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
        copyofA = mxGetDoubles(copyofA_M);
        /* copyofA = copyofA*(1/s) */
        memcpy(copyofA, Ak, numberofbytes);
        isCopyofAUsed = true;

        /* P = I */
        /* Aux = -2*c1*I */
        initPQ(P, Aux, &nrowsA);

        /* Q = A^2; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Ak, &nrowsA, Ak, &nrowsA, &zzero, Q, &nrowsA);

        c = 0.113636363636364;
        /*
         * P = P + c2*A^2
         * P = I + c2*A^2
         */
        // daxpy(&total, &c, Q, &onei, P, &onei);

        cc = (-2.0) * 0.015151515151515;
        /*
         * Aux = -2*c1 - 2*c3*A^2
         */
        // daxpy(&total, &c, Q, &onei, Aux, &onei);

        // summ_abc_avx(Q, P, Aux, &c, &cc, &total);
        daxbzpy_sse(Q, P, Aux, c, cc, total);

        /* Ak = A^4 */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Q, &nrowsA, Q, &nrowsA, &zzero, Ak, &nrowsA);

        c = 0.001262626262626;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4
         */
        // daxpy(&total, &c, Ak, &onei, P, &onei);

        cc = (-2.0) * 6.313131313131313e-05;
        /*
         * Aux = -2*c1 - 2*c3*A^2 - 2*c5*A^4
         */
        // daxpy(&total, &cc, Ak, &onei, Aux, &onei);
        // summ_abc_avx(Ak, P, Aux, &c, &cc, &total);
        daxbzpy_sse(Ak, P, Aux, c, cc, total);

        c = 1.503126503126503e-06;
        /*
         * P = P + c4*A^4
         * P = I + c2*A^2 + c4*A^4 + c6*A^6
         */

        /* Ak = A^6 */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &c, Q, &nrowsA, Ak, &nrowsA, &zone, P, &nrowsA);

        /*
         * Q = A*Aux
         * Q = -2*c1*A - 2*c3*A^3 - 2*c5*A^5
         */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Q, &nrowsA);

        /*
         * P = P - Q
         */
        // c = -0.5;
        // daxpy(&total, &c, Q, &onei, P, &onei);
        /*
         * Q = Q + (P - 2*Q)
         */
        // daxpy(&total, &zone, P, &onei, Q, &onei);
        // summ_pq_avx(P, Q, &total);
        dupdate_pq_sse(P, Q, total);

        break;
    case 1:
        /* identity matrix */
        /* initialize to identity P, Q, I */
        setidentity2(P, Q, &nrowsA);
        // /* P = P + 1/2Ak, Q = Q - 1/2Ak */
        c = 0.5;
        daxpy(&total, &c, Ak, &onei, P, &onei);
        c = -0.5;
        daxpy(&total, &c, Ak, &onei, Q, &onei);
        break;

    default:
        mexErrMsgTxt("poldegree must be between 2 and 6");
    }

    /* P = Q\P; */
    /* Create inputs for DGESV */
    dims[0] = nrowsA;
    dims[1] = nrowsA;
    mxPivot = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    iPivot = mxGetInt32s(mxPivot);

    /* Call LAPACK, P = Q\P */
    dgesv(&nrowsA, &nrowsA, Q, &nrowsA, iPivot, P, &nrowsA, &info);

    poldegree = (mwSize)(s / 2);
    /* for k=1:s, P = P*P; end */
    // #pragma vectorize
    for (k = 1; k <= poldegree; k++)
    {
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, P, &nrowsA, P, &nrowsA, &zzero, Aux, &nrowsA);
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Aux, &nrowsA, Aux, &nrowsA, &zzero, P, &nrowsA);
    }
    if ((int)(s) % 2 != 0)
    {
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, P, &nrowsA, P, &nrowsA, &zzero, Aux, &nrowsA);
        memcpy(P, Aux, numberofbytes);
        // memcpy_fast(P, Aux, numberofbytes);
    }

    mxDestroyArray(Ak_M);
    mxDestroyArray(Q_M);
    mxDestroyArray(Aux_M);
    mxDestroyArray(mxPivot);
    if (isCopyofAUsed)
    {
        mxDestroyArray(copyofA_M);
    }
}

/* Function to calculate matrix exponential */
/* nlhs (number of output variables) and plhs (output variables) are output
 * nrhs and prhs are input, with nrhs being the number of input variables
 * and prhs, the input variables
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    bool isComplex = false;
    mwSize nrowsA, ncolumnsA;
    mxDouble q;
    int poldegree;

    /* Check for proper number of arguments */
    if (nrhs != 3)
    {
        mexErrMsgTxt("expm64v4: two input arguments required.");
    }
    else if (nlhs > 1)
    {
        mexErrMsgTxt("expm64v4: too many output arguments.");
    }

    nrowsA = mxGetM(prhs[0]);
    ncolumnsA = mxGetN(prhs[0]);
    if (nrowsA != ncolumnsA)
    {
        mexErrMsgTxt("expm64: Input matrix must be square!");
    }

    isComplex = mxIsComplex(prhs[0]);

    /* Second argument must be a scalar */
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
    {
        mexErrMsgTxt("expm64: Second argument must be a scalar.");
    }

    q = mxGetScalar(prhs[1]);
    poldegree = (int)q;

    /* Third argument must be a scalar */
    if (!mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
    {
        mexErrMsgTxt("expm64: Second argument must be a scalar.");
    }

    q = mxGetScalar(prhs[2]);
    q = ceil(q);

    if (isComplex)
    {
        complex_pade(prhs[0], plhs, nrowsA, ncolumnsA, poldegree, q);
    }
    else
    {
        dpade(prhs[0], plhs, nrowsA, ncolumnsA, poldegree, q);
    }
}
