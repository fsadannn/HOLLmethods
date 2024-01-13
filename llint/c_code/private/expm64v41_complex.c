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

void setidentity2(double *P, double *Q, mwSize *n)
{
    mwSize i, total;
    total = (*n) * (*n);
#pragma vectorize
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
#pragma vectorize
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i].real = 1.0;
        Q[i].real = 1.0;
    }
}

void complex_pade(const mxArray *prhs0, mxArray *plhs[], mwSize nrowsA, mwSize ncolumnsA, int p_poldegree, mxDouble s)
{
    /* scalar values to use in zgemm */
    mxComplexDouble zone, zzero, is, c, mc;

    char *chn = "N";
    int k;

    mwSize numberofbytes, total, poldegree = p_poldegree, onei = 1;

    mxArray *Q_M, *Ak_M, *copyofA_M, *Aux_M, *mxPivot;
    mxComplexDouble *A, *P, *Q, *Ak, *copyofA, *Aux;

    mwSize dims[2];

    mxInt32 info;
    mxInt32 *iPivot;

    zone.real = 1.0;
    zone.imag = 0.0;
    zzero.real = 0.0;
    zzero.imag = 0.0;
    is.imag = 0.0;
    c.imag = 0.0;
    mc.imag = 0.0;

    A = mxGetComplexDoubles(prhs0);
    total = nrowsA * ncolumnsA;
    numberofbytes = total * mxGetElementSize(prhs0);

    /*creating auxiliary matrices */
    /* P and Q will store the matrix polynomials, are initialized to identity */
    plhs[0] = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    P = mxGetComplexDoubles(plhs[0]);
    Q_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    Q = mxGetComplexDoubles(Q_M);

    /* identity matrix */
    /* initialize to identity P, Q, I */
    setidentity2_complex(P, Q, &nrowsA);

    /* Ak will store powers of A, is initialized to be A*/
    Ak_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    Ak = mxGetComplexDoubles(Ak_M);

    /* copy of A */
    copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    copyofA = mxGetComplexDoubles(copyofA_M);

    /* auxiliary matrix for calls to zgemm*/
    Aux_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxCOMPLEX);
    Aux = mxGetComplexDoubles(Aux_M);

    /* s = 2^s; */
    is.real = 1.0 / pow(2, s);

    memcpy(Ak, A, numberofbytes);
    /* Ak = Ak*(1/s) */
    zscal(&total, &is, Ak, &onei);

    /* copyofA = copyofA*(1/s) */
    memcpy(copyofA, Ak, numberofbytes);

    // /* P = P + 1/2Ak, Q = Q - 1/2Ak */
    c.real = 0.5;
    zaxpy(&total, &c, Ak, &onei, P, &onei);
    mc.real = -c.real;
    zaxpy(&total, &mc, Ak, &onei, Q, &onei);

    switch (poldegree)
    {
    case 2:
        c.real = 0.083333333333333;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 3:
        c.real = 0.100000000000000;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 0.008333333333333;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;

    case 4:
        c.real = 0.107142857142857;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 0.011904761904762;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);

        c.real = 5.952380952380952e-04;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 5:
        c.real = 0.111111111111111;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 0.013888888888889;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);

        c.real = 9.920634920634920e-04;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 3.306878306878306e-05;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;

    case 6:
        c.real = 0.113636363636364;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 0.015151515151515;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);

        c.real = 0.001262626262626;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 6.313131313131313e-05;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);

        c.real = 1.503126503126503e-06;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 7:
        c.real = 0.11538461538461539;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 0.016025641025641024;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);

        c.real = 0.001456876456876457;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 8.741258741258741e-05;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);

        c.real = 3.2375032375032376e-06;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        zaxpy(&total, &c, Aux, &onei, Q, &onei);

        c.real = 5.781255781255781e-08;
        mc.real = -c.real;
        /* Ak = A*Ak; */
        zgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        zaxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        zaxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;
    case 1:
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
#pragma vectorize
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
    mxDestroyArray(copyofA_M);
    mxDestroyArray(mxPivot);
}

void dpade(const mxArray *prhs0, mxArray *plhs[], mwSize nrowsA, mwSize ncolumnsA, int p_poldegree, mxDouble s)
{
    /* scalar values to use in dgemm */
    mxDouble zone, zzero, is, c, mc;

    char *chn = "N";
    int k;

    mwSize numberofbytes, total, poldegree = p_poldegree, onei = 1;

    mxArray *Q_M, *Ak_M, *copyofA_M, *Aux_M, *mxPivot;
    mxDouble *A, *P, *Q, *Ak, *copyofA, *Aux;

    mwSize dims[2];

    mxInt32 info;
    mxInt32 *iPivot;

    zone = 1.0;
    zzero = 0.0;

    A = mxGetDoubles(prhs0);
    total = nrowsA * ncolumnsA;
    numberofbytes = total * mxGetElementSize(prhs0);

    /*creating auxiliary matrices */
    /* P and Q will store the matrix polynomials, are initialized to identity */
    plhs[0] = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    P = mxGetDoubles(plhs[0]);
    Q_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Q = mxGetDoubles(Q_M);

    /* identity matrix */
    /* initialize to identity P, Q, I */
    setidentity2(P, Q, &nrowsA);

    /* Ak will store powers of A, is initialized to be A*/
    Ak_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Ak = mxGetDoubles(Ak_M);

    /* copy of A */
    copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    copyofA = mxGetDoubles(copyofA_M);

    /* auxiliary matrix for calls to dgemm*/
    Aux_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Aux = mxGetDoubles(Aux_M);

    /* s = 2^s; */
    is = 1.0 / pow(2, s);

    memcpy(Ak, A, numberofbytes);
    /* Ak = Ak*(1/s) */
    dscal(&total, &is, Ak, &onei);

    /* copyofA = copyofA*(1/s) */
    memcpy(copyofA, Ak, numberofbytes);

    // /* P = P + 1/2Ak, Q = Q - 1/2Ak */
    c = 0.5;
    daxpy(&total, &c, Ak, &onei, P, &onei);
    mc = -c;
    daxpy(&total, &mc, Ak, &onei, Q, &onei);

    switch (poldegree)
    {
    case 2:
        c = 0.083333333333333;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 3:
        c = 0.100000000000000;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.008333333333333;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;

    case 4:
        c = 0.107142857142857;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.011904761904762;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 5.952380952380952e-04;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 5:
        c = 0.111111111111111;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.013888888888889;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 9.920634920634920e-04;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 3.306878306878306e-05;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;

    case 6:
        c = 0.113636363636364;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.015151515151515;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 0.001262626262626;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 6.313131313131313e-05;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 1.503126503126503e-06;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 7:
        c = 0.11538461538461539;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.016025641025641024;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 0.001456876456876457;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 8.741258741258741e-05;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 3.2375032375032376e-06;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Ak, &nrowsA, &zzero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 5.781255781255781e-08;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, copyofA, &nrowsA, Aux, &nrowsA, &zzero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;
    case 1:
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
#pragma vectorize
    for (k = 1; k <= poldegree; k++)
    {
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, P, &nrowsA, P, &nrowsA, &zzero, Aux, &nrowsA);
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, Aux, &nrowsA, Aux, &nrowsA, &zzero, P, &nrowsA);
    }
    if ((int)(s) % 2 != 0)
    {
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &zone, P, &nrowsA, P, &nrowsA, &zzero, Aux, &nrowsA);
        memcpy(P, Aux, numberofbytes);
    }

    mxDestroyArray(Ak_M);
    mxDestroyArray(Q_M);
    mxDestroyArray(Aux_M);
    mxDestroyArray(copyofA_M);
    mxDestroyArray(mxPivot);
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
