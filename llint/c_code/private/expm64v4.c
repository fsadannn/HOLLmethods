/*
 *Mex-Function to calculate an approximation to exp(A), where A is a
 *square matrix with real coefficients.
 *
 *The approximation is the quotient of two polynomials P(A)/Q(A) of degree
 *q. Both q and A are input to the routine expm64v4. The output is the
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
    for (i = 0; i < total; i += (*n) + 1)
    {
        P[i] = 1.0;
        Q[i] = 1.0;
    }
}

/* Function to calculate matrix exponential */
/* nlhs (number of output variables) and plhs (output variables) are output
 * nrhs and prhs are input, with nrhs being the number of input variables
 * and prhs, the input variables
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Internal variables */
    double q, c, mc, is, ps, s;
    double *A, *P, *Q, *Ak, *Aux, *copyofA;
    char *chn = "N";
    /* scalar values to use in dgemm */
    double one = 1.0, zero = 0.0;
    /* size_t is as long as a long integer */
    size_t k; /* the length of mwSize depends on compilation option, it may
     * be as long as size_t (option largeArrayDims)
     */
    mwSize ncolumnsA, nrowsA, poldegree, numberofbytes, total, onei = 1;
    mxArray *Q_M, *Ak_M, *Aux_M, *mxPivot, *copyofA_M; /*,*P_M,;*/
    mwSignedIndex info, dims[2];
    mwSignedIndex *iPivot;

    /* Check for proper number of arguments */
    if (nrhs != 3)
    {
        mexErrMsgTxt("expm64v4: two input arguments required.");
    }
    else if (nlhs > 1)
    {
        mexErrMsgTxt("expm64v4: too many output arguments.");
    }

    /* First argument must be a square matrix */
    A = mxGetPr(prhs[0]);
    /* dimensions of input matrix */
    nrowsA = mxGetM(prhs[0]);
    ncolumnsA = mxGetN(prhs[0]);
    if (nrowsA != ncolumnsA)
        mexErrMsgTxt("expm64: Input matrix must be square!");
    if (!mxIsDouble(prhs[0]))
        mexErrMsgTxt("expm64: Input matrix must be real!");

    /* Second argument must be a scalar */
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
    {
        mexErrMsgTxt("expm64: Second argument must be a scalar.");
    }
    q = mxGetScalar(prhs[1]);
    poldegree = (mwSize)q;

    s = mxGetScalar(prhs[2]);

    total = nrowsA * ncolumnsA;
    numberofbytes = total * sizeof(double);

    /*creating auxiliary matrices */
    /* P and Q will store the matrix polynomials, are initialized
     * to identity */
    /*P_M = mxCreateDoubleMatrix(nrowsA,ncolumnsA, mxREAL);*/
    plhs[0] = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    /* P = mxGetPr(P_M);*/
    P = mxGetPr(plhs[0]);
    Q_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Q = mxGetPr(Q_M);
    /* identity matrix */
    /* initialize to identity P, Q, I */
    setidentity2(P, Q, &nrowsA);

    /* Ak will store powers of A, is initialized to be A*/
    Ak_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Ak = mxGetPr(Ak_M);

    /* copy of A */
    copyofA_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    copyofA = mxGetPr(copyofA_M);

    /* auxiliary matrix for calls to dgemm*/
    Aux_M = mxCreateDoubleMatrix(nrowsA, ncolumnsA, mxREAL);
    Aux = mxGetPr(Aux_M);

    /* s = 2^s; */
    ps = pow(2, s);
    is = 1 / ps;

    memcpy(Ak, A, numberofbytes);
    /* Ak = Ak*(1/s) */
    dscal(&total, &is, Ak, &onei);

    /* copyofA = copyofA*(1/s) */
    memcpy(copyofA, Ak, numberofbytes);

    /* P = P + 1/2Ak, Q = Q - 1/2Ak */
    c = 0.5;
    daxpy(&total, &c, Ak, &onei, P, &onei);
    mc = -c;
    daxpy(&total, &mc, Ak, &onei, Q, &onei);

    switch (poldegree)
    {
    case 2:
        c = 0.083333333333333;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 3:
        c = 0.100000000000000;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.008333333333333;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;

    case 4:
        c = 0.107142857142857;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.011904761904762;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 5.952380952380952e-04;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 5:
        c = 0.111111111111111;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.013888888888889;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 9.920634920634920e-04;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 3.306878306878306e-05;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);
        break;

    case 6:
        c = 0.113636363636364;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.015151515151515;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 0.001262626262626;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 6.313131313131313e-05;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 1.503126503126503e-06;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);
        break;

    case 7:
        c = 0.11538461538461539;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 0.016025641025641024;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 0.001456876456876457;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 8.741258741258741e-05;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Ak, &onei, P, &onei);
        /* Q = Q - c Ak; */
        daxpy(&total, &mc, Ak, &onei, Q, &onei);

        c = 3.2375032375032376e-06;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Ak, &nrowsA, &zero, Aux, &nrowsA);

        /* P = P + c Ak; */
        daxpy(&total, &c, Aux, &onei, P, &onei);
        /* Q = Q + c Ak; */
        daxpy(&total, &c, Aux, &onei, Q, &onei);

        c = 5.781255781255781e-08;
        mc = -c;
        /* Ak = A*Ak; */
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, copyofA, &nrowsA, Aux, &nrowsA, &zero, Ak, &nrowsA);

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
    iPivot = (mwSignedIndex *)mxGetData(mxPivot);

    /* Call LAPACK, P = Q\P */
    dgesv(&nrowsA, &nrowsA, Q, &nrowsA, iPivot, P, &nrowsA, &info);

    poldegree = (mwSize)(s / 2);
/* for k=1:s, P = P*P; end */
#pragma vectorize
    for (k = 1; k <= poldegree; k++)
    {
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, P, &nrowsA, P, &nrowsA, &zero, Aux, &nrowsA);
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, Aux, &nrowsA, Aux, &nrowsA, &zero, P, &nrowsA);
    }
    if ((int)(s) % 2 != 0)
    {
        dgemm(chn, chn, &nrowsA, &nrowsA, &nrowsA, &one, P, &nrowsA, P, &nrowsA, &zero, Aux, &nrowsA);
        memcpy(P, Aux, numberofbytes);
    }

    mxDestroyArray(Ak_M);
    mxDestroyArray(Q_M);
    mxDestroyArray(Aux_M);
    mxDestroyArray(copyofA_M);
    mxDestroyArray(mxPivot);/
}
