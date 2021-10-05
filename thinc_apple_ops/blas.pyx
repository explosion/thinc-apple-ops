cimport numpy as np
import numpy

cdef extern from "Accelerate/Accelerate.h":
    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO:      CblasUpper, CblasLower
    enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:      CblasLeft, CblasRight

    # BLAS level 1 routines

    void cblas_sswap(int M, float  *x, int incX, float  *y, int incY) nogil
    void cblas_sscal(int N, float  alpha, float  *x, int incX) nogil
    void cblas_scopy(int N, float  *x, int incX, float  *y, int incY) nogil
    void cblas_saxpy(int N, float  alpha, float  *x, int incX, float  *y, int incY ) nogil
    float cblas_sdot(int N, float  *x, int incX, float  *y, int incY ) nogil
    float cblas_snrm2(int N, float  *x, int incX) nogil
    float cblas_sasum(int N, float  *x, int incX) nogil
    int cblas_isamax(int N, float  *x, int incX) nogil

    # BLAS level 2 routines
    void cblas_sgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                                 float  alpha, float  *A, int lda, float  *x, int incX,
                                 float  beta, float  *y, int incY) nogil

    void cblas_sger(CBLAS_ORDER Order, int M, int N, float  alpha, float  *x,
                                int incX, float  *y, int incY, float  *A, int lda) nogil

    # BLAS level 3 routines
    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc) nogil


cpdef np.ndarray gemm(float[:, ::1] A, float[:, ::1] B, bint trans1=False, bint trans2=False): 
    cdef int nM = A.shape[0] if not trans1 else A.shape[1]
    cdef int nK = A.shape[1] if not trans1 else A.shape[0]
    cdef int nN = B.shape[1] if not trans2 else B.shape[0]
    cdef np.ndarray out = numpy.zeros((nM, nN), dtype="f")

    cdef float[:, ::1] C = out
    if nM == 0 or nK == 0 or nN == 0:
        return out

    cblas_sgemm(
        CblasRowMajor,
        CblasTrans if trans1 else CblasNoTrans,
        CblasTrans if trans2 else CblasNoTrans,
        nM,
        nN,
        nK,
        1.0,
        &A[0, 0],
        A.shape[1],
        &B[0, 0],
        B.shape[1],
        1.0,
        &C[0, 0],
        C.shape[1]
    )
    return out
