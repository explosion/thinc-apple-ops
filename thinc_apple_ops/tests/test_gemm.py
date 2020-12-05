import thinc_apple_ops.blas
import numpy


def test_basic_sgemm():
    A = numpy.ndarray((5, 4), dtype="f")
    B = numpy.ndarray((4, 7), dtype="f")
    C = thinc_apple_ops.blas.gemm(A, B)
    assert C.shape == (A.shape[0], B.shape[1])
