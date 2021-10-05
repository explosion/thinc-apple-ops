import pytest
import thinc_apple_ops.blas
import numpy


def test_basic_sgemm():
    A = numpy.ndarray((5, 4), dtype="f")
    B = numpy.ndarray((4, 7), dtype="f")
    C = thinc_apple_ops.blas.gemm(A, B)
    assert C.shape == (A.shape[0], B.shape[1])


@pytest.mark.parametrize("A_shape,B_shape,transA,transB", [
    [(0, 0), (0, 0), False, False],
    [(0, 0), (0, 0), True, False],
    [(0, 0), (0, 0), False, True],
    [(0, 0), (0, 0), True, True],
    [(0, 5), (5, 0), False, False],
    [(5, 0), (5, 0), False, True],
    [(5, 0), (5, 0), True, False]
])
def test_zero_size(A_shape, B_shape, transA, transB):
    A = numpy.ndarray(A_shape, dtype="f")
    B = numpy.ndarray(B_shape, dtype="f")
    if not transA and not transB:
        C = numpy.dot(A, B)
    elif transA:
        C = numpy.dot(A.T, B)
    elif transB:
        C = numpy.dot(A, B.T)
    else:
        C = numpy.dot(A.T, B.T)
    C_ = thinc_apple_ops.blas.gemm(A, B, trans1=transA, trans2=transB)
    assert C.shape == C_.shape


