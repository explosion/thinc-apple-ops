import pytest
import thinc_apple_ops.blas
import numpy


def test_basic_sgemm():
    A = numpy.random.randn(5, 4).astype("f")
    B = numpy.random.randn(4, 7).astype("f")
    C = thinc_apple_ops.blas.gemm(A, B)
    assert C.shape == (A.shape[0], B.shape[1])

    C_out = numpy.empty((5, 7), dtype="f")
    thinc_apple_ops.blas.gemm(A, B, out=C_out)

    numpy.testing.assert_allclose(C, C_out)


def test_incorrect_output_size():
    A = numpy.ndarray((5, 4), dtype="f")
    B = numpy.ndarray((4, 7), dtype="f")

    with pytest.raises(ValueError, match=r"Shape mismatch for output matrix"):
        thinc_apple_ops.blas.gemm(A, B, out=numpy.ndarray((3, 7), dtype="f"))

    with pytest.raises(ValueError, match=r"Shape mismatch for output matrix"):
        thinc_apple_ops.blas.gemm(A, B, out=numpy.ndarray((5, 3), dtype="f"))


@pytest.mark.parametrize(
    "A_shape,B_shape,transA,transB",
    [
        [(0, 0), (0, 0), False, False],
        [(0, 0), (0, 0), True, False],
        [(0, 0), (0, 0), False, True],
        [(0, 0), (0, 0), True, True],
        [(0, 5), (5, 0), False, False],
        [(5, 0), (5, 0), False, True],
        [(5, 0), (5, 0), True, False],
    ],
)
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


@pytest.mark.parametrize(
    "A_shape,B_shape,transA,transB",
    [
        [(4, 5), (4, 5), False, False],
        [(5, 4), (4, 5), True, False],
        [(4, 5), (5, 4), False, True],
        [(5, 4), (5, 4), True, True],
    ],
)
def test_incorrect_shapes(A_shape, B_shape, transA, transB):
    A = numpy.ndarray(A_shape, dtype="f")
    B = numpy.ndarray(B_shape, dtype="f")
    with pytest.raises(ValueError, match=r"Shape mismatch"):
        thinc_apple_ops.blas.gemm(A, B, trans1=transA, trans2=transB)
