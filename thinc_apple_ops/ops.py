from typing import Optional
import numpy
from thinc.api import NumpyOps
from thinc.types import Floats2d
from . import blas

    
class AppleOps(NumpyOps):
    """Thinc Ops class that calls into Apple's native libraries for some
    operations. Other operations fall back to numpy."""
    name = "apple"
    xp = numpy

    def gemm(
        self,
        x: Floats2d,
        y: Floats2d,
        out: Optional[Floats2d] = None,
        trans1: bool = False,
        trans2: bool = False,
    ) -> Floats2d:
        """Perform General Matrix Multiplication (GeMM) and optionally store
        the result in the specified output variable.
        """
        C = blas.gemm(x, y, trans1=trans1, trans2=trans2)
        if out is None:
            return C
        else:
            out[:] = C
            return out
