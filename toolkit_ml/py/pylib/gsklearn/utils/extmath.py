# Authors: Gael Varoquaux
# Alexandre Gramfort
# Alexandre T. Passos
# Olivier Grisel
# Lars Buitinck
# Stefan van der Walt
# Kyle Kastner
# License: BSD 3 clause

from __future__ import division
import warnings

import numpy as np
from scipy import linalg
from scipy.sparse import issparse

from .fixes import np_version
from .validation import check_array

def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    """
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return fast_dot(a, b)


def _impose_f_order(X):
    """Helper Function"""
    # important to access flags instead of calling np.isfortran,
    # this catches corner cases.
    if X.flags.c_contiguous:
        return check_array(X.T, copy=False, order='F'), True
    else:
        return check_array(X, copy=False, order='F'), False


def _fast_dot(A, B):
    if B.shape[0] != A.shape[A.ndim - 1]:  # check adopted from '_dotblas.c'
        raise ValueError

    if A.dtype != B.dtype or any(x.dtype not in (np.float32, np.float64)
                                 for x in [A, B]):
        warnings.warn('Data must be of same type. Supported types '
                      'are 32 and 64 bit float. '
                      'Falling back to np.dot.', NonBLASDotWarning)
        raise ValueError

    if min(A.shape) == 1 or min(B.shape) == 1 or A.ndim != 2 or B.ndim != 2:
        raise ValueError

    # scipy 0.9 compliant API
    dot = linalg.get_blas_funcs(['gemm'], (A, B))[0]
    A, trans_a = _impose_f_order(A)
    B, trans_b = _impose_f_order(B)
    return dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)


def _have_blas_gemm():
    try:
        linalg.get_blas_funcs(['gemm'])
        return True
    except (AttributeError, ValueError):
        warnings.warn('Could not import BLAS, falling back to np.dot')
        return False


# Only use fast_dot for older NumPy; newer ones have tackled the speed issue.
if np_version < (1, 7, 2) and _have_blas_gemm():
    def fast_dot(A, B):
        """Compute fast dot products directly calling BLAS.

        This function calls BLAS directly while warranting Fortran contiguity.
        This helps avoiding extra copies `np.dot` would have created.
        For details see section `Linear Algebra on large Arrays`:
        http://wiki.scipy.org/PerformanceTips

        Parameters
        ----------
        A, B: instance of np.ndarray
            Input arrays. Arrays are supposed to be of the same dtype and to
            have exactly 2 dimensions. Currently only floats are supported.
            In case these requirements aren't met np.dot(A, B) is returned
            instead. To activate the related warning issued in this case
            execute the following lines of code:

            >> import warnings
            >> from sklearn.utils.validation import NonBLASDotWarning
            >> warnings.simplefilter('always', NonBLASDotWarning)
        """
        try:
            return _fast_dot(A, B)
        except ValueError:
            # Maltyped or malformed data.
            return np.dot(A, B)
else:
    fast_dot = np.dot

# vim: set ts=4 sw=4:
