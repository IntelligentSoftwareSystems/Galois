# Author: Donald Nguyen <ddn@cs.utexas.edu>
# License: BSD 3 clause

import numpy as np

from .validation import (as_float_array, check_X_y, check_array, check_random_state,
    check_consistent_length, column_or_1d)
from .class_weight import compute_class_weight


__all__ = [
        'as_float_array', 
        'check_array', 
        'check_consistent_length', 
        'check_random_state',
        'check_X_y'
        'column_or_1d', 
        'compute_class_weight',
        ]


def safe_mask(X, mask):
    """Return a mask which is safe to use on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.

    mask: array
        Mask to be used on X.

    Returns
    -------
        mask
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.int):
        return mask

    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask


class ConvergenceWarning(Warning):
    "Custom warning to capture convergence problems"


# vim: set ts=4 sw=4:
