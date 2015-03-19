# Donald Nguyen <ddn@cs.utexas.edu>
#
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp
cimport numpy as np
cimport svm


np.import_array()


def _train_linear_sparse(
        X, 
        np.ndarray[np.float64_t, ndim=1, mode='c'] Y, 
        loss, penalty, bint is_dual, 
        double tol, double bias, double C, 
        np.ndarray[np.float64_t, ndim=1] class_weight, int max_iter,
        unsigned random_seed):
    cdef np.float64_t* X_data = (<np.float64_t*> (<np.ndarray[np.float64_t, ndim=1, mode='c']> X.data).data)
    cdef int X_data_len = X.data.shape[0]
    cdef np.int32_t* X_ind = (<np.int32_t*> (<np.ndarray[np.int32_t, ndim=1, mode='c']> X.indices).data)
    cdef np.int32_t* X_indptr = (<np.int32_t*> (<np.ndarray[np.int32_t, ndim=1, mode='c']> X.indptr).data)
    cdef int X_ind_len = X.indices.shape[0]
    cdef bint l1_loss = True if loss == 'l1' else False
    cdef bint l1_penalty = True if penalty == 'l1' else False
    cdef np.float64_t* Y_data = (<np.float64_t*> Y.data)
    cdef int Y_data_len = Y.data.shape[1]
    cdef np.float64_t* class_weight_data = (<np.float64_t*> class_weight.data)
    cdef int class_weight_data_len = class_weight.shape[0]
    cdef bint error = False

    with nogil:
        if l1_penalty and l1_loss and is_dual:
            train_linear_l1_l1_dual(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif l1_penalty and l1_loss and not is_dual:
            train_linear_l1_l1_primal(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif l1_penalty and not l1_loss and is_dual:
            train_linear_l1_l2_dual(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif l1_penalty and not l1_loss and not is_dual:
            train_linear_l1_l2_primal(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif not l1_penalty and l1_loss and is_dual:
            train_linear_l2_l1_dual(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif not l1_penalty and l1_loss and not is_dual:
            train_linear_l2_l1_primal(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif not l1_penalty and not l1_loss and is_dual:
            train_linear_l2_l2_dual(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        elif not l1_penalty and not l1_loss and not is_dual:
            train_linear_l2_l2_primal(X_data, X_data_len,
                    X_ind, X_indptr, X_ind_len,
                    Y_data, Y_data_len,
                    tol, bias, C, 
                    class_weight_data, class_weight_data_len,
                    max_iter, random_seed)
        else:
            error = True
    if error:
        raise ValueError('Unknown algorithm')


def _train_linear_dense(
        np.ndarray[np.float64_t, ndim=2, mode='c'] X,
        np.ndarray[np.float64_t, ndim=1, mode='c'] Y, 
        loss, penalty, bint is_dual, 
        double tol, double bias, double C, 
        np.ndarray[np.float64_t, ndim=1] class_weight, int max_iter,
        unsigned random_seed):
    raise ValueError('Dense features not yet supported')


def train_linear(X, np.ndarray[np.float64_t, ndim=1, mode='c'] Y, 
                 loss, penalty, bint is_dual, 
                 double tol, double bias, double C, 
                 np.ndarray[np.float64_t, ndim=1] class_weight, int max_iter,
                 unsigned random_seed):
    """Train svm
    """
    if sp.isspmatrix(X):
        _train_linear_sparse(X, Y, loss, penalty, is_dual, tol, bias, C,
                class_weight, max_iter, random_seed)
    else:
        _train_linear_dense(X, Y, loss, penalty, is_dual, tol, bias, C,
                class_weight, max_iter, random_seed)

    return (0, 0)

# vim: set ts=4 sw=4:
