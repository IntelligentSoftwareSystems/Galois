cimport numpy as np

cdef extern from "toolkit_ml/toolkit_ml.h" namespace "toolkit_ml":
    cdef void train_linear_l1_l1_dual(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l1_l1_primal(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l1_l2_dual(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l1_l2_primal(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l2_l1_dual(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l2_l1_primal(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l2_l2_dual(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil
    cdef void train_linear_l2_l2_primal(
            double* X_data, int X_data_len,
            np.int32_t* X_ind, np.int32_t* X_indptr, int X_ind_len,
            double* Y_data, int Y_data_len,
            double tol, double bias, double C,
            double* class_weight_data, int class_weight_data_len,
            int max_iter, unsigned random_seed) nogil

# vim: set ts=4 sw=4:
