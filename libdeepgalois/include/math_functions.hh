#ifndef _MATH_FUNCTIONS_
#define _MATH_FUNCTIONS_
#include <cmath>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "types.h"

const float negative_slope = 0;

void vadd(const vec_t &a, const vec_t &b, vec_t &out); // vector add
void vadd(size_t n, const float_t *a, const float_t *b, float_t *out);
void vsub(const vec_t &a, const vec_t &b, vec_t &out);
void vmul(const vec_t &a, const vec_t &b, vec_t &out);
void vdiv(const vec_t &a, const vec_t &b, vec_t &out);
void add_scalar(const float_t alpha, vec_t &Y);
void sub_scalar(const float_t alpha, vec_t &Y);
void mul_scalar(const float_t alpha, vec_t &Y);
void mul_scalar(size_t n, const float_t alpha, const float_t *in, float_t *out);
void div_scalar(const float_t alpha, vec_t &Y);
float_t dot(const vec_t &x, const vec_t &y);
void mvmul(const vec_t &matrix, const vec_t &in_vector, vec_t &out_vector);
void vvmul(const vec_t &a, const vec_t &b, tensor_t &out);
void matadd(size_t x, size_t y, const tensor_t &A, const tensor_t &B, tensor_t &C);
void copy2D1D(const tensor_t &in, vec_t &out);
void copy1D1D(const vec_t &in, vec_t &out);
void copy1D1D(size_t len, const float_t *in, float_t *out);
void matmul2D(const tensor_t &A, const tensor_t &B, tensor_t &C);
void matmul1D1D(const size_t dim_x, const size_t dim_y, const size_t dim_z, const vec_t &A, const vec_t &B, vec_t &C); // matrix multiply
void matmul2D1D(const size_t dim_y, const tensor_t &A, const vec_t &B, vec_t &C);
void transpose2D(const tensor_t &in, tensor_t &out);
void transpose2D1D(const tensor_t &in, vec_t &out);
void transpose(size_t x, size_t y, const vec_t &in, vec_t &out);
void transpose(size_t x, size_t y, const float_t *in, float_t *out);
int argmax(const size_t n, const vec_t &x); // the arguments of the maxima
int argmax(const size_t n, const float_t *x); // the arguments of the maxima
void clear(vec_t &in);
void clear(size_t n, float_t *in);
void relu(const vec_t &in, vec_t &out); // ReLU
void relu(size_t n, const float_t *in, float_t *out); // ReLU
void d_relu(const vec_t &in_diff, const vec_t &data, vec_t &out_diff); // ReLU derivative
void dropout(const float scale, const float dropout_rate, const vec_t &in, std::vector<unsigned> &mask, vec_t &out); // dropout
void dropout(const float scale, const float dropout_rate, const vec_t &in, std::vector<unsigned> &mask, float_t *out);
void dropout(size_t n, const float scale, const float dropout_rate, const float_t *in, unsigned *mask, float_t *out);
void d_dropout(const float scale, const vec_t &in_diff, std::vector<unsigned> &mask, vec_t &out_diff); // dropout derivative
void d_dropout(size_t n, const float scale, const float_t *in_diff, unsigned *mask, float_t *out_diff);
void softmax(const vec_t &input, vec_t &output);
void softmax(size_t n, const float_t *input, float_t *output);
void d_softmax(const vec_t &y, const vec_t &p, vec_t &dy, const vec_t &dp);
void d_softmax(size_t n, const float_t *y, const float_t *p, float_t *dy, const float_t *dp);
float_t cross_entropy(const vec_t &y, const vec_t &p);
float_t cross_entropy(size_t n, const float_t *y, const float_t *p);
void d_cross_entropy(const vec_t &y, const vec_t &p, vec_t &d);
void d_cross_entropy(size_t n, const float_t *y, const float_t *p, float_t *d);

void vadd_gpu(const int n, const float_t *a, const float_t *b, float_t *out); // vector add
void relu_gpu(const int n, const float_t *in, float_t *out); // ReLU
void d_relu_gpu(const int n, const float_t *in_diff, const float_t *data, float_t *out_diff); // ReLU derivative
void dropout_gpu(const float scale, const float dropout_rate, const float_t *in, unsigned *mask, float_t *out); // dropout
void d_dropout_gpu(const float scale, const float_t *in_diff, const unsigned *mask, float_t *out_diff); // dropout derivative
void matmul1D1D_gpu(const size_t dim_x, const size_t dim_y, const size_t dim_z, const float_t *A, const float_t *B, float_t *C); // matrix multiply
int argmax_gpu(const size_t n, const float_t *x); // the arguments of the maxima

#endif
