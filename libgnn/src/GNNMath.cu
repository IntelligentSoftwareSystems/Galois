#include "galois/GNNMath.cuh"

bool galois::cublas_is_init = false;
cublasHandle_t galois::global_cublas_handle;

void galois::InitCuBLAS() { CUBLAS_CHECK(cublasCreate(&global_cublas_handle)); }

void galois::CBlasSGEMMGPU(const cublasOperation_t trans_a,
                           const cublasOperation_t trans_b, size_t input_rows,
                           size_t input_columns, size_t output_columns,
                           const GNNFloat* a, const GNNFloat* b,
                           GNNFloat* output) {
  if (!cublas_is_init) {
    InitCuBLAS();
    cublas_is_init = true;
  }
  size_t lead_dim_a = (trans_a == CUBLAS_OP_N) ? input_columns : input_rows;
  size_t lead_dim_b = (trans_b == CUBLAS_OP_N) ? output_columns : input_columns;
  float dummy0      = 0.0;
  float dummy1      = 1.0;
  // because cusparse assumes column major even though we're passing in row
  // major, the order of multiply is reversed so that it does what we
  // want anyways
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  CUBLAS_CHECK(cublasSgemm(global_cublas_handle, trans_b, trans_a,
                           output_columns, input_rows, input_columns, &dummy1,
                           b, lead_dim_b, a, lead_dim_a, &dummy0, output,
                           output_columns));
}
