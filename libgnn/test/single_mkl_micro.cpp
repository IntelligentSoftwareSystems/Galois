#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <mkl.h>

#ifdef USE_SHARED_GALOIS
#include "galois/Galois.h"
#include "galois/PODResizeableArray.h"
#endif
#ifdef USE_DIST_GALOIS
#include "galois/DistGalois.h"
#include "galois/PODResizeableArray.h"
#endif

#ifdef USE_OMP
#include "omp.h"
#endif

// MKL wrapper
#ifdef USE_OMP
void CBlasSGEMMOMP(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                size_t input_rows, size_t input_columns, size_t output_columns,
                const float* a, const float* b, float* output) {
  // set lead dimension based on cblas spec w.r.t. transpose setting
  size_t lead_dim_a = (trans_a == CblasNoTrans) ? input_columns : input_rows;
  size_t lead_dim_b =
      (trans_b == CblasNoTrans) ? output_columns : input_columns;

  #pragma omp parallel for
  for (int i = 0; i < omp_get_num_threads(); i++) {
    unsigned chunk_size = input_rows / omp_get_num_threads();
    unsigned my_start = chunk_size * i;
    unsigned my_end = chunk_size * (i + 1);
    if (omp_get_num_threads() - 1 == i) {
      my_end = input_rows;
    }
    unsigned rows_to_use = my_end - my_start;

    const float* my_a = a + (my_start * input_columns);
    float* my_output = output + (my_start * output_columns);

    // do the MM
    cblas_sgemm(CblasRowMajor, trans_a, trans_b, rows_to_use, output_columns,
                input_columns, 1.0, my_a, lead_dim_a, b, lead_dim_b,
                false ? 1.0 : 0.0, my_output, output_columns);
  }
}
#endif

#if defined(USE_SHARED_GALOIS) || defined(USE_DIST_GALOIS)
void CBlasSGEMMGalois(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                size_t input_rows, size_t input_columns, size_t output_columns,
                const float* a, const float* b, float* output) {
  // set lead dimension based on cblas spec w.r.t. transpose setting
  size_t lead_dim_a = (trans_a == CblasNoTrans) ? input_columns : input_rows;
  size_t lead_dim_b =
      (trans_b == CblasNoTrans) ? output_columns : input_columns;

  static std::vector<galois::PODResizeableArray<float>> temps;
  if (trans_a == CblasTrans) {
    temps.resize(galois::getActiveThreads());
  }

  galois::on_each(
    [&] (size_t i, size_t num_threads) {
      if (trans_a != CblasTrans) {
        unsigned chunk_size = input_rows / num_threads;
        unsigned my_start = chunk_size * i;
        unsigned my_end = chunk_size * (i + 1);
        if (num_threads - 1 == i) {
          my_end = input_rows;
        }
        unsigned rows_to_use = my_end - my_start;

        const float* my_a = a + (my_start * input_columns);
        float* my_output = output + (my_start * output_columns);

        // do the MM
        cblas_sgemm(CblasRowMajor, trans_a, trans_b, rows_to_use, output_columns,
                    input_columns, 1.0, my_a, lead_dim_a, b, lead_dim_b,
                    false ? 1.0 : 0.0, my_output, output_columns);
      } else {
        galois::PODResizeableArray<float>& my_pod = temps[i];
        my_pod.resize(input_rows * output_columns);

        unsigned chunk_size = input_columns / num_threads;
        unsigned my_start = chunk_size * i;
        unsigned my_end = chunk_size * (i + 1);
        if (num_threads - 1 == i) {
          my_end = input_columns;
        }
        unsigned b_rows_to_use = my_end - my_start;

        const float* my_a = a + (my_start * input_rows);
        const float* my_b = b + (my_start * output_columns);

        // do the MM
        cblas_sgemm(CblasRowMajor, trans_a, trans_b, input_rows, output_columns,
                    b_rows_to_use, 1.0, my_a, lead_dim_a, my_b, lead_dim_b,
                    false ? 1.0 : 0.0, my_pod.data(), output_columns);
      }
    }
  );

  if (trans_a == CblasTrans) {
    printf("Manual summation\n");
    for (galois::PODResizeableArray<float>& temp_out : temps) {
      for (unsigned i = 0; i < temp_out.size(); i++) {
        output[i] += temp_out[i];
      }
    }
  }
}
#endif


void CacheFlush(std::vector<float>* matrix) {
  for (size_t i = 0; i < matrix->size(); i++) {
    (*matrix)[i] = i;
  }
}

int main(int argc, char* argv[]) {
#ifdef USE_SHARED_GALOIS
  galois::SharedMemSys G;
  if (argc != 2) {
    printf("Thread arg not specified\n");
    exit(1);
  }
  galois::setActiveThreads(std::stoi(argv[1]));
  printf("Initialized Galois Shared Mem with %u threads\n",
         galois::getActiveThreads());
#endif

#ifdef USE_DIST_GALOIS
  galois::DistMemSys G;
  if (argc != 2) {
    printf("Thread arg not specified\n");
    exit(1);
  }
  galois::setActiveThreads(std::stoi(argv[1]));
  printf("Initialized Galois Dist Mem with %u threads\n",
         galois::getActiveThreads());
#endif

  printf("%d %s\n", argc, argv[0]);

  // dimensions from test case
  size_t a_dim = 12000000;
  size_t b_dim = 128;
  size_t c_dim = 16;

  // inputs
  std::vector<float> matrix_1(a_dim * b_dim);
  std::vector<float> matrix_2(a_dim * c_dim);
  // output
  //std::vector<float> matrix_3(a_dim * c_dim);
  std::vector<float> matrix_3(b_dim * c_dim);

  size_t kBigSize = 1000000000;
  std::vector<float> very_big_matrix(kBigSize);

  // change reps here; maybe make it command line arg
  for (size_t reps = 0; reps < 5; reps++) {
    // reinit
    srand(0);
    for (size_t i = 0; i < matrix_1.size(); i++) {
      matrix_1[i] = rand() / static_cast<float>(RAND_MAX / 10);
    }
    srand(1);
    for (size_t i = 0; i < matrix_2.size(); i++) {
      matrix_2[i] = rand() / static_cast<float>(RAND_MAX / 10);
    }

    very_big_matrix.clear();
    very_big_matrix.resize(kBigSize);
    // cache flush
    CacheFlush(&very_big_matrix);

    printf("Rep %lu\n", reps);

    auto start = std::chrono::high_resolution_clock::now();
    // transpose because it's the same as the problematic call in GNN
    // TODO(loc) non transpose version
#ifdef USE_OMP
    CBlasSGEMMOMP(CblasNoTrans, CblasNoTrans, a_dim, b_dim, c_dim, matrix_1.data(),
               matrix_2.data(), matrix_3.data());
#endif
#if defined(USE_SHARED_GALOIS) || defined(USE_DIST_GALOIS)
    //CBlasSGEMMGalois(CblasNoTrans, CblasNoTrans, a_dim, b_dim, c_dim, matrix_1.data(),
    //           matrix_2.data(), matrix_3.data());
    CBlasSGEMMGalois(CblasTrans, CblasNoTrans, b_dim, a_dim, c_dim, matrix_1.data(),
               matrix_2.data(), matrix_3.data());
#endif
    //CBlasSGEMM(CblasTrans, CblasNoTrans, b_dim, a_dim, c_dim, matrix_1.data(),
    //           matrix_2.data(), matrix_3.data());
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::time_point_cast<std::chrono::milliseconds>(stop) - 
                    std::chrono::time_point_cast<std::chrono::microseconds>(start);
    printf("Run duration is %lf ms\n", duration.count() / 1000.0);
  }

  return 0;
}
