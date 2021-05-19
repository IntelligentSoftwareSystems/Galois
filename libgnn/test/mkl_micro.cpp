#include <cstdlib>
#include <vector>
#include <random>
#include <mkl.h>

#ifdef USE_SHARED_GALOIS
#include "galois/Galois.h"
#include "galois/LargeArray.h"
#endif
#ifdef USE_DIST_GALOIS
#include "galois/DistGalois.h"
#include "galois/LargeArray.h"
#endif
#ifdef USE_SHARED_GALOIS_DELETE
#include "galois/Galois.h"
#endif

#ifdef USE_OMP
#include "omp.h"
#endif

// MKL wrapper
void CBlasSGEMM(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                size_t input_rows, size_t input_columns, size_t output_columns,
                const float* a, const float* b, float* output) {
  // set lead dimension based on cblas spec w.r.t. transpose setting
  size_t lead_dim_a = (trans_a == CblasNoTrans) ? input_columns : input_rows;
  size_t lead_dim_b =
      (trans_b == CblasNoTrans) ? output_columns : input_columns;
  // do the MM
  cblas_sgemm(CblasRowMajor, trans_a, trans_b, input_rows, output_columns,
              input_columns, 1.0, a, lead_dim_a, b, lead_dim_b,
              false ? 1.0 : 0.0, output, output_columns);
}

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

#ifdef USE_SHARED_GALOIS_DELETE
  std::unique_ptr<galois::SharedMemSys> G;
  G = std::make_unique<galois::SharedMemSys>();

  if (argc != 2) {
    printf("Thread arg not specified\n");
    exit(1);
  }
  galois::setActiveThreads(std::stoi(argv[1]));
  printf("Initialized Galois Shared Mem with %u threads\n",
         galois::getActiveThreads());
  printf("Deleting galois\n");
  G.reset();
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

#if defined(USE_SHARED_GALOIS) || defined(USE_DIST_GALOIS)
  printf("Using Galois large arrays\n");
  // inputs
  galois::LargeArray<float> matrix_1;
  matrix_1.create(a_dim * b_dim);
  galois::LargeArray<float> matrix_2;
  matrix_2.create(a_dim * c_dim);
  // output
  galois::LargeArray<float> matrix_3;
  matrix_3.create(b_dim * c_dim);
#else
  // inputs
  std::vector<float> matrix_1(a_dim * b_dim);
  std::vector<float> matrix_2(a_dim * c_dim);
  // output
  std::vector<float> matrix_3(b_dim * c_dim);
#endif

  size_t kBigSize = 1000000000;
  std::vector<float> very_big_matrix(kBigSize);

  // change reps here; maybe make it command line arg
  for (size_t reps = 0; reps < 3; reps++) {
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

    // dummy OMP TBB loop
#ifdef USE_OMP
#pragma omp parallel
    for (size_t i = 0; i < very_big_matrix.size(); i++) {
      very_big_matrix[i] = i;
    }
#endif

    printf("Rep %lu\n", reps);

    // transpose because it's the same as the problematic call in GNN
    // TODO(loc) non transpose version
    CBlasSGEMM(CblasTrans, CblasNoTrans, b_dim, a_dim, c_dim, matrix_1.data(),
               matrix_2.data(), matrix_3.data());
  }

  return 0;
}
