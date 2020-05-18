
#pragma once
#include <cuda.h>
#include <assert.h>
#include "pangolin/types.cuh"
#include "pangolin/cutils.h"

class Bitsets {
public:
  int num_sets;
  int num_bits_capacity;
  int num_bits;
  uint64_t** h_bit_vectors;
  uint64_t** d_bit_vectors;
  Bitsets() {}
  Bitsets(int n, int nbits) { alloc(n, nbits); }
  ~Bitsets() {}
  void set_size(int n, int nbits) {
    num_sets          = n;
    num_bits_capacity = nbits;
    num_bits          = nbits;
  }
  void alloc(int n, int nbits) {
    assert(sizeof(unsigned long long int) * 8 == 64);
    assert(sizeof(uint64_t) * 8 == 64);
    num_sets          = n;
    num_bits_capacity = nbits;
    num_bits          = nbits;
    h_bit_vectors     = (uint64_t**)malloc(n * sizeof(uint64_t*));
    for (int i = 0; i < n; i++) {
      CUDA_SAFE_CALL(
          cudaMalloc(&h_bit_vectors[i], vec_size() * sizeof(uint64_t)));
      reset(i);
    }
    CUDA_SAFE_CALL(cudaMalloc(&d_bit_vectors, n * sizeof(uint64_t*)));
    CUDA_SAFE_CALL(cudaMemcpy(d_bit_vectors, h_bit_vectors,
                              n * sizeof(uint64_t*), cudaMemcpyHostToDevice));
  }
  void clear() {
    for (int i = 0; i < num_sets; i++)
      reset(i);
    CUDA_SAFE_CALL(cudaMemcpy(d_bit_vectors, h_bit_vectors,
                              num_sets * sizeof(uint64_t*),
                              cudaMemcpyHostToDevice));
  }
  void clean() {
    for (int i = 0; i < num_sets; i++)
      if (h_bit_vectors[i] != NULL)
        cudaFree(h_bit_vectors[i]);
    if (d_bit_vectors != NULL)
      cudaFree(d_bit_vectors);
    if (h_bit_vectors != NULL)
      free(h_bit_vectors);
  }
  void reset(int i) {
    CUDA_SAFE_CALL(
        cudaMemset(h_bit_vectors[i], 0, vec_size() * sizeof(uint64_t)));
  }
  __device__ void set(int sid, int bid) {
    if (sid >= num_sets)
      printf("sid=%d, num_sets=%d\n", sid, num_sets);
    assert(sid < num_sets);
    assert(bid < num_bits);
    int bit_index                     = bid / 64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (bid % 64);
    if ((d_bit_vectors[sid][bit_index] & bit_offset) == 0) { // test and set
      atomicOr((unsigned long long int*)&d_bit_vectors[sid][bit_index],
               bit_offset);
    }
  }
  __device__ int count_num_ones(int sid, size_t bid) {
    return __popcll(d_bit_vectors[sid][bid]);
  }
  __device__ __host__ size_t vec_size() const {
    size_t bit_vector_size = (num_bits + 63) / 64;
    return bit_vector_size;
  }
};
