// thread-safe dynamic bitset in CUDA
#pragma once
#include <cuda.h>
#include <math.h>
#include <iterator>

class DynamicBitset {
  size_t num_bits;
  size_t bit_vector_size;
  unsigned long long int *bit_vector;

public:
  DynamicBitset()
  {
    num_bits = 0;
    bit_vector_size = 0;
    bit_vector = NULL;
  }

  DynamicBitset(size_t nbits)
  {
    alloc(nbits);
  }

  ~DynamicBitset() 
  {
    if (bit_vector != NULL) cudaFree(bit_vector);
  }

  void alloc(size_t nbits) {
    assert(num_bits == 0);
    assert(sizeof(unsigned long long int) * 8 == 64);
    num_bits = nbits;
    bit_vector_size = ceil((float)num_bits/64);
    CUDA_SAFE_CALL(cudaMalloc(&bit_vector, bit_vector_size * sizeof(unsigned long long int)));
    clear();
  }

  __device__ __host__ size_t size() const {
    return num_bits;
  }

  __device__ __host__ size_t alloc_size() const {
    return bit_vector_size * sizeof(unsigned long long int);
  }

  void clear() {
    CUDA_SAFE_CALL(cudaMemset(bit_vector, 0, bit_vector_size * sizeof(unsigned long long int)));
  }

  // assumes bit_vector is not updated (set) in parallel
  __device__ bool test(const size_t id) const {
    size_t bit_index = id/64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (id%64);
    return ((bit_vector[bit_index] & bit_offset) != 0);
  }

  __device__ void set(const size_t id) {
    size_t bit_index = id/64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (id%64);
    atomicOr(&bit_vector[bit_index], bit_offset);
  }

  void copy_to_cpu(unsigned long long int *bit_vector_cpu_copy) {
    assert(bit_vector_cpu_copy != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(bit_vector_cpu_copy, bit_vector, bit_vector_size * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
  }

  void copy_to_gpu(unsigned long long int * cpu_bit_vector) {
    assert(cpu_bit_vector != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(bit_vector, cpu_bit_vector, bit_vector_size * sizeof(unsigned long long int), cudaMemcpyHostToDevice));
  }
};

class DynamicBitsetIterator : public std::iterator<std::random_access_iterator_tag, bool> {
  DynamicBitset *bitset;
  size_t offset;

public:
  __device__ __host__ __forceinline__
  DynamicBitsetIterator(DynamicBitset *b, size_t i = 0) : bitset(b), offset(i) {}

  __device__ __host__ __forceinline__
  DynamicBitsetIterator& operator++() { offset++; return *this; }

  __device__ __host__ __forceinline__
  DynamicBitsetIterator& operator--() { offset--; return *this; }

  __device__ __host__ __forceinline__
  bool operator<(const DynamicBitsetIterator &bi) { return (offset < bi.offset); }

  __device__ __host__ __forceinline__
  bool operator<=(const DynamicBitsetIterator &bi) { return (offset <= bi.offset); }

  __device__ __host__ __forceinline__
  bool operator>(const DynamicBitsetIterator &bi) { return (offset > bi.offset); }

  __device__ __host__ __forceinline__
  bool operator>=(const DynamicBitsetIterator &bi) { return (offset >= bi.offset); }

  __device__ __host__ __forceinline__
  DynamicBitsetIterator& operator+=(size_t i) { offset += i; return *this; }

  __device__ __host__ __forceinline__
  DynamicBitsetIterator& operator-=(size_t i) { offset -= i; return *this; }

  __device__ __host__ __forceinline__
  DynamicBitsetIterator operator+(size_t i) { return DynamicBitsetIterator(bitset, offset + i); }

  __device__ __host__ __forceinline__
  DynamicBitsetIterator operator-(size_t i) { return DynamicBitsetIterator(bitset, offset - i); }

  __device__ __host__ __forceinline__
  difference_type operator-(const DynamicBitsetIterator &bi) { return (offset - bi.offset); }

  __device__ __forceinline__
  bool operator*() const { return bitset->test(offset); }

  __device__ __forceinline__
  bool operator[](const size_t id) const { return bitset->test(offset+id); }
};

#ifdef __HETEROGENEOUS_GALOIS_DEPRECATED__
class DynamicByteset {
  size_t byte_vector_size;
  char *byte_vector;

public:
  DynamicByteset()
  {
    byte_vector_size = 0;
    byte_vector = NULL;
  }

  DynamicByteset(size_t nbytes)
  {
    alloc(nbytes);
  }

  ~DynamicByteset() 
  {
    if (byte_vector != NULL) cudaFree(byte_vector);
  }

  void alloc(size_t nbytes) {
    assert(byte_vector_size == 0);
    byte_vector_size = nbytes;
    CUDA_SAFE_CALL(cudaMalloc(&byte_vector, byte_vector_size * sizeof(char)));
    clear();
  }

  __device__ __host__ size_t size() const {
    return byte_vector_size;
  }

  __device__ __host__ size_t alloc_size() const {
    return byte_vector_size * sizeof(char);
  }

  void clear() {
    CUDA_SAFE_CALL(cudaMemset(byte_vector, 0, byte_vector_size * sizeof(char)));
  }

  // assumes bit_vector is not updated (set) in parallel
  __device__ bool test(size_t id) const {
    return (byte_vector[id] != 0);
  }

  __device__ void set(size_t id) {
    byte_vector[id] = 1;
  }
};
#endif

