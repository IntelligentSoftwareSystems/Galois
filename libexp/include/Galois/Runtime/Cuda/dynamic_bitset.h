// thread-safe dynamic bitset in CUDA
#pragma once
#include <cuda.h>
#include <math.h>

#ifdef __CUDA_ARCH__
__constant__
#endif
unsigned _byte_bit_count[256] = {
  0, /*   0 */ 1, /*   1 */ 1, /*   2 */ 2, /*   3 */ 1, /*   4 */
  2, /*   5 */ 2, /*   6 */ 3, /*   7 */ 1, /*   8 */ 2, /*   9 */
  2, /*  10 */ 3, /*  11 */ 2, /*  12 */ 3, /*  13 */ 3, /*  14 */
  4, /*  15 */ 1, /*  16 */ 2, /*  17 */ 2, /*  18 */ 3, /*  19 */
  2, /*  20 */ 3, /*  21 */ 3, /*  22 */ 4, /*  23 */ 2, /*  24 */
  3, /*  25 */ 3, /*  26 */ 4, /*  27 */ 3, /*  28 */ 4, /*  29 */
  4, /*  30 */ 5, /*  31 */ 1, /*  32 */ 2, /*  33 */ 2, /*  34 */
  3, /*  35 */ 2, /*  36 */ 3, /*  37 */ 3, /*  38 */ 4, /*  39 */
  2, /*  40 */ 3, /*  41 */ 3, /*  42 */ 4, /*  43 */ 3, /*  44 */
  4, /*  45 */ 4, /*  46 */ 5, /*  47 */ 2, /*  48 */ 3, /*  49 */
  3, /*  50 */ 4, /*  51 */ 3, /*  52 */ 4, /*  53 */ 4, /*  54 */
  5, /*  55 */ 3, /*  56 */ 4, /*  57 */ 4, /*  58 */ 5, /*  59 */
  4, /*  60 */ 5, /*  61 */ 5, /*  62 */ 6, /*  63 */ 1, /*  64 */
  2, /*  65 */ 2, /*  66 */ 3, /*  67 */ 2, /*  68 */ 3, /*  69 */
  3, /*  70 */ 4, /*  71 */ 2, /*  72 */ 3, /*  73 */ 3, /*  74 */
  4, /*  75 */ 3, /*  76 */ 4, /*  77 */ 4, /*  78 */ 5, /*  79 */
  2, /*  80 */ 3, /*  81 */ 3, /*  82 */ 4, /*  83 */ 3, /*  84 */
  4, /*  85 */ 4, /*  86 */ 5, /*  87 */ 3, /*  88 */ 4, /*  89 */
  4, /*  90 */ 5, /*  91 */ 4, /*  92 */ 5, /*  93 */ 5, /*  94 */
  6, /*  95 */ 2, /*  96 */ 3, /*  97 */ 3, /*  98 */ 4, /*  99 */
  3, /* 100 */ 4, /* 101 */ 4, /* 102 */ 5, /* 103 */ 3, /* 104 */
  4, /* 105 */ 4, /* 106 */ 5, /* 107 */ 4, /* 108 */ 5, /* 109 */
  5, /* 110 */ 6, /* 111 */ 3, /* 112 */ 4, /* 113 */ 4, /* 114 */
  5, /* 115 */ 4, /* 116 */ 5, /* 117 */ 5, /* 118 */ 6, /* 119 */
  4, /* 120 */ 5, /* 121 */ 5, /* 122 */ 6, /* 123 */ 5, /* 124 */
  6, /* 125 */ 6, /* 126 */ 7, /* 127 */ 1, /* 128 */ 2, /* 129 */
  2, /* 130 */ 3, /* 131 */ 2, /* 132 */ 3, /* 133 */ 3, /* 134 */
  4, /* 135 */ 2, /* 136 */ 3, /* 137 */ 3, /* 138 */ 4, /* 139 */
  3, /* 140 */ 4, /* 141 */ 4, /* 142 */ 5, /* 143 */ 2, /* 144 */
  3, /* 145 */ 3, /* 146 */ 4, /* 147 */ 3, /* 148 */ 4, /* 149 */
  4, /* 150 */ 5, /* 151 */ 3, /* 152 */ 4, /* 153 */ 4, /* 154 */
  5, /* 155 */ 4, /* 156 */ 5, /* 157 */ 5, /* 158 */ 6, /* 159 */
  2, /* 160 */ 3, /* 161 */ 3, /* 162 */ 4, /* 163 */ 3, /* 164 */
  4, /* 165 */ 4, /* 166 */ 5, /* 167 */ 3, /* 168 */ 4, /* 169 */
  4, /* 170 */ 5, /* 171 */ 4, /* 172 */ 5, /* 173 */ 5, /* 174 */
  6, /* 175 */ 3, /* 176 */ 4, /* 177 */ 4, /* 178 */ 5, /* 179 */
  4, /* 180 */ 5, /* 181 */ 5, /* 182 */ 6, /* 183 */ 4, /* 184 */
  5, /* 185 */ 5, /* 186 */ 6, /* 187 */ 5, /* 188 */ 6, /* 189 */
  6, /* 190 */ 7, /* 191 */ 2, /* 192 */ 3, /* 193 */ 3, /* 194 */
  4, /* 195 */ 3, /* 196 */ 4, /* 197 */ 4, /* 198 */ 5, /* 199 */
  3, /* 200 */ 4, /* 201 */ 4, /* 202 */ 5, /* 203 */ 4, /* 204 */
  5, /* 205 */ 5, /* 206 */ 6, /* 207 */ 3, /* 208 */ 4, /* 209 */
  4, /* 210 */ 5, /* 211 */ 4, /* 212 */ 5, /* 213 */ 5, /* 214 */
  6, /* 215 */ 4, /* 216 */ 5, /* 217 */ 5, /* 218 */ 6, /* 219 */
  5, /* 220 */ 6, /* 221 */ 6, /* 222 */ 7, /* 223 */ 3, /* 224 */
  4, /* 225 */ 4, /* 226 */ 5, /* 227 */ 4, /* 228 */ 5, /* 229 */
  5, /* 230 */ 6, /* 231 */ 4, /* 232 */ 5, /* 233 */ 5, /* 234 */
  6, /* 235 */ 5, /* 236 */ 6, /* 237 */ 6, /* 238 */ 7, /* 239 */
  4, /* 240 */ 5, /* 241 */ 5, /* 242 */ 6, /* 243 */ 5, /* 244 */
  6, /* 245 */ 6, /* 246 */ 7, /* 247 */ 5, /* 248 */ 6, /* 249 */
  6, /* 250 */ 7, /* 251 */ 6, /* 252 */ 7, /* 253 */ 7, /* 254 */
  8  /* 255 */
};

class DynamicBitset {
  size_t num_bits;
  size_t bit_vector_size;
  unsigned long long int *bit_vector;

  // assumes bit_vector is not updated in parallel
  __device__ __host__ size_t element_count(unsigned long long int *value) const {
    const unsigned char * byte_ptr = (const unsigned char *) value;
    size_t result = 0;
    for (unsigned i = 0; i < 8; ++i) {
      result += _byte_bit_count[byte_ptr[i]];
    }
    return result;
  }

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

  void clear() {
    CUDA_SAFE_CALL(cudaMemset(bit_vector, 0, bit_vector_size * sizeof(unsigned long long int)));
  }

  // assumes bit_vector is not updated in parallel
  __device__ bool test(size_t id) const {
    size_t bit_index = id/64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (id%64);
    return ((bit_vector[bit_index] & bit_offset) != 0);
  }

  __device__ void set(size_t id) {
    size_t bit_index = id/64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (id%64);
    atomicOr(&bit_vector[bit_index], bit_offset);
  }

  __device__ void reset(size_t id) {
    size_t bit_index = id/64;
    unsigned long long int bit_offset = 1;
    bit_offset <<= (id%64);
    atomicAnd(&bit_vector[bit_index], ~bit_offset);
  }

  // assumes bit_vector is not updated in parallel
  __device__ size_t count(size_t id) const {
    unsigned index = id/64;
    unsigned offset = id%64;
    size_t result = 0;
    for (unsigned i = 0; i < index; ++i) {
      result += element_count(bit_vector + i);
    }
    if (offset > 0) {
      unsigned long long int value = bit_vector[index];
      value <<= (64-offset);
      result += element_count(&value);
    }
    return result;
  }

  // assumes bit_vector is not updated in parallel
  __device__ size_t count() const {
    size_t result = 0;
    for (unsigned i = 0; i < bit_vector_size; ++i) {
      result += element_count(bit_vector + i);
    }
    return result;
  }

  size_t copy_to_cpu(unsigned long long int *bit_vector_cpu_copy) {
    assert(bit_vector_cpu_copy != NULL);
    CUDA_SAFE_CALL(cudaMemcpy(bit_vector_cpu_copy, bit_vector, bit_vector_size * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    size_t result = 0;
    for (unsigned i = 0; i < bit_vector_size; ++i) {
      result += element_count(bit_vector_cpu_copy + i);
    }
    return result;
  }

  void copy_to_gpu(unsigned long long int * cpu_bit_vector) {
    CUDA_SAFE_CALL(cudaMemcpy(bit_vector, cpu_bit_vector, bit_vector_size * sizeof(unsigned long long int), cudaMemcpyHostToDevice));
  }
};
