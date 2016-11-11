#pragma once
#include <cuda.h>
#include "Galois/Runtime/Cuda/dynamic_bitset.h"

template<typename Type>
__global__ void batch_get_subset(index_type subset_size, const unsigned int * __restrict__ indices, Type * __restrict__ subset, const Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
		subset[src] = array[index];
	}
}

template<typename Type>
__global__ void batch_get_reset_subset(index_type subset_size, const unsigned int * __restrict__ indices, Type * __restrict__ subset, Type * __restrict__ array, Type reset_value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
		subset[src] = array[index];
		array[index] = reset_value;
	}
}

template<typename Type>
__global__ void batch_set_subset(index_type subset_size, const unsigned int * __restrict__ indices, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
		array[index] = subset[src];
	}
}

template<typename Type>
__global__ void batch_add_subset(index_type subset_size, const unsigned int * __restrict__ indices, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
		array[index] += subset[src];
	}
}

template<typename Type>
__global__ void batch_min_subset(index_type subset_size, const unsigned int * __restrict__ indices, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
		array[index] = (array[index] > subset[src]) ? subset[src] : array[index];
	}
}

template<typename Type>
__global__ void batch_max_subset(index_type subset_size, const unsigned int * __restrict__ indices, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
		array[index] = (array[index] < subset[src]) ? subset[src] : array[index];
	}
}

template<typename Type>
__global__ void batch_get_subset_bitset(index_type subset_size, const unsigned int * __restrict__ indices, DynamicBitset * __restrict__ is_subset_updated, DynamicBitset * __restrict__ is_array_updated) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (is_array_updated->test(index)) {
      is_subset_updated->set(src);
    }
	}
}

template<typename Type>
__global__ void batch_get_subset_conditional(index_type subset_size, const unsigned int * __restrict__ indices, const DynamicBitset * __restrict__ is_subset_updated, Type * __restrict__ subset, const Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    if (is_subset_updated->test(src)) {
      unsigned index = indices[src];
      index_type i = is_subset_updated->count(src);
      subset[i] = array[index];
    }
	}
}

template<typename Type>
__global__ void batch_get_reset_subset_conditional(index_type subset_size, const unsigned int * __restrict__ indices, const DynamicBitset * __restrict__ is_subset_updated, Type * __restrict__ subset, Type * __restrict__ array, Type reset_value) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    if (is_subset_updated->test(src)) {
      unsigned index = indices[src];
      index_type i = is_subset_updated->count(src);
      subset[i] = array[index];
      array[index] = reset_value;
    }
	}
}

template<typename Type>
__global__ void batch_set_subset_conditional(index_type subset_size, const unsigned int * __restrict__ indices, const DynamicBitset * __restrict__ is_subset_updated, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    if (is_subset_updated->test(src)) {
      unsigned index = indices[src];
      index_type i = is_subset_updated->count(src);
      array[index] = subset[i];
    }
	}
}

template<typename Type>
__global__ void batch_add_subset_conditional(index_type subset_size, const unsigned int * __restrict__ indices, const DynamicBitset * __restrict__ is_subset_updated, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    if (is_subset_updated->test(src)) {
      unsigned index = indices[src];
      index_type i = is_subset_updated->count(src);
      array[index] += subset[i];
    }
	}
}

template<typename Type>
__global__ void batch_min_subset_conditional(index_type subset_size, const unsigned int * __restrict__ indices, const DynamicBitset * __restrict__ is_subset_updated, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    if (is_subset_updated->test(src)) {
      unsigned index = indices[src];
      index_type i = is_subset_updated->count(src);
      array[index] = (array[index] > subset[i]) ? subset[i] : array[index];
    }
	}
}

template<typename Type>
__global__ void batch_max_subset_conditional(index_type subset_size, const unsigned int * __restrict__ indices, const DynamicBitset * __restrict__ is_subset_updated, const Type * __restrict__ subset, Type * __restrict__ array) {
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	index_type src_end = subset_size;
	for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    if (is_subset_updated->test(src)) {
      unsigned index = indices[src];
      index_type i = is_subset_updated->count(src);
      array[index] = (array[index] < subset[i]) ? subset[i] : array[index];
    }
	}
}
