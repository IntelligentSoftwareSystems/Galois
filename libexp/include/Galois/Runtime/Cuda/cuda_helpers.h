#pragma once
#include <cuda.h>

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
