#pragma once
#include "Galois/Runtime/Cuda/dynamic_bitset.h"
#include "Galois/Runtime/Cuda/cuda_context.h"
#include "gg.h"

#ifdef __GALOIS_CUDA_CHECK_ERROR__
#define check_cuda_kernel check_cuda(cudaDeviceSynchronize()); check_cuda(cudaGetLastError());
#else
#define check_cuda_kernel check_cuda(cudaGetLastError());
#endif

enum SharedType { sharedMaster, sharedSlave };
enum UpdateOp { setOp, addOp, minOp };

void kernel_sizing(dim3 &blocks, dim3 &threads) {
	threads.x = 256;
	threads.y = threads.z = 1;
	blocks.x = ggc_get_nSM() * 8;
	blocks.y = blocks.z = 1;
}

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

void batch_get_updated_indices(index_type subset_size, unsigned int * __restrict__ indices, unsigned int * __restrict__ updated_indices, DynamicBitset * __restrict__ is_subset_updated, size_t * __restrict__ updated_size) {
  DynamicBitsetIterator flag_iterator(is_subset_updated);
  Shared<size_t> updated_size_ptr;
  updated_size_ptr.alloc(1);
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, indices, flag_iterator, updated_indices, updated_size_ptr.gpu_wr_ptr(true), subset_size);
	check_cuda_kernel;
  CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, indices, flag_iterator, updated_indices, updated_size_ptr.gpu_wr_ptr(true), subset_size);
	check_cuda_kernel;
  CUDA_SAFE_CALL(cudaFree(d_temp_storage));
  *updated_size = *updated_size_ptr.cpu_rd_ptr();
}

template<typename Type, SharedType sharedType, bool reset>
void batch_get_shared_field(struct CUDA_Context_Common *ctx, struct CUDA_Context_Field<Type> *field, unsigned from_id, unsigned long long int *bit_vector, Type *v, size_t *v_size, unsigned *data_mode, Type i = 0) {
  struct CUDA_Context_Shared *shared;
  Shared<Type> *shared_data;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
    shared_data = field->master_data;
  } else { // sharedSlave
    shared = &ctx->slave;
    shared_data = field->slave_data;
  }
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);
  //ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2"), timer3("timer3"), timer4("timer 4");
  //timer.start();
  //timer1.start();
  shared->is_updated[from_id].cpu_rd_ptr()->clear();
	batch_get_subset_bitset <<<blocks, threads>>>(shared->num_nodes[from_id], shared->nodes[from_id].gpu_rd_ptr(), shared->is_updated[from_id].gpu_rd_ptr(), field->is_updated.gpu_rd_ptr());
	check_cuda_kernel;
  //timer1.stop();
  //timer2.start();
  batch_get_updated_indices(shared->num_nodes[from_id], shared->nodes[from_id].gpu_rd_ptr(), shared->nodes_updated[from_id], shared->is_updated[from_id].gpu_rd_ptr(), v_size);
  //timer2.stop();
  //timer3.start();
  if (reset) {
    batch_get_reset_subset<Type> <<<blocks, threads>>>(*v_size, shared->nodes_updated[from_id], shared_data[from_id].gpu_wr_ptr(true), field->data.gpu_rd_ptr(), i);
  } else {
    batch_get_subset<Type> <<<blocks, threads>>>(*v_size, shared->nodes_updated[from_id], shared_data[from_id].gpu_wr_ptr(true), field->data.gpu_rd_ptr());
  }
	check_cuda_kernel;
  //timer3.stop();
  //timer4.start();
  if ((*v_size) == 0) {
    *data_mode = 0;
  } else {
    *data_mode = 1;
    shared->is_updated[from_id].cpu_rd_ptr()->copy_to_cpu(bit_vector);
    memcpy(v, shared_data[from_id].cpu_rd_ptr(), sizeof(unsigned int) * (*v_size));
  }
  //timer4.stop();
  //timer.stop();
  //fprintf(stderr, "Get_r %u->%u: %u bitset %u data. Time (ms): %llu + %llu + %llu + %llu = %llu\n",
  //  ctx->id, from_id, 
  //  shared->is_updated[from_id].cpu_rd_ptr()->alloc_size(), sizeof(unsigned int) * (*v_size),
  //  timer1.duration_ms(), timer2.duration_ms(), timer3.duration_ms(), timer4.duration_ms(),
  //  timer.duration_ms());
}

template<typename Type, SharedType sharedType, UpdateOp op>
void batch_set_shared_field(struct CUDA_Context_Common *ctx, struct CUDA_Context_Field<Type> *field, unsigned from_id, unsigned long long int *bit_vector, Type *v) {
  struct CUDA_Context_Shared *shared;
  Shared<Type> *shared_data;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
    shared_data = field->master_data;
  } else { // sharedSlave
    shared = &ctx->slave;
    shared_data = field->slave_data;
  }
	dim3 blocks;
	dim3 threads;
	kernel_sizing(blocks, threads);
  //ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2"), timer3("timer3"), timer4("timer 4");
  //timer.start();
  //timer1.start();
  shared->is_updated[from_id].cpu_rd_ptr()->copy_to_gpu(bit_vector);
  //timer1.stop();
  //timer2.start();
  size_t v_size;
  batch_get_updated_indices(shared->num_nodes[from_id], shared->nodes[from_id].gpu_rd_ptr(), shared->nodes_updated[from_id], shared->is_updated[from_id].gpu_rd_ptr(), &v_size);
  //timer2.stop();
  //timer3.start();
	memcpy(shared_data[from_id].cpu_wr_ptr(true), v, sizeof(unsigned int) * v_size);
  //timer3.stop();
  //timer4.start();
  if (op == setOp) {
    batch_set_subset<Type> <<<blocks, threads>>>(v_size, shared->nodes_updated[from_id], shared_data[from_id].gpu_rd_ptr(), field->data.gpu_wr_ptr());
  } else if (op == addOp) {
    batch_add_subset<Type> <<<blocks, threads>>>(v_size, shared->nodes_updated[from_id], shared_data[from_id].gpu_rd_ptr(), field->data.gpu_wr_ptr());
  } else if (op == minOp) {
    batch_min_subset<Type> <<<blocks, threads>>>(v_size, shared->nodes_updated[from_id], shared_data[from_id].gpu_rd_ptr(), field->data.gpu_wr_ptr());
  }
	check_cuda_kernel;
  //timer4.stop();
  //timer.stop();
  //fprintf(stderr, "Set %u<-%u: Time (ms): %llu + %llu + %llu + %llu = %llu\n",
  //  ctx->id, from_id, 
  //  timer1.duration_ms(), timer2.duration_ms(), timer3.duration_ms(), timer4.duration_ms(),
  //  timer.duration_ms());
}
