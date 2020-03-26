/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifndef __DEV_EDGE_SYNC__
#define __DEV_EDGE_SYNC__
/**
 * @file DeviceEdgeSync.h
 *
 * CUDA header for GPU runtime
 *
 * @todo better file description + document this file
 */
#pragma once
#include "galois/cuda/DynamicBitset.h"
#include "galois/cuda/EdgeContext.h"
#include "galois/runtime/DataCommMode.h"
#include "cub/util_allocator.cuh"

#ifdef __GALOIS_CUDA_CHECK_ERROR__
#define check_cuda_kernel                                                      \
  check_cuda(cudaDeviceSynchronize());                                         \
  check_cuda(cudaGetLastError());
#else
#define check_cuda_kernel check_cuda(cudaGetLastError());
#endif

enum SharedType { sharedMaster, sharedMirror };
enum UpdateOp { setOp, addOp, minOp };

void kernel_sizing(dim3& blocks, dim3& threads) {
  threads.x = 256;
  threads.y = threads.z = 1;
  blocks.x              = ggc_get_nSM() * 8;
  blocks.y = blocks.z = 1;
}

template <typename DataType>
__global__ void batch_get_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 DataType* __restrict__ subset,
                                 const DataType* __restrict__ array) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    subset[src]    = array[index];
  }
}

template <typename DataType, typename OffsetIteratorType>
__global__ void batch_get_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 DataType* __restrict__ subset,
                                 const DataType* __restrict__ array) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    subset[src]    = array[index];
  }
}

template <typename DataType>
__global__ void batch_get_reset_subset(index_type subset_size,
                                       const unsigned int* __restrict__ indices,
                                       DataType* __restrict__ subset,
                                       DataType* __restrict__ array,
                                       DataType reset_value) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    subset[src]    = array[index];
    array[index]   = reset_value;
  }
}

template <typename DataType, typename OffsetIteratorType>
__global__ void batch_get_reset_subset(index_type subset_size,
                                       const unsigned int* __restrict__ indices,
                                       const OffsetIteratorType offsets,
                                       DataType* __restrict__ subset,
                                       DataType* __restrict__ array,
                                       DataType reset_value) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    subset[src]    = array[index];
    array[index]   = reset_value;
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_set_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    array[index]   = subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_set_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    array[index]   = subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_add_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    array[index] += subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_add_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    array[index] += subset[src];
    if (sharedType != sharedMirror) {
      is_array_updated->set(index);
    }
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_min_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (array[index] > subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_min_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    if (array[index] > subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType, SharedType sharedType>
__global__ void batch_max_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (array[index] < subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType, SharedType sharedType, typename OffsetIteratorType>
__global__ void batch_max_subset(index_type subset_size,
                                 const unsigned int* __restrict__ indices,
                                 const OffsetIteratorType offsets,
                                 const DataType* __restrict__ subset,
                                 DataType* __restrict__ array,
                                 DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[offsets[src]];
    if (array[index] < subset[src]) {
      array[index] = subset[src];
      if (sharedType != sharedMirror) {
        is_array_updated->set(index);
      }
    }
  }
}

template <typename DataType>
__global__ void batch_reset(DataType* __restrict__ array, index_type begin,
                            index_type end, DataType val) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = end;
  for (index_type src = begin + tid; src < src_end; src += nthreads) {
    array[src] = val;
  }
}

__global__ void
batch_get_subset_bitset(index_type subset_size,
                        const unsigned int* __restrict__ indices,
                        DynamicBitset* __restrict__ is_subset_updated,
                        DynamicBitset* __restrict__ is_array_updated) {
  unsigned tid       = TID_1D;
  unsigned nthreads  = TOTAL_THREADS_1D;
  index_type src_end = subset_size;
  for (index_type src = 0 + tid; src < src_end; src += nthreads) {
    unsigned index = indices[src];
    if (is_array_updated->test(index)) {
      is_subset_updated->set(src);
    }
  }
}

// inclusive range
__global__ void bitset_reset_range(DynamicBitset* __restrict__ bitset,
                                   size_t vec_begin, size_t vec_end, bool test1,
                                   size_t bit_index1, uint64_t mask1,
                                   bool test2, size_t bit_index2,
                                   uint64_t mask2) {
  unsigned tid      = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  for (size_t src = vec_begin + tid; src < vec_end; src += nthreads) {
    bitset->batch_reset(src);
  }

  if (tid == 0) {
    if (test1) {
      bitset->batch_bitwise_and(bit_index1, mask1);
    }
    if (test2) {
      bitset->batch_bitwise_and(bit_index2, mask2);
    }
  }
}

template <typename DataType>
void reset_bitset_field(struct CUDA_Context_Field_Edges<DataType>* field,
                        size_t begin, size_t end) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);
  const DynamicBitset* bitset_cpu = field->is_updated.cpu_rd_ptr();
  assert(begin <= (bitset_cpu->size() - 1));
  assert(end <= (bitset_cpu->size() - 1));

  size_t vec_begin = (begin + 63) / 64;
  size_t vec_end;

  if (end == (bitset_cpu->size() - 1))
    vec_end = bitset_cpu->vec_size();
  else
    vec_end = (end + 1) / 64; // floor

  size_t begin2 = vec_begin * 64;
  size_t end2   = vec_end * 64;

  bool test1;
  size_t bit_index1;
  uint64_t mask1;

  bool test2;
  size_t bit_index2;
  uint64_t mask2;

  if (begin2 > end2) {
    test2 = false;

    if (begin < begin2) {
      test1       = true;
      bit_index1  = begin / 64;
      size_t diff = begin2 - begin;
      assert(diff < 64);
      mask1 = ((uint64_t)1 << (64 - diff)) - 1;

      // create or mask
      size_t diff2 = end - end2 + 1;
      assert(diff2 < 64);
      mask2 = ~(((uint64_t)1 << diff2) - 1);
      mask1 |= ~mask2;
    } else {
      test1 = false;
    }
  } else {
    if (begin < begin2) {
      test1       = true;
      bit_index1  = begin / 64;
      size_t diff = begin2 - begin;
      assert(diff < 64);
      mask1 = ((uint64_t)1 << (64 - diff)) - 1;
    } else {
      test1 = false;
    }

    if (end >= end2) {
      test2       = true;
      bit_index2  = end / 64;
      size_t diff = end - end2 + 1;
      assert(diff < 64);
      mask2 = ~(((uint64_t)1 << diff) - 1);
    } else {
      test2 = false;
    }
  }

  bitset_reset_range<<<blocks, threads>>>(field->is_updated.gpu_rd_ptr(),
                                          vec_begin, vec_end, test1, bit_index1,
                                          mask1, test2, bit_index2, mask2);
}

template <typename DataType>
void reset_data_field(struct CUDA_Context_Field_Edges<DataType>* field,
                      size_t begin, size_t end, DataType val) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  batch_reset<DataType><<<blocks, threads>>>(
      field->data.gpu_wr_ptr(), (index_type)begin, (index_type)end, val);
}

void get_offsets_from_bitset(index_type bitset_size,
                             unsigned int* __restrict__ offsets,
                             DynamicBitset* __restrict__ bitset,
                             size_t* __restrict__ num_set_bits) {
  cub::CachingDeviceAllocator g_allocator(
      true); // Caching allocator for device memory
  DynamicBitsetIterator flag_iterator(bitset);
  IdentityIterator offset_iterator;
  Shared<size_t> num_set_bits_ptr;
  num_set_bits_ptr.alloc(1);
  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                             offset_iterator, flag_iterator, offsets,
                             num_set_bits_ptr.gpu_wr_ptr(true), bitset_size);
  check_cuda_kernel;
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  // CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                             offset_iterator, flag_iterator, offsets,
                             num_set_bits_ptr.gpu_wr_ptr(true), bitset_size);
  check_cuda_kernel;
  // CUDA_SAFE_CALL(cudaFree(d_temp_storage));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  *num_set_bits = *num_set_bits_ptr.cpu_rd_ptr();
}

template <typename DataType, SharedType sharedType, bool reset>
void batch_get_shared_edge(struct CUDA_Context_Common_Edges* ctx,
                           struct CUDA_Context_Field_Edges<DataType>* field,
                           unsigned from_id, uint8_t* send_buffer,
                           DataType i = 0) {
  struct CUDA_Context_Shared_Edges* shared;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
  } else { // sharedMirror
    shared = &ctx->mirror;
  }
  DeviceOnly<DataType>* shared_data = &field->shared_data;
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  // ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2");
  // timer.start();
  // timer1.start();
  size_t v_size = shared->num_edges[from_id];
  if (reset) {
    batch_get_reset_subset<DataType><<<blocks, threads>>>(
        v_size, shared->edges[from_id].device_ptr(), shared_data->device_ptr(),
        field->data.gpu_wr_ptr(), i);
  } else {
    batch_get_subset<DataType><<<blocks, threads>>>(
        v_size, shared->edges[from_id].device_ptr(), shared_data->device_ptr(),
        field->data.gpu_rd_ptr());
  }
  check_cuda_kernel;
  // timer1.stop();
  // timer2.start();
  DataCommMode data_mode = onlyData;
  memcpy(send_buffer, &data_mode, sizeof(data_mode));
  memcpy(send_buffer + sizeof(data_mode), &v_size, sizeof(v_size));
  shared_data->copy_to_cpu(
      (DataType*)(send_buffer + sizeof(data_mode) + sizeof(v_size)), v_size);
  // timer2.stop();
  // timer.stop();
  // fprintf(stderr, "Get %u->%u: Time (ms): %llu + %llu = %llu\n",
  //  ctx->id, from_id,
  //  timer1.duration_ms(), timer2.duration_ms(),
  //  timer.duration_ms());
}

template <typename DataType>
void serializeMessage(struct CUDA_Context_Common_Edges* ctx,
                      DataCommMode data_mode, size_t bit_set_count,
                      size_t num_shared, DeviceOnly<DataType>* shared_data,
                      uint8_t* send_buffer) {
  if (data_mode == noData) {
    // do nothing
    return;
  }

  size_t offset = 0;

  // serialize data_mode
  memcpy(send_buffer, &data_mode, sizeof(data_mode));
  offset += sizeof(data_mode);

  if (data_mode != onlyData) {
    // serialize bit_set_count
    memcpy(send_buffer + offset, &bit_set_count, sizeof(bit_set_count));
    offset += sizeof(bit_set_count);
  }

  if ((data_mode == gidsData) || (data_mode == offsetsData)) {
    // serialize offsets vector
    memcpy(send_buffer + offset, &bit_set_count, sizeof(bit_set_count));
    offset += sizeof(bit_set_count);
    ctx->offsets.copy_to_cpu((unsigned int*)(send_buffer + offset),
                             bit_set_count);
    offset += bit_set_count * sizeof(unsigned int);
  } else if ((data_mode == bitsetData)) {
    // serialize bitset
    memcpy(send_buffer + offset, &num_shared, sizeof(num_shared));
    offset += sizeof(num_shared);
    size_t vec_size = ctx->is_updated.cpu_rd_ptr()->vec_size();
    memcpy(send_buffer + offset, &vec_size, sizeof(vec_size));
    offset += sizeof(vec_size);
    ctx->is_updated.cpu_rd_ptr()->copy_to_cpu(
        (uint64_t*)(send_buffer + offset));
    offset += vec_size * sizeof(uint64_t);
  }

  // serialize data vector
  memcpy(send_buffer + offset, &bit_set_count, sizeof(bit_set_count));
  offset += sizeof(bit_set_count);
  shared_data->copy_to_cpu((DataType*)(send_buffer + offset), bit_set_count);
  // offset += bit_set_count * sizeof(DataType);
}

template <typename DataType, SharedType sharedType, bool reset>
void batch_get_shared_edge(struct CUDA_Context_Common_Edges* ctx,
                           struct CUDA_Context_Field_Edges<DataType>* field,
                           unsigned from_id, uint8_t* send_buffer,
                           size_t* v_size, DataCommMode* data_mode,
                           DataType i = 0) {
  struct CUDA_Context_Shared_Edges* shared;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
  } else { // sharedMirror
    shared = &ctx->mirror;
  }
  DeviceOnly<DataType>* shared_data = &field->shared_data;
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  // ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2"),
  // timer3("timer3"), timer4("timer 4"); timer.start();
  if (enforce_data_mode != onlyData) {
    // timer1.start();
    ctx->is_updated.cpu_rd_ptr()->resize(shared->num_edges[from_id]);
    ctx->is_updated.cpu_rd_ptr()->reset();
    batch_get_subset_bitset<<<blocks, threads>>>(
        shared->num_edges[from_id], shared->edges[from_id].device_ptr(),
        ctx->is_updated.gpu_rd_ptr(), field->is_updated.gpu_rd_ptr());
    check_cuda_kernel;
    // timer1.stop();
    // timer2.start();
    get_offsets_from_bitset(shared->num_edges[from_id],
                            ctx->offsets.device_ptr(),
                            ctx->is_updated.gpu_rd_ptr(), v_size);
    // timer2.stop();
  }
  *data_mode = get_data_mode<DataType>(*v_size, shared->num_edges[from_id]);
  // timer3.start();
  if ((*data_mode) == onlyData) {
    *v_size = shared->num_edges[from_id];
    if (reset) {
      batch_get_reset_subset<DataType><<<blocks, threads>>>(
          *v_size, shared->edges[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(), i);
    } else {
      batch_get_subset<DataType><<<blocks, threads>>>(
          *v_size, shared->edges[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_rd_ptr());
    }
  } else { // bitsetData || offsetsData
    if (reset) {
      batch_get_reset_subset<DataType><<<blocks, threads>>>(
          *v_size, shared->edges[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), i);
    } else {
      batch_get_subset<DataType><<<blocks, threads>>>(
          *v_size, shared->edges[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_rd_ptr());
    }
  }
  check_cuda_kernel;
  // timer3.stop();
  // timer4.start();
  serializeMessage(ctx, *data_mode, *v_size, shared->num_edges[from_id],
                   shared_data, send_buffer);
  // timer4.stop();
  // timer.stop();
  // fprintf(stderr, "Get %u->%u: %d mode %u bitset %u indices. Time (ms): %llu
  // + %llu + %llu + %llu = %llu\n",
  //  ctx->id, from_id, *data_mode,
  //  ctx->is_updated.cpu_rd_ptr()->alloc_size(), sizeof(unsigned int) *
  //  (*v_size), timer1.duration_ms(), timer2.duration_ms(),
  //  timer3.duration_ms(), timer4.duration_ms(), timer.duration_ms());
}

template <typename DataType>
void deserializeMessage(struct CUDA_Context_Common_Edges* ctx,
                        DataCommMode data_mode, size_t& bit_set_count,
                        size_t num_shared, DeviceOnly<DataType>* shared_data,
                        uint8_t* recv_buffer) {
  size_t offset = 0; // data_mode is already deserialized

  if (data_mode != onlyData) {
    // deserialize bit_set_count
    memcpy(&bit_set_count, recv_buffer + offset, sizeof(bit_set_count));
    offset += sizeof(bit_set_count);
  } else {
    bit_set_count = num_shared;
  }

  assert(data_mode != gidsData); // not supported for deserialization on GPUs
  if (data_mode == offsetsData) {
    // deserialize offsets vector
    offset += sizeof(bit_set_count);
    ctx->offsets.copy_to_gpu((unsigned int*)(recv_buffer + offset),
                             bit_set_count);
    offset += bit_set_count * sizeof(unsigned int);
  } else if ((data_mode == bitsetData)) {
    // deserialize bitset
    ctx->is_updated.cpu_rd_ptr()->resize(num_shared);
    offset += sizeof(num_shared);
    size_t vec_size = ctx->is_updated.cpu_rd_ptr()->vec_size();
    offset += sizeof(vec_size);
    ctx->is_updated.cpu_rd_ptr()->copy_to_gpu(
        (uint64_t*)(recv_buffer + offset));
    offset += vec_size * sizeof(uint64_t);
    // get offsets
    size_t v_size;
    get_offsets_from_bitset(num_shared, ctx->offsets.device_ptr(),
                            ctx->is_updated.gpu_rd_ptr(), &v_size);

    assert(bit_set_count == v_size);
  }

  // deserialize data vector
  offset += sizeof(bit_set_count);
  shared_data->copy_to_gpu((DataType*)(recv_buffer + offset), bit_set_count);
  // offset += bit_set_count * sizeof(DataType);
}

template <typename DataType, SharedType sharedType, UpdateOp op>
void batch_set_shared_edge(struct CUDA_Context_Common_Edges* ctx,
                           struct CUDA_Context_Field_Edges<DataType>* field,
                           unsigned from_id, uint8_t* recv_buffer,
                           DataCommMode data_mode) {
  assert(data_mode != noData);
  struct CUDA_Context_Shared_Edges* shared;
  if (sharedType == sharedMaster) {
    shared = &ctx->master;
  } else { // sharedMirror
    shared = &ctx->mirror;
  }
  DeviceOnly<DataType>* shared_data = &field->shared_data;
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);
  size_t v_size;

  // ggc::Timer timer("timer"), timer1("timer1"), timer2("timer2");
  // timer.start();
  // timer1.start();
  deserializeMessage(ctx, data_mode, v_size, shared->num_edges[from_id],
                     shared_data, recv_buffer);
  // timer1.stop();
  // timer2.start();
  if (data_mode == onlyData) {
    if (op == setOp) {
      batch_set_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->edges[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(),
          field->is_updated.gpu_wr_ptr());
    } else if (op == addOp) {
      batch_add_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->edges[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(),
          field->is_updated.gpu_wr_ptr());
    } else if (op == minOp) {
      batch_min_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->edges[from_id].device_ptr(),
          shared_data->device_ptr(), field->data.gpu_wr_ptr(),
          field->is_updated.gpu_wr_ptr());
    }
  } else if (data_mode == gidsData) {
    if (op == setOp) {
      batch_set_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == addOp) {
      batch_add_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == minOp) {
      batch_min_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    }
  } else { // bitsetData || offsetsData
    if (op == setOp) {
      batch_set_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->edges[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == addOp) {
      batch_add_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->edges[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    } else if (op == minOp) {
      batch_min_subset<DataType, sharedType><<<blocks, threads>>>(
          v_size, shared->edges[from_id].device_ptr(),
          ctx->offsets.device_ptr(), shared_data->device_ptr(),
          field->data.gpu_wr_ptr(), field->is_updated.gpu_wr_ptr());
    }
  }
  check_cuda_kernel;
  // timer2.stop();
  // timer.stop();
  // fprintf(stderr, "Set %u<-%u: %d mode Time (ms): %llu + %llu = %llu\n",
  //  ctx->id, from_id, data_mode,
  //  timer1.duration_ms(), timer2.duration_ms(),
  //  timer.duration_ms());
}
#endif
