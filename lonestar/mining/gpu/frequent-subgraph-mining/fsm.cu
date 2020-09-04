// Copyright (c) 2019, Xuhao Chen
#include "fsm.h"
#include "pangolin/timer.h"
#include "pangolin/cutils.h"
#define USE_PID
#define USE_DOMAIN
#define EDGE_INDUCED
#define ENABLE_LABEL
#include <cub/cub.cuh>
#include "pangolin/miner.cuh"
#include "pangolin/bitsets.h"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#define MAX_NUM_PATTERNS 21251

struct OrderedEdge {
  IndexT src;
  IndexT dst;
};

inline __device__ int get_init_pattern_id(node_data_type src_label,
                                          node_data_type dst_label,
                                          int nlabels) {
  return (int)src_label * nlabels + (int)dst_label;
}

inline __device__ unsigned get_pattern_id(node_data_type label0,
                                          node_data_type label1,
                                          node_data_type label2, int nlabels) {
  return nlabels * (nlabels * label0 + label1) + label2;
}

inline __device__ bool is_quick_automorphism(unsigned size, IndexT* vids,
                                             history_type his2,
                                             history_type his, IndexT src,
                                             IndexT dst) {
  if (dst <= vids[0])
    return true;
  if (dst == vids[1])
    return true;
  if (his == 0 && dst < vids[1])
    return true;
  if (size == 2) {
  } else if (size == 3) {
    if (his == 0 && his2 == 0 && dst <= vids[2])
      return true;
    if (his == 0 && his2 == 1 && dst == vids[2])
      return true;
    if (his == 1 && his2 == 1 && dst <= vids[2])
      return true;
  } else {
  }
  return false;
}

inline __device__ void swap(IndexT first, IndexT second) {
  if (first > second) {
    IndexT tmp = first;
    first      = second;
    second     = tmp;
  }
}

inline __device__ int compare(OrderedEdge oneEdge, OrderedEdge otherEdge) {
  swap(oneEdge.src, oneEdge.dst);
  swap(otherEdge.src, otherEdge.dst);
  if (oneEdge.src == otherEdge.src)
    return oneEdge.dst - otherEdge.dst;
  else
    return oneEdge.src - otherEdge.src;
}

inline __device__ bool is_edge_automorphism(unsigned size, IndexT* vids,
                                            history_type* hiss,
                                            history_type his, IndexT src,
                                            IndexT dst) {
  if (size < 3)
    return is_quick_automorphism(size, vids, hiss[2], his, src, dst);
  if (dst <= vids[0])
    return true;
  if (his == 0 && dst <= vids[1])
    return true;
  if (dst == vids[hiss[his]])
    return true;
  OrderedEdge added_edge;
  added_edge.src = src;
  added_edge.dst = dst;
  for (unsigned index = his + 1; index < size; ++index) {
    OrderedEdge edge;
    edge.src = vids[hiss[index]];
    edge.dst = vids[index];
    int cmp  = compare(added_edge, edge);
    if (cmp <= 0)
      return true;
  }
  return false;
}

__global__ void extend_alloc(unsigned m, unsigned level, CSRGraph graph,
                             EmbeddingList emb_list, IndexT* num_new_emb) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ IndexT vid[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  __shared__ history_type his[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  if (pos < m) {
    emb_list.get_edge_embedding(level, pos, vid[tid], his[tid]);
    num_new_emb[pos] = 0;
    // if (pos == 1) printf("src=%d, dst=%d\n", vid[tid][0], vid[tid][1]);
    for (unsigned i = 0; i < level + 1; ++i) {
      IndexT src       = vid[tid][i];
      IndexT row_begin = graph.edge_begin(src);
      IndexT row_end   = graph.edge_end(src);
      for (IndexT e = row_begin; e < row_end; e++) {
        IndexT dst = graph.getEdgeDst(e);
        if (!is_edge_automorphism(level + 1, vid[tid], his[tid], i, src, dst))
          num_new_emb[pos]++;
      }
    }
  }
}

__global__ void extend_insert(unsigned m, unsigned level, CSRGraph graph,
                              EmbeddingList emb_list, IndexT* indices) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ IndexT vids[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  __shared__ history_type his[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  if (pos < m) {
    emb_list.get_edge_embedding(level, pos, vids[tid], his[tid]);
    IndexT start = indices[pos];
    for (unsigned i = 0; i < level + 1; ++i) {
      IndexT src       = vids[tid][i];
      IndexT row_begin = graph.edge_begin(src);
      IndexT row_end   = graph.edge_end(src);
      for (IndexT e = row_begin; e < row_end; e++) {
        IndexT dst = graph.getEdgeDst(e);
        if (!is_edge_automorphism(level + 1, vids[tid], his[tid], i, src,
                                  dst)) {
          emb_list.set_idx(level + 1, start, pos);
          emb_list.set_his(level + 1, start, i);
          emb_list.set_vid(level + 1, start++, dst);
        }
      }
    }
  }
}

__global__ void init_aggregate(unsigned m, unsigned num_emb, CSRGraph graph,
                               EmbeddingList emb_list, unsigned* pids,
                               int nlabels, unsigned threshold,
                               Bitsets small_sets, Bitsets large_sets) {
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < num_emb) {
    IndexT src               = emb_list.get_idx(1, pos);
    IndexT dst               = emb_list.get_vid(1, pos);
    node_data_type src_label = graph.getData(src);
    node_data_type dst_label = graph.getData(dst);
    int pid                  = 0;
    if (src_label <= dst_label)
      pid = get_init_pattern_id(src_label, dst_label, nlabels);
    else
      pid = get_init_pattern_id(dst_label, src_label, nlabels);
    pids[pos] = pid;
    if (src_label < dst_label) {
      small_sets.set(pid, src);
      large_sets.set(pid, dst);
    } else if (src_label > dst_label) {
      small_sets.set(pid, dst);
      large_sets.set(pid, src);
    } else {
      small_sets.set(pid, src);
      small_sets.set(pid, dst);
      large_sets.set(pid, src);
      large_sets.set(pid, dst);
    }
  }
}

__global__ void count_ones(int id, Bitsets sets, int* count) {
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = 0;
  if (pos < sets.vec_size())
    num = sets.count_num_ones(id, pos);
  int block_total = BlockReduce(temp_storage).Sum(num);
  if (threadIdx.x == 0)
    atomicAdd(count, block_total);
}

int init_support_count(unsigned m, int npatterns, unsigned threshold,
                       Bitsets small_sets, Bitsets large_sets,
                       bool* init_support_map) {
  int num_freq_patterns = 0;
  for (int i = 0; i < npatterns; i++) {
    int a, b, *d_count;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_count, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
    count_ones<<<(m - 1) / 256 + 1, 256>>>(i, small_sets, d_count);
    CudaTest("solving count_ones `failed");
    CUDA_SAFE_CALL(
        cudaMemcpy(&a, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
    count_ones<<<(m - 1) / 256 + 1, 256>>>(i, large_sets, d_count);
    CUDA_SAFE_CALL(
        cudaMemcpy(&b, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned support = a < b ? a : b;
    if (support >= threshold) {
      init_support_map[i] = 1;
      num_freq_patterns++;
    } else
      init_support_map[i] = 0;
  }
  return num_freq_patterns;
}

int support_count(unsigned m, unsigned npatterns, unsigned threshold,
                  Bitsets small_sets, Bitsets middle_sets, Bitsets large_sets,
                  bool* support_map) {
  int num_freq_patterns = 0;
  for (int i = 0; i < npatterns; i++) {
    int a, b, c, *d_count;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_count, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
    count_ones<<<(m - 1) / 256 + 1, 256>>>(i, small_sets, d_count);
    CUDA_SAFE_CALL(
        cudaMemcpy(&a, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
    count_ones<<<(m - 1) / 256 + 1, 256>>>(i, large_sets, d_count);
    CUDA_SAFE_CALL(
        cudaMemcpy(&b, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));
    count_ones<<<(m - 1) / 256 + 1, 256>>>(i, middle_sets, d_count);
    CUDA_SAFE_CALL(
        cudaMemcpy(&c, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned small   = a < b ? a : b;
    unsigned support = small < c ? small : c;
    if (support >= threshold) {
      support_map[i] = 1;
      num_freq_patterns++;
    } else
      support_map[i] = 0;
  }
  return num_freq_patterns;
}

__global__ void init_filter_check(unsigned m, unsigned* pids,
                                  bool* init_support_map,
                                  IndexT* is_frequent_emb) {
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < m) {
    unsigned pid     = pids[pos];
    bool is_frequent = init_support_map[pid];
    if (is_frequent)
      is_frequent_emb[pos] = 1;
  }
}

__global__ void copy_vids(unsigned m, EmbeddingList emb_list, IndexT* vid_list0,
                          IndexT* vid_list1) {
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < m) {
    vid_list0[pos] = emb_list.get_idx(1, pos);
    vid_list1[pos] = emb_list.get_vid(1, pos);
  }
}

__global__ void init_filter(unsigned m, EmbeddingList emb_list,
                            IndexT* vid_list0, IndexT* vid_list1,
                            IndexT* indices, IndexT* is_frequent_emb) {
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < m) {
    if (is_frequent_emb[pos]) {
      IndexT src     = vid_list0[pos];
      IndexT dst     = vid_list1[pos];
      unsigned start = indices[pos];
      emb_list.set_vid(1, start, dst);
      emb_list.set_idx(1, start, src);
    }
  }
}

__global__ void aggregate_check(unsigned num_emb, unsigned level,
                                CSRGraph graph, EmbeddingList emb_list,
                                unsigned* pids, int nlabels, unsigned threshold,
                                unsigned* ne) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ IndexT vids[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  __shared__ history_type his[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  if (pos < num_emb) {
    emb_list.get_edge_embedding(level, pos, vids[tid], his[tid]);
    unsigned n = level + 1;
    assert(n < 4);
    IndexT first      = vids[tid][0];
    IndexT second     = vids[tid][1];
    IndexT third      = vids[tid][2];
    node_data_type l0 = graph.getData(first);
    node_data_type l1 = graph.getData(second);
    node_data_type l2 = graph.getData(third);
    history_type h2   = his[tid][2];
    unsigned pid      = 0;
    if (n == 3) {
      if (h2 == 0) {
        if (l1 < l2) {
          pid = get_pattern_id(l0, l2, l1, nlabels);
        } else {
          pid = get_pattern_id(l0, l1, l2, nlabels);
        }
      } else {
        assert(h2 == 1);
        if (l0 < l2) {
          pid = get_pattern_id(l1, l2, l0, nlabels);
        } else {
          pid = get_pattern_id(l1, l0, l2, nlabels);
        }
      }
    } else {
    }
    pids[pos] = pid;
    atomicAdd(&ne[pid], 1);
  }
}

__global__ void find_candidate_patterns(unsigned num_patterns, unsigned* ne,
                                        unsigned minsup, unsigned* id_map,
                                        unsigned* num_new_patterns) {
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < num_patterns) {
    if (ne[pos] >= minsup) {
      unsigned new_id = atomicAdd(num_new_patterns, 1);
      id_map[pos]     = new_id;
    }
  }
}

__global__ void aggregate(unsigned m, unsigned num_emb, unsigned level,
                          CSRGraph graph, EmbeddingList emb_list,
                          unsigned* pids, unsigned* ne, unsigned* id_map,
                          int nlabels, unsigned threshold, Bitsets small_sets,
                          Bitsets middle_sets, Bitsets large_sets) {
  unsigned tid = threadIdx.x;
  unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ IndexT vids[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  __shared__ history_type his[BLOCK_SIZE][PANGOLIN_MAX_SIZE];
  if (pos < num_emb) {
    emb_list.get_edge_embedding(level, pos, vids[tid], his[tid]);
    assert(level == 2);
    IndexT first      = vids[tid][0];
    IndexT second     = vids[tid][1];
    IndexT third      = vids[tid][2];
    node_data_type l0 = graph.getData(first);
    node_data_type l1 = graph.getData(second);
    node_data_type l2 = graph.getData(third);
    history_type h2   = his[tid][2];
    IndexT small, middle, large;
    unsigned pid = pids[pos];
    if (ne[pid] >= threshold) {
      pid = id_map[pid];
      if (h2 == 0) {
        middle = first;
        if (l1 < l2) {
          small = second;
          large = third;
        } else {
          small = third;
          large = second;
        }
        small_sets.set(pid, small);
        middle_sets.set(pid, middle);
        large_sets.set(pid, large);
        if (l1 == l2) {
          small_sets.set(pid, large);
          large_sets.set(pid, small);
        }
      } else {
        assert(h2 == 1);
        middle = second;
        if (l0 < l2) {
          small = first;
          large = third;
        } else {
          small = third;
          large = first;
        }
        small_sets.set(pid, small);
        middle_sets.set(pid, middle);
        large_sets.set(pid, large);
        if (l0 == l2) {
          small_sets.set(pid, large);
          large_sets.set(pid, small);
        }
      }
    }
  }
}

void parallel_prefix_sum(int n, IndexT* in, IndexT* out) {
  IndexT total = 0;
  for (size_t i = 0; i < n; i++) {
    out[i] = total;
    total += in[i];
  }
  out[n] = total;
}

void fsm_gpu_solver(std::string fname, unsigned k, unsigned minsup,
                    AccType& total_num) {
  CSRGraph graph_cpu, graph_gpu;
  int nlabels = graph_cpu.read(fname); // read graph into CPU memoryA
  int m       = graph_cpu.get_nnodes();
  int nnz     = graph_cpu.get_nedges();
  graph_cpu.copy_to_gpu(graph_gpu); // copy graph to GPU memory
  EmbeddingList emb_list;
  emb_list.init(nnz, k + 1, false);
  emb_list.init_cpu(&graph_cpu);

  int nthreads          = BLOCK_SIZE;
  int nblocks           = DIVIDE_INTO(nnz, nthreads);
  int num_init_patterns = (nlabels + 1) * (nlabels + 1);
  std::cout << "Number of init patterns: " << num_init_patterns << std::endl;
  unsigned num_emb = emb_list.size();
  std::cout << "number of single-edge embeddings: " << num_emb << "\n";
  unsigned* pids;
  CUDA_SAFE_CALL(cudaMalloc((void**)&pids, sizeof(unsigned) * num_emb));
  bool* h_init_support_map = (bool*)malloc(sizeof(bool) * num_init_patterns);
  bool* d_init_support_map;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_init_support_map,
                            sizeof(bool) * num_init_patterns));
  IndexT* is_frequent_emb;
  CUDA_SAFE_CALL(
      cudaMalloc((void**)&is_frequent_emb, sizeof(IndexT) * (num_emb + 1)));
  CUDA_SAFE_CALL(
      cudaMemset(is_frequent_emb, 0, sizeof(IndexT) * (num_emb + 1)));
  IndexT *vid_list0, *vid_list1;
  CUDA_SAFE_CALL(cudaMalloc((void**)&vid_list0, sizeof(IndexT) * num_emb));
  CUDA_SAFE_CALL(cudaMalloc((void**)&vid_list1, sizeof(IndexT) * num_emb));
  Bitsets small_sets, large_sets, middle_sets;
  small_sets.alloc(MAX_NUM_PATTERNS, m);
  large_sets.alloc(MAX_NUM_PATTERNS, m);
  middle_sets.alloc(MAX_NUM_PATTERNS, m);
  small_sets.set_size(num_init_patterns, m);
  large_sets.set_size(num_init_patterns, m);
  middle_sets.set_size(num_init_patterns, m);

  IndexT *num_new_emb, *indices;
  CUDA_SAFE_CALL(cudaMalloc((void**)&indices, sizeof(IndexT) * (num_emb + 1)));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  nblocks = (num_emb - 1) / nthreads + 1;
  unsigned* d_num_new_patterns;
  unsigned h_num_new_patterns = 0;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_num_new_patterns, sizeof(unsigned)));
  printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks,
         nthreads);

  Timer t;
  t.Start();
  unsigned level = 1;
  init_aggregate<<<nblocks, nthreads>>>(m, num_emb, graph_gpu, emb_list, pids,
                                        nlabels, minsup, small_sets,
                                        large_sets);
  CudaTest("solving init_aggregate `failed");
  std::cout << "Init_aggregate Done\n";
  int num_freq_patterns = init_support_count(
      m, num_init_patterns, minsup, small_sets, large_sets, h_init_support_map);
  total_num += num_freq_patterns;
  if (num_freq_patterns == 0) {
    std::cout << "No frequent pattern found\n\n";
    return;
  }
  std::cout << "Number of frequent single-edge patterns: " << num_freq_patterns
            << "\n";
  CUDA_SAFE_CALL(cudaMemcpy(d_init_support_map, h_init_support_map,
                            sizeof(bool) * num_init_patterns,
                            cudaMemcpyHostToDevice));
  init_filter_check<<<nblocks, nthreads>>>(num_emb, pids, d_init_support_map,
                                           is_frequent_emb);
  CudaTest("solving init_filter_check `failed");
  thrust::exclusive_scan(thrust::device, is_frequent_emb,
                         is_frequent_emb + num_emb + 1, indices);
  IndexT new_size;
  CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[num_emb], sizeof(IndexT),
                            cudaMemcpyDeviceToHost));
  std::cout << "number of embeddings after pruning: " << new_size << "\n";
  copy_vids<<<nblocks, nthreads>>>(num_emb, emb_list, vid_list0, vid_list1);
  CudaTest("solving copy_vids `failed");
  init_filter<<<nblocks, nthreads>>>(num_emb, emb_list, vid_list0, vid_list1,
                                     indices, is_frequent_emb);
  CudaTest("solving init_filter `failed");
  CUDA_SAFE_CALL(cudaFree(indices));
  CUDA_SAFE_CALL(cudaFree(is_frequent_emb));
  CUDA_SAFE_CALL(cudaFree(pids));
  // small_sets.clean();
  // large_sets.clean();
  small_sets.clear();
  large_sets.clear();
  CUDA_SAFE_CALL(cudaFree(vid_list0));
  CUDA_SAFE_CALL(cudaFree(vid_list1));
  CUDA_SAFE_CALL(cudaFree(d_init_support_map));
  emb_list.remove_tail(new_size);

  while (1) {
    num_emb = emb_list.size();
    std::cout << "number of embeddings in level " << level << ": " << num_emb
              << "\n";
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&num_new_emb, sizeof(IndexT) * (num_emb + 1)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&indices, sizeof(IndexT) * (num_emb + 1)));
    std::cout << "Done allocating memory for embeddings in level " << level
              << "\n";
    nblocks = (num_emb - 1) / nthreads + 1;
    extend_alloc<<<nblocks, nthreads>>>(num_emb, level, graph_gpu, emb_list,
                                        num_new_emb);
    CudaTest("solving extend_alloc failed");
    thrust::exclusive_scan(thrust::device, num_new_emb,
                           num_new_emb + num_emb + 1, indices);
    CudaTest("Scan failed");
    CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[num_emb], sizeof(IndexT),
                              cudaMemcpyDeviceToHost));
    std::cout << "number of new embeddings: " << new_size << "\n";
    emb_list.add_level(new_size);
    extend_insert<<<nblocks, nthreads>>>(num_emb, level, graph_gpu, emb_list,
                                         indices);
    CudaTest("solving extend_insert failed");
    std::cout << "Extend_insert Done\n";
    num_emb = emb_list.size();
    CUDA_SAFE_CALL(cudaFree(num_new_emb));
    CUDA_SAFE_CALL(cudaFree(indices));
    level++;

    int num_patterns = nlabels * num_init_patterns;
    nblocks          = (num_emb - 1) / nthreads + 1;
    std::cout << "Number of patterns in level " << level << ": " << num_patterns
              << std::endl;
    std::cout << "number of embeddings in level " << level << ": " << num_emb
              << "\n";
    unsigned *ne, *id_map;
    CUDA_SAFE_CALL(cudaMalloc((void**)&ne, sizeof(unsigned) * num_patterns));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&id_map, sizeof(unsigned) * num_patterns));
    CUDA_SAFE_CALL(cudaMemset(ne, 0, sizeof(unsigned) * num_patterns));
    CUDA_SAFE_CALL(cudaMalloc((void**)&pids, sizeof(unsigned) * num_emb));
    std::cout << "Done allocating memory for aggregation in level " << level
              << "\n";
    aggregate_check<<<nblocks, nthreads>>>(num_emb, level, graph_gpu, emb_list,
                                           pids, nlabels, minsup, ne);
    CudaTest("solving aggregate_check failed");
    CUDA_SAFE_CALL(cudaMemset(d_num_new_patterns, 0, sizeof(unsigned)));
    find_candidate_patterns<<<(num_patterns - 1) / nthreads + 1, nthreads>>>(
        num_patterns, ne, minsup, id_map, d_num_new_patterns);
    CudaTest("solving find_candidate_patterns failed");
    CUDA_SAFE_CALL(cudaMemcpy(&h_num_new_patterns, d_num_new_patterns,
                              sizeof(unsigned), cudaMemcpyDeviceToHost));
    std::cout << "Number of candidate patterns in level " << level << ": "
              << h_num_new_patterns << std::endl;

    // small_sets.alloc(h_num_new_patterns, m);
    // large_sets.alloc(h_num_new_patterns, m);
    // middle_sets.alloc(h_num_new_patterns, m);
    small_sets.set_size(h_num_new_patterns, m);
    large_sets.set_size(h_num_new_patterns, m);
    middle_sets.set_size(h_num_new_patterns, m);
    std::cout << "Done allocating sets\n";
    aggregate<<<nblocks, nthreads>>>(m, num_emb, level, graph_gpu, emb_list,
                                     pids, ne, id_map, nlabels, minsup,
                                     small_sets, middle_sets, large_sets);
    CudaTest("solving aggregate failed");
    bool* h_support_map = (bool*)malloc(sizeof(bool) * h_num_new_patterns);
    num_freq_patterns = support_count(m, h_num_new_patterns, minsup, small_sets,
                                      middle_sets, large_sets, h_support_map);
    CudaTest("solving support_count failed");
    CUDA_SAFE_CALL(cudaFree(ne));
    CUDA_SAFE_CALL(cudaFree(id_map));
    std::cout << "num_frequent_patterns: " << num_freq_patterns << "\n";
    total_num += num_freq_patterns;
    if (num_freq_patterns == 0)
      break;
    if (level == k)
      break;
    // filter<<<nblocks, nthreads>>>(level, emb_list);
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  printf("\truntime = %f ms.\n", t.Millisecs());
}
