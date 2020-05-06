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

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <process.h>
#include <time.h>
#include <Psapi.h>
#else
#include <sys/time.h>
#endif
#include <cuda.h>
#include "SGDCommonCu.h"
#include "SGDGraphCu.h"
#include <algorithm>

#define R 2
#define C 2
#define BLOCKSIZE 1
#define GRANULARITY 5000
#define _P_DATA1_(_t25) _P_DATA1[_t25]
#define _P_DATA1__(_t25) _P_DATA1[_t25 + 1]
#define new_ratings(_t49, _t50, _t51, _t52)                                    \
  new_ratings[(_t50)*R * C + (_t51)*C + (_t52)]
struct a_list {
  int col_[1];
  float ratings[R * C];
  struct a_list* next;
};

struct mk {
  struct a_list* ptr;
};

#ifndef GALOISGPU_APPS_SGD_CUDA_SGDASYNCEDGECU_H_
#define GALOISGPU_APPS_SGD_CUDA_SGDASYNCEDGECU_H_
#define col_(_t5) col[_t5]
#define index_(i) index[i]
#define index__(i) index[i + 1]
#define __rose_lt(x, y) ((x) < (y) ? (x) : (y))

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
bool out_degree_compare(std::pair<int, int> i, std::pair<int, int> j) {
  return (i.second > j.second);
}

struct data_list {
  float data[R][C];
  int col;
  struct data_list* next;
};
__device__ void segreduce_warp2(float* y, float* val) {
  int tx     = threadIdx.x;
  float left = 0;

  if (tx >= 1) {
    left = val[tx - 1];
    val[tx] += left;
    left = 0;
  }
  __syncthreads();

  if (tx >= 2) {
    left = val[tx - 2];
    val[tx] += left;
    left = 0;
  }
  __syncthreads();
  if (tx >= 4) {
    left = val[tx - 4];
    val[tx] += left;
    left = 0;
  }
  __syncthreads();
  if (tx >= 8) {
    left = val[tx - 8];
    val[tx] += left;
    left = 0;
  }
  __syncthreads();

  if (tx == SGD_FEATURE_SIZE - 1)
    *y += val[tx];
  __syncthreads();
}

__global__ void sgd_blk_diag_operator(float* fv, int* metadata,
                                      float* new_ratings, int* _P_DATA2,
                                      int* _P_DATA1, float step_size, int t2) {
  int l;
  int i;
  int bx;
  bx = blockIdx.x;
  int tx;
  tx = threadIdx.x;
  int ty;
  ty = threadIdx.y;
  __device__ __shared__ float _P2[BLOCKSIZE];
  __device__ __shared__ float _P3[BLOCKSIZE * SGD_FEATURE_SIZE];
  int newVariable5;
  float _P4[C];
  float _P5[R];
  // int t4;
  // int t6;
  // int t8;
  int t10;
  int t12;
  int movie_size = metadata[2];
  if (ty <= _P_DATA1__(t2) - _P_DATA1_(t2) - BLOCKSIZE * bx - 1)
    newVariable5 = _P_DATA2[_P_DATA1_(t2) + BLOCKSIZE * bx + ty];
  if (ty <= _P_DATA1__(t2) - _P_DATA1_(t2) - BLOCKSIZE * bx - 1) {
    for (t10 = 0; t10 <= R - 1; t10 += 1)
      _P5[R * newVariable5 + t10 - R * newVariable5] =
          fv[(R * newVariable5 + t10) * SGD_FEATURE_SIZE + tx];
    for (i = 0; i <= R - 1; i += 1) {
      for (t12 = 0; t12 <= C - 1; t12 += 1)
        _P4[C * newVariable5 + C * t2 + t12 + 1 -
            (C * t2 + C * newVariable5 + 1)] =
            fv[(C * newVariable5 + C * t2 + t12 + 1) * SGD_FEATURE_SIZE + tx];
      for (l = 0; l <= C - 1; l += 1) {
        if (0 <= new_ratings(t2, BLOCKSIZE * bx + _P_DATA1_(t2) + ty, i, l)) {
          _P2[ty] = -new_ratings[(BLOCKSIZE * bx + _P_DATA1_(t2) + ty) * R * C +
                                 i * C + l];
          _P3[tx + ty * SGD_FEATURE_SIZE] =
              (_P5[R * newVariable5 + i - R * newVariable5] *
               _P4[C * t2 - movie_size + 1 + C * newVariable5 + l + movie_size -
                   (C * t2 + C * newVariable5 + 1)]);
          segreduce_warp2(&_P2[ty], &_P3[0 + ty * SGD_FEATURE_SIZE]);
          _P5[R * newVariable5 + i - R * newVariable5] -=
              (step_size *
               ((_P2[ty] * _P4[C * t2 - movie_size + 1 + C * newVariable5 + l +
                               movie_size - (C * t2 + C * newVariable5 + 1)]) +
                (0.05f * _P5[R * newVariable5 + i - R * newVariable5])));
          _P4[C * t2 - movie_size + 1 + C * newVariable5 + l + movie_size -
              (C * t2 + C * newVariable5 + 1)] -=
              (step_size *
               ((_P2[ty] * _P5[R * newVariable5 + i - R * newVariable5]) +
                (0.05f * _P4[C * t2 - movie_size + 1 + C * newVariable5 + l +
                             movie_size - (C * t2 + C * newVariable5 + 1)])));
        } else if (new_ratings(t2, BLOCKSIZE * bx + _P_DATA1_(t2) + ty, i, l) <=
                   -2) {
          _P2[ty] = -new_ratings[(BLOCKSIZE * bx + _P_DATA1_(t2) + ty) * R * C +
                                 i * C + l];
          _P3[tx + ty * SGD_FEATURE_SIZE] =
              (_P5[R * newVariable5 + i - R * newVariable5] *
               _P4[C * t2 - movie_size + 1 + C * newVariable5 + l + movie_size -
                   (C * t2 + C * newVariable5 + 1)]);
          segreduce_warp2(&_P2[ty], &_P3[0 + ty * SGD_FEATURE_SIZE]);
          _P5[R * newVariable5 + i - R * newVariable5] -=
              (step_size *
               ((_P2[ty] * _P4[C * t2 - movie_size + 1 + C * newVariable5 + l +
                               movie_size - (C * t2 + C * newVariable5 + 1)]) +
                (0.05f * _P5[R * newVariable5 + i - R * newVariable5])));
          _P4[C * t2 - movie_size + 1 + C * newVariable5 + l + movie_size -
              (C * t2 + C * newVariable5 + 1)] -=
              (step_size *
               ((_P2[ty] * _P5[R * newVariable5 + i - R * newVariable5]) +
                (0.05f * _P4[C * t2 - movie_size + 1 + C * newVariable5 + l +
                             movie_size - (C * t2 + C * newVariable5 + 1)])));
        }
      }
      for (t12 = 0; t12 <= C - 1; t12 += 1)
        fv[(C * newVariable5 + C * t2 + t12 + 1) * SGD_FEATURE_SIZE + tx] =
            _P4[C * newVariable5 + C * t2 + t12 + 1 -
                (C * t2 + C * newVariable5 + 1)];
    }
    for (t10 = 0; t10 <= R - 1; t10 += 1)
      fv[(R * newVariable5 + t10) * SGD_FEATURE_SIZE + tx] =
          _P5[R * newVariable5 + t10 - R * newVariable5];
  }
}

struct Timer {
  double _start;
  double _end;
  Timer() : _start(0), _end(0) {}
  void clear() { _start = _end = 0; }
  void start() { _start = rtclock(); }
  void stop() { _end = rtclock(); }
  double get_time_seconds(void) { return (_end - _start); }

#ifdef _WIN32
  static double rtclock() {
    LARGE_INTEGER tickPerSecond, tick;
    QueryPerformanceFrequency(&tickPerSecond);
    QueryPerformanceCounter(&tick);
    return (tick.QuadPart * 1000000 / tickPerSecond.QuadPart) * 1.0e-6;
  }
#else
  static double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0)
      printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
  }
#endif
};

/************************************************************************
 ************************************************************************/
///
struct RunStats {
  int round;
  int curr_step;
  float total_time;
  float time_per_diagonal;
  float insp_time;
  RunStats(int r, int s, float t, float t_p_d, float i_t) {
    round             = r;
    curr_step         = s;
    total_time        = t;
    time_per_diagonal = t_p_d;
    insp_time         = i_t;
  }
  RunStats() {
    round = curr_step = 0;
    total_time = time_per_diagonal = insp_time = 0.0f;
  }
  //   fprintf(stderr, "diag\t%d\t%d\t%6.6g\t%6.6g\t%6.6g\t", round,
  //   curr_step,total_time,total_time/(double)(m+ n -1), insp_time);
};
struct StatAccumulator {
  std::vector<RunStats> stats;
  void push_stats(int r, int s, float t, float t_p_d, float i_t) {
    RunStats rs(r, s, t, t_p_d, i_t);
    stats.push_back(rs);
    //      fprintf(stderr, "diag#\t%d\t%d\t%6.6g\t%6.6g\t%6.6g\t", r,
    //      s,t,t_p_d, i_t);
  }
  void print() {
    RunStats sum;
    for (int i = 0; i < stats.size(); ++i) {
      RunStats& s = stats[i];
      sum.round += s.round;
      sum.curr_step += s.curr_step;
      sum.total_time += s.total_time;
      sum.time_per_diagonal += s.time_per_diagonal;
      sum.insp_time += s.insp_time;
    }
    size_t num_items = stats.size();
    printf("\nAverage time per iteration: %6.6g\n", sum.total_time / num_items);
  }
};
///
template <typename T>
struct CUDAArray {
  T* device_data;
  T* host_data;
  size_t _size;
  CUDAArray(size_t s) : _size(s) {
    host_data   = new T[_size];
    device_data = NULL;
  }
  ~CUDAArray() {
    delete[] host_data;
    if (device_data != NULL)
      gpuErrchk(cudaFree(device_data));
  }
  void copy_to_device() {
    gpuErrchk(cudaMemcpy(device_data, host_data, sizeof(T) * _size,
                         cudaMemcpyHostToDevice));
  }
  void create_on_device() {
    gpuErrchk(cudaMalloc(&device_data, sizeof(T) * _size));
  }
  void copy_to_host() {
    gpuErrchk(cudaMemcpy(host_data, device_data, sizeof(T) * _size,
                         cudaMemcpyDeviceToHost));
  }
  T* host_ptr() { return host_data; }
  T* device_ptr() { return device_data; }
  size_t size() { return _size; }
};

/************************************************************************

 ************************************************************************/

// typedef float EdgeDataType;
typedef unsigned int EdgeDataType;

struct SGDAsynEdgeCudaFunctor {
  typedef SGD_LC_LinearArray_Undirected_Graph<unsigned int, EdgeDataType>
      GraphTy;
  typedef CUDAArray<int> ArrayType;
  typedef CUDAArray<float> FeatureArrayType;
  ////////////////////////////////////////////////////////////
  /************************************************
   *
   *************************************************/
  StatAccumulator stats;
  GraphTy graph;
  std::vector<int> movies;
  std::map<int, int> old_pos_to_new_pos;
  std::vector<std::pair<int, int>>
      sorted_nodes; // 1st field is the position of the node, second field is
                    // the out_degree

  std::vector<int> user_indices;
  ArrayType* metadata;

  ArrayType* index;
  ArrayType* diag_number;
  ArrayType* new_index;
  ArrayType* col;
  ArrayType* new_col;
  FeatureArrayType* new_ratings;

  // to track diagonal arrays
  int count_of_diagonals;
  FeatureArrayType* features;
  FeatureArrayType* ratings;
  float accumulated_error;
  int round;
  unsigned int max_rating;
  char filename[512];
  std::vector<int> user_edge_count;
  /************************************************************************
   *
   *metadata (16)
   *edge_info, worklist, ratings (2+1+1)*NE
   *locks, features*FEATURE_SIZE, (1+FEATURE_SIZE)NN
   ************************************************************************/
  SGDAsynEdgeCudaFunctor(bool road, const char* p_filename) : round(0) {
    strcpy(filename, p_filename);
    // fprintf(stderr, "Creating SGDAsynEdgeCudaFunctor -  features =[%d].\n",
    // SGD_FEATURE_SIZE);
    graph.read(p_filename);
    allocate();
    initialize();
    printf("Feature size: %d\n", SGD_FEATURE_SIZE);
    // printf("Number of movies found: %ld\n", movies.size());
  }
  /************************************************************************
   *
   ************************************************************************/
  SGDAsynEdgeCudaFunctor(int num_m, int num_u) : round(0) {
    strcpy(filename, "generated-input");
    // fprintf(stderr, "Creating SGDAsynEdgeFunctor -  features =[%d] .\n",
    // SGD_FEATURE_SIZE);
    complete_bipartitie(graph, num_m, num_u);
    allocate();
    initialize();
    fprintf(stderr, "Number of movies found :: %ld\n", movies.size());
  }
  /************************************************************************
   *
   ************************************************************************/
  SGDAsynEdgeCudaFunctor(int num_m) : round(0) {
    strcpy(filename, "gen-diagonal-input");
    // fprintf(stderr, "Creating SGDAsynEdgeFunctor -  features =[%d] .\n",
    // SGD_FEATURE_SIZE);
    diagonal_graph(graph, num_m);
    allocate();
    initialize();
    fprintf(stderr, "Number of movies found :: %ld\n", movies.size());
  }
  /************************************************************************
   *
   ************************************************************************/
  void allocate() {
    features = new FeatureArrayType(graph.num_nodes() * SGD_FEATURE_SIZE);
    features->create_on_device();
    ratings  = new FeatureArrayType(graph.num_edges());
    metadata = new ArrayType(16);
    metadata->create_on_device();
    index = new ArrayType(graph.num_nodes() + 1);
    col   = new ArrayType(graph.num_edges());
  }
  /************************************************************************
   *
   ************************************************************************/
  void deallocate() {
    delete features;
    delete ratings;
    delete metadata;
    delete index;
  }
  /************************************************************************
   *
   ************************************************************************/
  void copy_to_device() { features->copy_to_device(); }
  /************************************************************************
   *
   ************************************************************************/
  void copy_to_host() { features->copy_to_host(); }
  /************************************************************************
   *
   ************************************************************************/
  void initialize() {
    {
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);
      int device;
      for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        // fprintf(stderr, "Device %s (%d) : CC %d.%d, MaxThreads:%d \n",
        //		deviceProp.name, device, deviceProp.major,
        //		deviceProp.minor, deviceProp.maxThreadsPerBlock);
      }
    }
    std::vector<int> all_edges;
    initialize_features_random(graph, features, movies);
    movies.clear();
    unsigned int max_degree = 0;
    // unsigned max_degree_id = 0;

    for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      for (int j = 0; j < graph.num_neighbors(i); j++) {
        if (graph.out_neighbors(i, j) >= graph.num_nodes())
          fprintf(stderr, "error in input at %d\n", i);
      }
    }

    for (unsigned int i = 0; i < graph.num_nodes(); ++i) {

      sorted_nodes.push_back(std::pair<int, int>(i, graph.num_neighbors(i)));
      if (graph.num_neighbors(i) > max_degree) {
        max_degree = graph.num_neighbors(i);
        // max_degree_id = i;
      }
      if (graph.num_neighbors(i) > 0) {
        movies.push_back(i);
      } else {
        user_indices.push_back(i);
      }
    }
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), out_degree_compare);
    max_rating = 0;
    for (unsigned int i = 0; i < graph.num_edges(); ++i) {
      max_rating = std::max(max_rating, graph.out_edge_data()[i]);
    }
    // fprintf(stderr, "] , max_Rating: %d, movies: %ld, Max degree:: %d for
    // node: %d\n", 		max_rating, movies.size(), max_degree,
    // max_degree_id);
    distribute_chunks(all_edges);
    cache_chunks(all_edges);
  }
  /************************************************************************
   *
   ************************************************************************/
  void cache_chunks(std::vector<int>& all_edges) {
    index->host_ptr()[0] = 0;
    int count            = 0;
    int user_count       = movies.size();

    for (int i = 0; i < sorted_nodes.size(); i++) {
      for (int j = 0; j < sorted_nodes[i].second; j++) {
        int old_pos = graph.out_neighbors(sorted_nodes[i].first, j);

        if (old_pos_to_new_pos.find(old_pos) != old_pos_to_new_pos.end())
          col->host_ptr()[count] = old_pos_to_new_pos.find(old_pos)->second;
        else {
          col->host_ptr()[count] = user_count;
          old_pos_to_new_pos.insert(std::pair<int, int>(old_pos, user_count));
          user_count++;
        }
        ratings->host_ptr()[count++] =
            graph.out_edge_data(sorted_nodes[i].first, j);
      }
      index->host_ptr()[i + 1] = count;
    }

    graph.outgoing_index()[0] = index->host_ptr()[0];
    for (int i = 0; i < sorted_nodes.size(); i++) {
      graph.outgoing_index()[i + 1] = index->host_ptr()[i + 1];
      for (int j = index->host_ptr()[i]; j < index->host_ptr()[i + 1]; j++) {
        graph.out_neighbors(i, j - index->host_ptr()[i]) = col->host_ptr()[j];
        graph.out_edge_data(i, j - index->host_ptr()[i]) =
            ratings->host_ptr()[j];
        ratings->host_ptr()[j] /= (float)max_rating;
      }
    }
  }
  /************************************************************************
   *
   ************************************************************************/
  void distribute_chunks(std::vector<int>& all_edges) {
    std::vector<int> in_edge_wl(graph.num_edges());
    for (size_t i = 0; i < graph.num_edges(); ++i) {
      in_edge_wl[i] = i;
    }
    size_t num_edges_to_process = in_edge_wl.size();
    int num_items               = graph.num_edges();
    all_edges.resize(num_items);
    memcpy(all_edges.data(), in_edge_wl.data(), num_items * sizeof(int));
  }
  /************************************************************************
   *
   ************************************************************************/
  void operator()(int num_steps) {
    // print_edges();
    // print_latent();
    // max_rating = 1;
    copy_to_device();
    compute_err(graph, features, max_rating);
    for (round = 0; round < num_steps; ++round) {
      this->gpu_operator();
      copy_to_host();
      float rmse = compute_err(graph, features, max_rating);
      if (rmse < 0.1)
        break;
    }
    stats.print();
    // print_latent();
  }

  void print_latent() {
    for (int n = 0; n < 10; n++) {
      FeatureType* features_l = &(features->host_ptr()[n * SGD_FEATURE_SIZE]);
      printf("latent(%d)[%.3f", n, features_l[0]);
      for (int i = 1; i < SGD_FEATURE_SIZE; i++)
        printf(" %.3f", features_l[i]);
      printf("]\n");
    }
  }
  void print_edges() {
    std::cout << "edges: [" << graph.out_edge_data()[0];
    for (int n = 1; n < 100; n++) {
      if (n >= graph.num_edges())
        break;
      std::cout << ", " << graph.out_edge_data()[n];
    }
    printf("]\n");
  }
  /************************************************************************
   *
   ************************************************************************/
  void gpu_operator() {
    int curr_step           = 0;
    metadata->host_ptr()[4] = graph.num_edges();
    double total_time       = 0;
    double insp_time        = 0;
    metadata->host_ptr()[2] = movies.size();
    metadata->host_ptr()[4] = 0;
    metadata->host_ptr()[0] = user_indices.size();
    metadata->copy_to_device();

    const float step_size = SGD_STEP_SIZE(round);
    dim3 block_size       = dim3(SGD_FEATURE_SIZE, BLOCKSIZE);
    int num_blocks        = ceil(movies.size() / (float)BLOCKSIZE);
    cudaError_t err;

    Timer timer, timer2;
    int iter;
    int num_items = graph.num_edges();
    int iter2     = 0;
    timer2.start();
    diag_inspector(movies.size(), user_indices.size(), index->host_ptr(),
                   ratings->host_ptr(), col->host_ptr(), iter2);
    timer2.stop();
    insp_time += timer2.get_time_seconds();
    new_col->copy_to_device();
    new_ratings->copy_to_device();
    new_index->copy_to_device();
    //		diag_number->copy_to_device();
    timer.start();

    int non_zero_blk_diags = 0;
    for (iter = 0; iter < count_of_diagonals; iter++) {
      if (new_index->host_ptr()[iter + 1] - new_index->host_ptr()[iter] > 0) {
        non_zero_blk_diags++;
        num_blocks =
            (new_index->host_ptr()[iter + 1] - new_index->host_ptr()[iter]) %
                        (BLOCKSIZE) ==
                    0
                ? (new_index->host_ptr()[iter + 1] -
                   new_index->host_ptr()[iter]) /
                      (BLOCKSIZE)
                :

                (new_index->host_ptr()[iter + 1] -
                 new_index->host_ptr()[iter]) /
                        (BLOCKSIZE) +
                    1;
        // std::cout << "num_blocks = " << num_blocks << "block_size = " <<
        // SGD_FEATURE_SIZE * BLOCKSIZE << "\n";
        sgd_blk_diag_operator<<<num_blocks, block_size>>>(
            features->device_ptr(), metadata->device_ptr(),
            new_ratings->device_ptr(), new_col->device_ptr(),
            new_index->device_ptr(), step_size, iter);
      }
    }

    cudaDeviceSynchronize();
    timer.stop();

    total_time += timer.get_time_seconds();
    if ((err = cudaGetLastError()) != cudaSuccess) {
      fprintf(stderr, "aborted %s \n", cudaGetErrorString(err));
      exit(-1);
    }

    metadata->copy_to_host();
    // fprintf(stderr, "blk_diag: round %d curr_step %d total_time %.3f
    // per_diag_time %6.3g insp_time %.3f\t", round, 		curr_step,
    // total_time,
    // total_time / (double) count_of_diagonals, insp_time);
    printf("round %d: total_time %.3f\t", round, total_time);
    stats.push_stats(round, curr_step, total_time,
                     total_time / (double)count_of_diagonals, insp_time);

    delete new_ratings;
    delete new_index;
    delete new_col;
  }

  int diag_inspector(int movies, int users, int* index, float* a, int* col,
                     int iter) {

    int t6;
    int t4;
    int t2;
    int newVariable4;
    int newVariable3;
    int newVariable2;
    struct a_list* _P_DATA4;
    int newVariable1;
    int newVariable0;
    struct mk* _P_DATA3;
    struct a_list** _P1;
    int chill_count_1;
    int* _P_DATA1;
    int _t31;
    int _t34;
    /*
    int t8;
    int *_P_DATA2;
    int chill_count_0;
    int _t39;
    int _t38;
    int _t37;
    int In_3;
    int In_2;
    int In_1;
    int _t36;
    int _t35;
    int _t33;
    int _t32;
    int _t30;
    int _t29;
    int _t28;
    int _t27;
    int _t25;
    int _t26;
    int _t24;
    int _t23;
    int _t22;
    int _t21;
    int _t20;
    int _t19;
    int _t18;
    int _t17;
    int _t16;
    int _t15;
    int _t14;
    int _t12;
    int _t11;
    int _t10;
    int _t9;
    int _t7;
    int _t6;
    int _t5;
    int _t4;
    int l;
    int _t3;
    int _t2;
    int _t1;
    int i;
    int j;
    int k;
    */
    _P_DATA1      = (int*)malloc(sizeof(int) * (users / C + movies / R));
    _P1           = (struct a_list**)malloc(sizeof(struct a_list*) *
                                  (users / C + movies / R - 1));
    _P_DATA1[0]   = 0;
    _P_DATA3      = ((
        struct mk*)(malloc(sizeof(struct mk) * (users / C + movies / R - 1))));
    chill_count_1 = 0;
    _P_DATA1[0]   = 0;
    for (_t31 = 0; _t31 <= users / C + movies / R - 2; _t31 += 1) {
      _P1[1 * _t31]          = 0;
      _P_DATA1[1 * _t31 + 1] = 0;
    }
    for (t2 = 0; t2 <= movies / R - 1; t2 += 1) {
      for (t4 = 0; t4 <= R - 1; t4 += 1)
        for (t6 = index_(R * t2 + t4); t6 <= index__(R * t2 + t4) - 1;
             t6 += 1) {
          _t31 = (col_(t6) - movies + R * (movies / R - 1)) / C - t2;
          _P_DATA3[_t31].ptr = 0;
        }
      for (t4 = 0; t4 <= R - 1; t4 += 1)
        for (t6 = index_(R * t2 + t4); t6 <= index__(R * t2 + t4) - 1;
             t6 += 1) {
          _t31 = (col_(t6) - movies + R * (movies / R - 1)) / C - t2;
          _t34 = (col_(t6) - movies + R * (movies / R - 1)) % C;
          if (_P_DATA3[_t31].ptr == 0) {
            _P_DATA4 = ((struct a_list*)(malloc(sizeof(struct a_list) * 1)));
            _P_DATA4->next     = _P1[_t31];
            _P1[_t31]          = _P_DATA4;
            _P_DATA3[_t31].ptr = _P1[_t31];
            for (newVariable0 = 0; newVariable0 <= R - 1; newVariable0 += 1)
              for (newVariable1 = 0; newVariable1 <= C - 1; newVariable1 += 1)
                _P_DATA3[_t31]
                    .ptr->ratings[C * newVariable0 + 1 * newVariable1] = -1;
            _P_DATA3[_t31].ptr->col_[0] = t2;
            chill_count_1 += 1;
            _P_DATA1[_t31 + 1] += 1;
          }
          _P_DATA3[_t31].ptr->ratings[C * t4 + 1 * _t34] = a[t6];
        }
    }

    new_col     = new ArrayType(chill_count_1);
    new_index   = new ArrayType(users / C + movies / R);
    new_ratings = new FeatureArrayType(chill_count_1 * R * C);
    new_col->create_on_device();
    new_index->create_on_device();
    new_ratings->create_on_device();
    new_index->host_ptr()[0] = 0;
    for (t2 = 0; t2 <= users / C + movies / R - 2; t2 += 1) {
      for (newVariable2 = 1 - _P_DATA1[1 * t2 + 1]; newVariable2 <= 0;
           newVariable2 += 1) {
        new_col->host_ptr()[_P_DATA1[1 * t2] - newVariable2] =
            _P1[1 * t2]->col_[0];
        for (newVariable3 = 0; newVariable3 <= R - 1; newVariable3 += 1)
          for (newVariable4 = 0; newVariable4 <= C - 1; newVariable4 += 1)
            new_ratings->host_ptr()[R * C * (_P_DATA1[1 * t2] - newVariable2) +
                                    C * newVariable3 + 1 * newVariable4] =
                _P1[1 * t2]->ratings[C * newVariable3 + 1 * newVariable4];
        _P_DATA4 = _P1[1 * t2]->next;
        free(_P1[1 * t2]);
        _P1[1 * t2] = _P_DATA4;
      }
      _P_DATA1[1 * t2 + 1] += _P_DATA1[1 * t2];
      new_index->host_ptr()[t2 + 1] = _P_DATA1[t2 + 1];
    }

    count_of_diagonals = users / C + movies / R - 1;
    free(_P_DATA1);
    free(_P_DATA3);
    free(_P1);
    return chill_count_1;
  }

  /************************************************************************
   *
   ************************************************************************/
  ~SGDAsynEdgeCudaFunctor() {
    deallocate();
    // fprintf(stderr, "Destroying SGDAsynEdgeCudaFunctor object.\n");
  }
};
//###################################################################//

#endif /* GALOISGPU_APPS_SGD_CUDA_SGDASYNCEDGECU_H_ */
