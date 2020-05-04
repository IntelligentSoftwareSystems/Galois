/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "coop_conv=False $ outline_iterate_gb=False $ backoff_blocking_factor=4 $ parcomb=False $ np_schedulers=set(['tb', 'fg']) $ cc_disable=set([]) $ hacks=set([]) $ np_factor=8 $ instrument=set([]) $ unroll=[] $ instrument_mode=None $ read_props=None $ outline_iterate=True $ ignore_nested_errors=False $ np=True $ write_props=None $ quiet_cgen=True $ retry_backoff=True $ cuda.graph_type=texture $ cuda.use_worklist_slots=True $ cuda.worklist_type=texture";
void debug_output(CSRGraphTy &g, unsigned int *valid_edges);;
static const int __tb_preprocess = TB_SIZE;
static const int __tb_count_triangles = TB_SIZE;
__global__ void preprocess(CSRGraph graph, unsigned int * valid_edges)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_preprocess;
  index_type node_end;
  index_type node_rup;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct empty_np, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  const index_type last  = graph.nnodes;
  node_end = (graph).nnodes;
  node_rup = ((0) + roundup((((graph).nnodes) - (0)), (blockDim.x)));
  for (index_type node = 0 + tid; node < node_rup; node += nthreads)
  {
    bool pop;
    int degree;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = graph.valid_node(node);;
    if (pop)
    {
      degree = graph.getOutDegree(node);
    }
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { index_type node; int degree; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].node = node;
    _np_closure[threadIdx.x].degree = degree;
    if (pop)
    {
      _np.size = (graph).getOutDegree(node);
      _np.start = (graph).getFirstEdge(node);
    }
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_TB ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_TB ? _np.size : 0;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    __syncthreads();
    while (true)
    {
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      __syncthreads();
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        __syncthreads();
        break;
      }
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      __syncthreads();
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      assert(nps.tb.src < __kernel_tb_size);
      node = _np_closure[nps.tb.src].node;
      degree = _np_closure[nps.tb.src].degree;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type edge;
        edge = ns +_np_j;
        {
          index_type dst = graph.getAbsDestination(edge);
          int dst_degree = graph.getOutDegree(dst);
          if ((dst_degree > degree) || (dst_degree == degree && dst > node))
          {
            graph.edge_data[edge] = dst;
            atomicAdd(valid_edges + node, 1);
          }
          else
          {
            graph.edge_data[edge] = graph.nnodes;
          }
        }
      }
      __syncthreads();
    }

    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        node = _np_closure[nps.fg.src[_np_i]].node;
        degree = _np_closure[nps.fg.src[_np_i]].degree;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type dst = graph.getAbsDestination(edge);
          int dst_degree = graph.getOutDegree(dst);
          if ((dst_degree > degree) || (dst_degree == degree && dst > node))
          {
            graph.edge_data[edge] = dst;
            atomicAdd(valid_edges + node, 1);
          }
          else
          {
            graph.edge_data[edge] = graph.nnodes;
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    node = _np_closure[threadIdx.x].node;
    degree = _np_closure[threadIdx.x].degree;
  }
}
__device__ unsigned int intersect(CSRGraph graph, index_type u, index_type v, unsigned int * valid_edges)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type u_start = graph.getFirstEdge(u);
  index_type u_end = u_start + valid_edges[u];
  index_type v_start = graph.getFirstEdge(v);
  index_type v_end = v_start + valid_edges[v];
  int count = 0;
  index_type u_it = u_start;
  index_type v_it = v_start;
  index_type a ;
  index_type b ;
  while (u_it < u_end && v_it < v_end)
  {
    a = graph.getAbsDestination(u_it);
    b = graph.getAbsDestination(v_it);
    int d = a - b;
    if (d <= 0)
    {
      u_it++;
    }
    if (d >= 0)
    {
      v_it++;
    }
    if (d == 0)
    {
      count++;
    }
  }
  return count;
}
__global__ void count_triangles(CSRGraph graph, unsigned int * valid_edges, int * count)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = __tb_count_triangles;
  index_type v_end;
  index_type v_rup;
  const int _NP_CROSSOVER_WP = 32;
  const int _NP_CROSSOVER_TB = __kernel_tb_size;
  const int BLKSIZE = __kernel_tb_size;
  const int ITSIZE = BLKSIZE * 8;

  typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
  typedef union np_shared<BlockScan::TempStorage, index_type, struct tb_np, struct empty_np, struct fg_np<ITSIZE> > npsTy;

  __shared__ npsTy nps ;
  int lcount =0;
  v_end = (graph).nnodes;
  v_rup = ((0) + roundup((((graph).nnodes) - (0)), (blockDim.x)));
  for (index_type v = 0 + tid; v < v_rup; v += nthreads)
  {
    bool pop;
    int d_v;
    multiple_sum<2, index_type> _np_mps;
    multiple_sum<2, index_type> _np_mps_total;
    pop = graph.valid_node(v);;
    struct NPInspector1 _np = {0,0,0,0,0,0};
    __shared__ struct { index_type v; } _np_closure [TB_SIZE];
    _np_closure[threadIdx.x].v = v;
    if (pop)
    {
      _np.size = ((graph).getFirstEdge(v)+ valid_edges[v]) - ((graph).getFirstEdge(v));
      _np.start = (graph).getFirstEdge(v);
    }
    _np_mps.el[0] = _np.size >= _NP_CROSSOVER_TB ? _np.size : 0;
    _np_mps.el[1] = _np.size < _NP_CROSSOVER_TB ? _np.size : 0;
    BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
    if (threadIdx.x == 0)
    {
      nps.tb.owner = MAX_TB_SIZE + 1;
    }
    __syncthreads();
    while (true)
    {
      if (_np.size >= _NP_CROSSOVER_TB)
      {
        nps.tb.owner = threadIdx.x;
      }
      __syncthreads();
      if (nps.tb.owner == MAX_TB_SIZE + 1)
      {
        __syncthreads();
        break;
      }
      if (nps.tb.owner == threadIdx.x)
      {
        nps.tb.start = _np.start;
        nps.tb.size = _np.size;
        nps.tb.src = threadIdx.x;
        _np.start = 0;
        _np.size = 0;
      }
      __syncthreads();
      int ns = nps.tb.start;
      int ne = nps.tb.size;
      if (nps.tb.src == threadIdx.x)
      {
        nps.tb.owner = MAX_TB_SIZE + 1;
      }
      assert(nps.tb.src < __kernel_tb_size);
      v = _np_closure[nps.tb.src].v;
      for (int _np_j = threadIdx.x; _np_j < ne; _np_j += BLKSIZE)
      {
        index_type edge;
        edge = ns +_np_j;
        {
          index_type u = graph.getAbsDestination(edge);
          index_type d_u = graph.getOutDegree(u);
          int xcount = 0;
          xcount = intersect(graph, u, v, valid_edges);
          if (xcount)
          {
            atomicAdd(count, xcount);
          }
        }
      }
      __syncthreads();
    }

    _np.total = _np_mps_total.el[1];
    _np.offset = _np_mps.el[1];
    while (_np.work())
    {
      int _np_i =0;
      _np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
      __syncthreads();

      for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
      {
        index_type edge;
        assert(nps.fg.src[_np_i] < __kernel_tb_size);
        v = _np_closure[nps.fg.src[_np_i]].v;
        edge= nps.fg.itvalue[_np_i];
        {
          index_type u = graph.getAbsDestination(edge);
          index_type d_u = graph.getOutDegree(u);
          int xcount = 0;
          xcount = intersect(graph, u, v, valid_edges);
          if (xcount)
          {
            atomicAdd(count, xcount);
          }
        }
      }
      _np.execute_round_done(ITSIZE);
      __syncthreads();
    }
    assert(threadIdx.x < __kernel_tb_size);
    v = _np_closure[threadIdx.x].v;
  }
}
__global__ void print_matrix_kernel(CSRGraph graph, unsigned int __begin, unsigned int __end, int hostid)
{
	unsigned tid = TID_1D;
	unsigned nthreads = TOTAL_THREADS_1D;
	unsigned long long count = 0;
	if(tid == 0) {
		for (index_type src = __begin + tid; src < __end; src++)
		{
				unsigned ne = (graph).getOutDegree(src);
				//limit the edges to 20 only
				if(ne > 10) ne = 10;
				int ns = (graph).getFirstEdge(src);
				for (int _np_j = 0; _np_j < ne; _np_j++)
				{
						index_type jj = ns +_np_j;
						index_type dst;
						dst = graph.getAbsDestination(jj);
						edge_data_type wt;
						wt = graph.getAbsWeight(jj);
						printf("[%d] %d %d %d degree: %d \n", hostid, src, dst, wt, (graph).getOutDegree(src));
						//printf("%d %d %d \n", src, dst, wt);
						unsigned long long weight;
						weight = wt;
						count += weight;
				}
		}
	}
	
}
void gg_main(CSRGraphTy& hg, CSRGraphTy& gg)
{
  dim3 blocks, threads;
  kernel_sizing(gg, blocks, threads);
  Shared<int> count (1);
  Shared<unsigned int> valid_edges (hg.nnodes);
  count.zero_gpu();
  valid_edges.zero_gpu();
  preprocess <<<blocks, __tb_preprocess>>>(gg, valid_edges.gpu_wr_ptr());
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, gg.edge_data , gg.edge_data, gg.edge_dst, gg.edge_dst, 
                                               gg.nedges, gg.nnodes - 1,  gg.row_start, gg.row_start + 1);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, gg.edge_data , gg.edge_data, gg.edge_dst, gg.edge_dst, 
                                               gg.nedges, gg.nnodes - 1,  gg.row_start, gg.row_start + 1);
  count_triangles <<<blocks, __tb_count_triangles>>>(gg, valid_edges.gpu_rd_ptr(), count.gpu_wr_ptr());
  printf("triangles: %d\n", *count.cpu_rd_ptr());
}
