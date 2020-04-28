/*
   csr_graph.cu

   Implements CSR Graph. Part of the GGC source code. 

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

/* -*- mode: c++ -*- */

#include "gg.h"
#include "csr_graph.h"

unsigned CSRGraph::init() {
  row_start = edge_dst = NULL;
  edge_data = NULL;
  node_data = NULL;
  nnodes = nedges = 0;
  device_graph = false;

  return 0;
}

unsigned CSRGraph::allocOnHost(bool no_edge_data) {
  assert(nnodes > 0);
  assert(!device_graph);

  if(row_start != NULL) // already allocated
    return true;

  size_t mem_usage = ((nnodes + 1) + nedges) * sizeof(index_type) 
    + (nnodes) * sizeof(node_data_type);
  if (!no_edge_data) mem_usage += (nedges) * sizeof(edge_data_type);
    
  printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);

  row_start = (index_type *) calloc(nnodes+1, sizeof(index_type));
  edge_dst  = (index_type *) calloc(nedges, sizeof(index_type));
  if (!no_edge_data) edge_data = (edge_data_type *) calloc(nedges, sizeof(edge_data_type));
  node_data = (node_data_type *) calloc(nnodes, sizeof(node_data_type));

  return ((no_edge_data || edge_data) && row_start && edge_dst && node_data);
}

unsigned CSRGraph::allocOnDevice(bool no_edge_data) {
  if(edge_dst != NULL)  // already allocated
    return true;  

  assert(edge_dst == NULL); // make sure not already allocated

  check_cuda(cudaMalloc((void **) &edge_dst, nedges * sizeof(index_type)));
  check_cuda(cudaMalloc((void **) &row_start, (nnodes+1) * sizeof(index_type)));

  if (!no_edge_data) check_cuda(cudaMalloc((void **) &edge_data, nedges * sizeof(edge_data_type)));
  check_cuda(cudaMalloc((void **) &node_data, nnodes * sizeof(node_data_type)));

  device_graph = true;

  assert(edge_dst && (no_edge_data || edge_data) && row_start && node_data);
  return true;
}

void CSRGraphTex::copy_to_gpu(struct CSRGraphTex &copygraph) {
  copygraph.nnodes = nnodes;
  copygraph.nedges = nedges;
  
  copygraph.allocOnDevice(edge_data == NULL);

  check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst, nedges * sizeof(index_type), cudaMemcpyHostToDevice));
  if (edge_data != NULL) check_cuda(cudaMemcpy(copygraph.edge_data, edge_data, nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(copygraph.node_data, node_data, nnodes * sizeof(node_data_type), cudaMemcpyHostToDevice));

  check_cuda(cudaMemcpy(copygraph.row_start, row_start, (nnodes+1) * sizeof(index_type), cudaMemcpyHostToDevice));
}

unsigned CSRGraphTex::allocOnDevice(bool no_edge_data) {
  if(CSRGraph::allocOnDevice(no_edge_data)) 
    {
      assert(sizeof(index_type) <= 4); // 32-bit only!
      assert(sizeof(node_data_type) <= 4); // 32-bit only!

      cudaResourceDesc resDesc;

      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
      resDesc.res.linear.desc.x = 32; // bits per channel

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = cudaReadModeElementType;

      resDesc.res.linear.devPtr = edge_dst;
      resDesc.res.linear.sizeInBytes = nedges*sizeof(index_type);
      check_cuda(cudaCreateTextureObject(&edge_dst_tx, &resDesc, &texDesc, NULL));

      resDesc.res.linear.devPtr = row_start;
      resDesc.res.linear.sizeInBytes = (nnodes + 1) * sizeof(index_type);
      check_cuda(cudaCreateTextureObject(&row_start_tx, &resDesc, &texDesc, NULL));

      resDesc.res.linear.devPtr = node_data;
      resDesc.res.linear.sizeInBytes = (nnodes) * sizeof(node_data_type);
      check_cuda(cudaCreateTextureObject(&node_data_tx, &resDesc, &texDesc, NULL));

      return 1;
    }

  return 0;
}

unsigned CSRGraph::deallocOnHost() {
  if(!device_graph) {
    free(row_start);
    free(edge_dst);
    if (edge_data != NULL) free(edge_data);
    free(node_data);
  }

  return 0;
}
unsigned CSRGraph::deallocOnDevice() {
  if(device_graph) {
    cudaFree(edge_dst);
    if (edge_data != NULL) cudaFree(edge_data);
    cudaFree(row_start);
    cudaFree(node_data);
  }

  return 0;
}

CSRGraph::CSRGraph() {
  init();
}

void CSRGraph::progressPrint(unsigned maxii, unsigned ii) {
  const unsigned nsteps = 10;
  unsigned ineachstep = (maxii / nsteps);
  if(ineachstep == 0) ineachstep = 1;
  /*if (ii == maxii) {
    printf("\t100%%\n");
    } else*/ if (ii % ineachstep == 0) {
    int progress = ((size_t) ii * 100) / maxii + 1;

    printf("\t%3d%%\r", progress);
    fflush(stdout);
  }
}

unsigned CSRGraph::readFromGR(const char file[], bool read_edge_data) {
  std::ifstream cfile;
  cfile.open(file);

  // copied from GaloisCpp/trunk/src/FileGraph.h
  int masterFD = open(file, O_RDONLY);
  if (masterFD == -1) {
    printf("FileGraph::structureFromFile: unable to open %s.\n", file);
    return 1;
  }

  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
    abort();
  }
  size_t masterLength = buf.st_size;

  int _MAP_BASE = MAP_PRIVATE;
  //#ifdef MAP_POPULATE
  //  _MAP_BASE  |= MAP_POPULATE;
  //#endif

  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    printf("FileGraph::structureFromFile: mmap failed.\n");
    abort();
  }

  ggc::Timer t("graphreader");
  t.start();

  //parse file
  uint64_t* fptr = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes = le64toh(*fptr++);
  uint64_t numEdges = le64toh(*fptr++);
  uint64_t *outIdx = fptr;
  fptr += numNodes;
  uint32_t *fptr32 = (uint32_t*)fptr;
  uint32_t *outs = fptr32; 
  fptr32 += numEdges;
  if (numEdges % 2) fptr32 += 1;
  edge_data_type  *edgeData = (edge_data_type *)fptr32;
	
  // cuda.
  nnodes = numNodes;
  nedges = numEdges;

  printf("nnodes=%d, nedges=%d, sizeEdge=%d.\n", nnodes, nedges, sizeEdgeTy);
  allocOnHost(!read_edge_data);

  row_start[0] = 0;

  for (unsigned ii = 0; ii < nnodes; ++ii) {
    row_start[ii+1] = le64toh(outIdx[ii]);
    //   //noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
    index_type degree = row_start[ii+1] - row_start[ii];

    for (unsigned jj = 0; jj < degree; ++jj) {
      unsigned edgeindex = row_start[ii] + jj;

      unsigned dst = le32toh(outs[edgeindex]);
      if (dst >= nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);

      edge_dst[edgeindex] = dst;

      if(sizeEdgeTy && read_edge_data)
        edge_data[edgeindex] = edgeData[edgeindex];
    }

    progressPrint(nnodes, ii);
  }

  cfile.close();	// probably galois doesn't close its file due to mmap.
  t.stop();

  // TODO: fix MB/s
  printf("read %lld bytes in %d ms (%0.2f MB/s)\n\r\n", masterLength, t.duration_ms(), (masterLength / 1000.0) / (t.duration_ms()));

  return 0;
}

unsigned CSRGraph::read(const char file[], bool read_edge_data) {
  return readFromGR(file, read_edge_data);
}

void CSRGraph::dealloc() {
  if(device_graph) 
    deallocOnDevice();
  else
    deallocOnHost();
}

void CSRGraph::copy_to_gpu(struct CSRGraph &copygraph) {
  copygraph.nnodes = nnodes;
  copygraph.nedges = nedges;
  
  copygraph.allocOnDevice(edge_data == NULL);

  check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst, nedges * sizeof(index_type), cudaMemcpyHostToDevice));
  if (edge_data != NULL) check_cuda(cudaMemcpy(copygraph.edge_data, edge_data, nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(copygraph.node_data, node_data, nnodes * sizeof(node_data_type), cudaMemcpyHostToDevice));

  check_cuda(cudaMemcpy(copygraph.row_start, row_start, (nnodes+1) * sizeof(index_type), cudaMemcpyHostToDevice));
}

void CSRGraph::copy_to_cpu(struct CSRGraph &copygraph) {
  assert(device_graph);
  
  // cpu graph is not allocated
  assert(copygraph.nnodes = nnodes);
  assert(copygraph.nedges = nedges);
  
  check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst, nedges * sizeof(index_type), cudaMemcpyDeviceToHost));
  if (edge_data != NULL) check_cuda(cudaMemcpy(copygraph.edge_data, edge_data, nedges * sizeof(edge_data_type), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(copygraph.node_data, node_data, nnodes * sizeof(node_data_type), cudaMemcpyDeviceToHost));

  check_cuda(cudaMemcpy(copygraph.row_start, row_start, (nnodes+1) * sizeof(index_type), cudaMemcpyDeviceToHost));
}

struct EdgeIterator {
  CSRGraph *g;
  index_type node;
  index_type s;

  __device__
  EdgeIterator(CSRGraph& g, index_type node) {
    this->g = &g;
    this->node = node;
  }

  __device__
  index_type size() const {
    return g->row_start[node + 1] - g->row_start[node];
  }

  __device__
  index_type start() {
    s = g->row_start[node];
    return s;
  }

  __device__
  index_type end() const {
    return g->row_start[node + 1];
  }

  __device__
  void next() {
    s++;
  }

  __device__
  index_type dst() const {
    return g->edge_dst[s];
  }

  __device__
  edge_data_type data() const {
    return g->edge_data[s];
  }
};

