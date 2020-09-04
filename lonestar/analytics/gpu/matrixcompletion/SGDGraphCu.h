/*
 * SGDGraphCu.h
 *
 *  Created on: Nov 12, 2014
 *      Author: rashid
 */
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <cassert>
#ifdef _WIN32
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#ifndef GALOISGPU_APPS_SGD_CUDA_SGDGRAPHCU_H_
#define GALOISGPU_APPS_SGD_CUDA_SGDGRAPHCU_H_

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#define le64toh(x) (x) // OSSwapLittleToHostInt64(x)
#define le32toh(x) (x) //  OSSwapLittleToHostInt32(x)
#elif __FreeBSD__
#include <sys/endian.h>
#elif __linux__
typedef ulong uint64_t;
typedef uint uint32_t;
#include <endian.h>
#ifndef le64toh
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define le64toh(x) (x)
#define le32toh(x) (x)
#else
#define le64toh(x) __bswap_64(x)
#define le32toh(x) __bswap_32(x)
#endif
#endif
#else
#endif

/*
 * LC_LinearArray_Undirected_Graph.h
 *
 *  Created on: Oct 24, 2013
 *  Single array representation, has outgoing edges.
 *      Author: rashid
 */

template <typename NodeDataTy, typename EdgeDataTy>
struct SGD_LC_LinearArray_Undirected_Graph {
  // Are you using gcc/4.7+ Error on line below for earlier versions.
  typedef NodeDataTy NodeDataType;
  typedef EdgeDataTy EdgeDataType;
  typedef unsigned int NodeIDType;
  typedef unsigned int EdgeIDType;
  size_t _num_nodes;
  size_t _num_edges;
  unsigned int _max_degree;
  const size_t SizeEdgeData;
  const size_t SizeNodeData;
  int* gpu_graph;
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  SGD_LC_LinearArray_Undirected_Graph()
      : SizeEdgeData(sizeof(EdgeDataType) / sizeof(unsigned int)),
        SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
    _max_degree = _num_nodes = _num_edges = 0;
    gpu_graph                             = 0;
  }
  void read(const char* filename) {
    readFromGR(filename);
    for (unsigned int i = 0; i < num_nodes(); ++i) {
      for (unsigned int e = outgoing_index()[i]; e < outgoing_index()[i + 1];
           ++e) {
        get_edge_src()[e] = i;
      }
    }
  }
  unsigned inline readFromGR(const char* file) {
    std::ifstream cfile;
    cfile.open(file);

    // copied from GaloisCpp/trunk/src/FileGraph.h
    int masterFD = open(file, O_RDONLY);
    if (masterFD == -1) {
      printf("FileGraph::structureFromFile: unable to open %s.\n", file);
      abort();
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

    // parse file
    uint64_t* fptr                           = (uint64_t*)m;
    __attribute__((unused)) uint64_t version = le64toh(*fptr++);
    assert(version == 1);
    __attribute__((unused)) uint64_t sizeEdgeTy = le64toh(*fptr++);
    uint64_t numNodes                           = le64toh(*fptr++);
    uint64_t numEdges                           = le64toh(*fptr++);
    uint64_t* outIdx                            = fptr;
    fptr += numNodes;
    uint32_t* fptr32 = (uint32_t*)fptr;
    uint32_t* outs   = fptr32;
    fptr32 += numEdges;
    if (numEdges % 2)
      fptr32 += 1;
    unsigned* edgeData = (unsigned*)fptr32;

    _num_nodes = numNodes;
    _num_edges = numEdges;
    std::cout << "num_nodes: " << _num_nodes << ", num_edges: " << _num_edges
              << "\n";
    init(_num_nodes, _num_edges);
    // node_data
    memset(node_data(), 0, sizeof(unsigned int) * _num_nodes);
    for (unsigned int i = 0; i < _num_edges; ++i) {
      out_neighbors()[i] = le32toh(outs[i]);
    }
    outgoing_index()[0] = 0;
    for (unsigned int i = 0; i < _num_nodes; ++i) {
      outgoing_index()[i + 1] = le32toh(outIdx[i]);
    }
    unsigned int start        = 0;
    unsigned int displacement = 0;
    for (unsigned int i = 0; i < _num_nodes; ++i) {
      unsigned int end = le32toh(outIdx[i]);
      for (unsigned int idx = start; idx < end; ++idx) {
        // node i's idx neighbor is to be populated here.
        out_edge_data()[displacement] = le32toh(edgeData[idx]);
        // out_edge_data()[displacement] = 1;
        out_neighbors()[displacement] = le32toh(outs[idx]);
        displacement++;
      }
      start = end;
    }
    /*   for (size_t i = 0; i < g._num_nodes; ++i)
          g.node_data()[i] = std::numeric_limits<unsigned int>::max() / 2;*/
    cfile.close();
    update_in_neighbors();
    return 0;
  }

  NodeDataType* node_data() { return (NodeDataType*)gpu_graph + 4; }
  unsigned int* outgoing_index() {
    return (unsigned int*)(node_data()) + _num_nodes * SizeNodeData;
  }
  unsigned int outgoing_index(const int idx) const {
    return ((unsigned int*)(gpu_graph + 4) + _num_nodes * SizeNodeData)[idx];
  }
  unsigned int* out_neighbors() {
    return (unsigned int*)outgoing_index() + _num_nodes + 1;
  }
  EdgeDataType* out_edge_data() {
    return (EdgeDataType*)(unsigned int*)(out_neighbors()) + _num_edges;
  }
  EdgeDataType& out_edge_data(unsigned int node_id, unsigned int nbr_id) {
    return ((EdgeDataType*)out_edge_data())[outgoing_index()[node_id] + nbr_id];
  }
  unsigned int& out_neighbors(unsigned int node_id, unsigned int nbr_id) {
    return ((unsigned int*)out_neighbors())[outgoing_index()[node_id] + nbr_id];
  }
  unsigned int* incoming_index() { return outgoing_index(); }
  unsigned int* in_neighbors() { return outgoing_index(); }
  EdgeDataType* in_edge_data() { return out_edge_data(); }
  unsigned int* get_edge_src() {
    return (unsigned*)out_edge_data() + _num_edges;
  }
  unsigned int get_edge_src(int edge_index) {
    return get_edge_src()[edge_index];
  }
  unsigned int* last() {
    return (unsigned int*)in_edge_data() + _num_edges * SizeEdgeData;
  }

  size_t num_nodes() { return _num_nodes; }
  size_t num_edges() { return _num_edges; }
  size_t num_neighbors(const unsigned int node_id) const {
    return outgoing_index(node_id + 1) - outgoing_index(node_id);
  }
  size_t max_degree() { return _max_degree; }
  void init(size_t n_n, size_t n_e) {
    _num_nodes = n_n;
    _num_edges = n_e;
    // const int arr_size = (4 + (_num_nodes * SizeNodeData) + (_num_nodes + 1)
    // + (_num_edges) + (_num_edges * SizeEdgeData) + (_num_edges)); std::cout
    // << "Allocating NN: " << _num_nodes << "(" << SizeNodeData << ") , NE :"
    // << _num_edges << ", TOTAL:: " << arr_size << "\n"; Num_nodes, num_edges,
    // [node_data] , [outgoing_index], [out_neighbors], [edge_data] , [src
    // indices] fprintf(stderr, "GraphSize :: %6.6g MB\n", arr_size /
    // (float(1024
    // * 1024)));
    gpu_graph =
        new int[(4 + (_num_nodes * SizeNodeData) + (_num_nodes + 1) +
                 (_num_edges) + (_num_edges * SizeEdgeData) + (_num_edges))];
    (gpu_graph)[0] = (int)_num_nodes;
    (gpu_graph)[1] = (int)_num_edges;
    (gpu_graph)[2] = (int)SizeNodeData;
    (gpu_graph)[3] = (int)SizeEdgeData;
    // allocate_on_gpu();
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void print_header(void) {
    std::cout << "Header :: [";
    for (unsigned int i = 0; i < 6; ++i) {
      std::cout << gpu_graph[i] << ",";
    }
    std::cout << "\n";
    return;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void print_node(unsigned int idx, const char* post = "") {
    if (idx < _num_nodes) {
      std::cout << "N-" << idx << "(" << (node_data())[idx] << ")"
                << " :: [";
      for (size_t i = (outgoing_index())[idx]; i < (outgoing_index())[idx + 1];
           ++i) {
        std::cout << " " << (out_neighbors())[i] << "(" << (out_edge_data())[i]
                  << "<" << node_data()[out_neighbors()[i]] << ">"
                  << "), ";
      }
      std::cout << "]" << post;
    }
    return;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void print_graph(void) {
    std::cout << "\n====Printing graph (" << _num_nodes << " , " << _num_edges
              << ")=====\n";
    for (size_t i = 0; i < _num_nodes; ++i) {
      print_node(i);
      std::cout << "\n";
    }
    return;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void update_in_neighbors(void) {}
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void print_compact(void) {
    std::cout << "Summary:: [" << _num_nodes << ", " << _num_edges << ", "
              << outgoing_index()[_num_nodes] << "]";
    std::cout << "\nOut-index [";
    for (size_t i = 0; i < _num_nodes + 1; ++i) {
      if (i < _num_nodes && outgoing_index()[i] > outgoing_index()[i + 1])
        std::cout << "**ERR**";
      std::cout << " " << outgoing_index()[i] << ",";
    }
    std::cout << "]\nNeigh[";
    for (size_t i = 0; i < _num_edges; ++i) {
      if (out_neighbors()[i] > _num_nodes)
        std::cout << "**ERR**";
      std::cout << " " << out_neighbors()[i] << ",";
    }
    std::cout << "]\nEData [";
    for (size_t i = 0; i < _num_edges; ++i) {
      std::cout << " " << out_edge_data()[i] << ",";
    }
    std::cout << "]";
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  unsigned int verify() {
    unsigned int* t_node_data      = node_data();
    unsigned int* t_outgoing_index = outgoing_index();
    unsigned int* t_neighbors      = out_neighbors();
    unsigned int* t_out_edge_data  = out_edge_data();
    unsigned int err_count         = 0;
    for (unsigned int node_id = 0; node_id < _num_nodes; ++node_id) {
      unsigned int curr_distance = t_node_data[node_id];
      // Go over all the neighbors.
      for (unsigned int idx = t_outgoing_index[node_id];
           idx < t_outgoing_index[node_id + 1]; ++idx) {
        unsigned int temp = t_node_data[t_neighbors[idx]];
        if (curr_distance + t_out_edge_data[idx] < temp) {
          if (err_count < 10) {
            std::cout << "Error :: ";
            print_node(node_id);
            std::cout << " With :: ";
            print_node(t_neighbors[idx]);
            std::cout << "\n";
          }
          err_count++;
        }
      }
    } // End for
    return err_count;
  }
  ////////////##############################################################///////////
  ////////////##############################################################///////////
  unsigned int verify_in() { return 0; }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void deallocate(void) { delete gpu_graph; }
};
// End LC_Graph
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
#endif /* GALOISGPU_APPS_SGD_CUDA_SGDGRAPHCU_H_ */
