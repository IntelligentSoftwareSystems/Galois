/*
 * LC_LinearArray_Graph.h
 *
 *  Created on: Jul 1, 2014
 *  Single array representation, has incoming and outgoing edges.
 *      Author: rashid@cs.utexas.edu
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <type_traits>

#ifndef LC_LinearArray_Graph_H_
#define LC_LinearArray_Graph_H_

namespace galois {
namespace opencl {

static const char* _str_LC_LinearArray_Graph =
    "typedef struct _GraphType { \n"
    "uint _num_nodes;\n"
    "uint _num_edges;\n "
    "uint node_data_size;\n "
    "uint edge_data_size;\n "
    "__global NodeData*node_data;\n "
    "__global uint *out_index;\n "
    "__global uint *out_neighbors;\n "
    "__global EdgeData *out_edge_data;\n }GraphType;\n"
    "uint out_neighbors_begin(__local GraphType * graph, uint node){ \n return "
    "0;\n}\n"
    "uint out_neighbors_end(__local GraphType * graph, uint node){ \n return "
    "graph->out_index[node+1]-graph->out_index[node];\n}\n"
    "uint out_neighbors_next(__local GraphType * graph, uint node){ \n return "
    "1;\n}\n"
    "uint out_neighbors(__local GraphType * graph,uint node,  uint nbr){ \n "
    "return graph->out_neighbors[graph->out_index[node]+nbr];\n}\n"
    "__global NodeData * node_data(__local GraphType * graph, uint node){ \n "
    "return &graph->node_data[node];\n}\n"
    "__global EdgeData * in_edge_data(__local GraphType * graph, uint node, "
    "uint nbr){ \n return &graph->in_edge_data[graph->in_index[node]+nbr];\n}\n"
    "__global EdgeData * out_edge_data(__local GraphType * graph,uint node,  "
    "uint nbr){ \n return "
    "&graph->out_edge_data[graph->out_index[node]+nbr];\n}\n"
    "void initialize(__local GraphType * graph, __global uint "
    "*mem_pool){\nuint offset =4;\n graph->_num_nodes=mem_pool[0];\n"
    "graph->_num_edges=mem_pool[1];\n graph->node_data_size =mem_pool[2];\n "
    "graph->edge_data_size=mem_pool[3];\n"
    "graph->node_data= (__global NodeData *)&mem_pool[offset];\n offset "
    "+=graph->_num_nodes* graph->node_data_size;\n"
    "graph->out_index=&mem_pool[offset];\n offset +=graph->_num_nodes + 1;\n "
    "graph->out_neighbors=&mem_pool[offset];\n"
    "offset +=graph->_num_edges;\n"
    "graph->out_edge_data=(__global EdgeData*)&mem_pool[offset];\n"
    "offset +=graph->_num_edges*graph->edge_data_size;\n}\n}";

template <template <typename> class GPUWrapper, typename NodeDataTy,
          typename EdgeDataTy>

struct LC_LinearArray_Graph {
  // Are you using gcc/4.7+ Error on line below for earlier versions.
#ifdef _WIN32
  typedef GPUWrapper<unsigned int> GPUType;
  typedef typename GPUWrapper<unsigned int>::HostPtrType HostPtrType;
  typedef typename GPUWrapper<unsigned int>::DevicePtrType DevicePtrType;
#else
  template <typename T>
  using ArrayType = GPUWrapper<T>;
  //   typedef GPUWrapper<unsigned int> ArrayType;
  typedef GPUWrapper<unsigned int> GPUType;
  typedef typename GPUWrapper<unsigned int>::HostPtrType HostPtrType;
  typedef typename GPUWrapper<unsigned int>::DevicePtrType DevicePtrType;
#endif
  typedef NodeDataTy NodeDataType;
  typedef EdgeDataTy EdgeDataType;
  typedef unsigned int NodeIDType;
  typedef unsigned int EdgeIDType;
  size_t _num_nodes;
  size_t _num_edges;
  unsigned int _max_degree;
  const size_t SizeEdgeData;
  const size_t SizeNodeData;
  GPUType* gpu_graph;
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  LC_LinearArray_Graph()
      : SizeEdgeData(sizeof(EdgeDataType) / sizeof(unsigned int)),
        SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
    //      fprintf(stderr, "Created LC_LinearArray_Graph with %d node %d edge
    //      data.", (int) SizeNodeData, (int) SizeEdgeData);
    _max_degree = _num_nodes = _num_edges = 0;
    gpu_graph                             = 0;
  }
  template <typename GaloisGraph>
  void load_from_galois(GaloisGraph& ggraph) {
    typedef typename GaloisGraph::GraphNode GNode;
    const size_t gg_num_nodes = ggraph.size();
    const size_t gg_num_edges = ggraph.sizeEdges();
    init(gg_num_nodes, gg_num_edges);
    const int* ptr      = (int*)this->gpu_graph->host_ptr();
    int edge_counter    = 0;
    int node_counter    = 0;
    outgoing_index()[0] = 0;
    for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
      int src_node               = *n;
      getData()[src_node]        = ggraph.getData(*n);
      outgoing_index()[src_node] = edge_counter;
      for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n);
           ++nbr) {
        GNode dst                     = ggraph.getEdgeDst(*nbr);
        out_neighbors()[edge_counter] = dst;
        out_edge_data()[edge_counter] = ggraph.getEdgeData(*nbr);
        //               std::cout<<src_node<<" "<<dst<<"
        //               "<<out_edge_data()[edge_counter]<<"\n";
        edge_counter++;
      }
    }
    outgoing_index()[gg_num_nodes] = edge_counter;
    //         outgoing_index()[node_counter] = edge_counter;
    fprintf(stderr, "Debug :: %d %d \n", node_counter, edge_counter);
    if (node_counter != gg_num_nodes)
      fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %zu\n", node_counter,
              gg_num_nodes);
    assert(edge_counter == gg_num_edges && "Failed to add all edges.");
    fprintf(stderr, "Loaded from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=%lu].\n",
            gg_num_nodes, gg_num_edges, sizeof(NodeDataType),
            sizeof(EdgeDataType));
  }
  template <typename GaloisGraph>
  void writeback_from_galois(GaloisGraph& ggraph) {
    typedef typename GaloisGraph::GraphNode GNode;
    const size_t gg_num_nodes = ggraph.size();
    const size_t gg_num_edges = ggraph.sizeEdges();
    int edge_counter          = 0;
    int node_counter          = 0;
    outgoing_index()[0]       = 0;
    for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
      int src_node = *n;
      //               std::cout<<*n<<", "<<ggraph.getData(*n)<<", "<<
      //               getData()[src_node]<<"\n";
      ggraph.getData(*n) = getData()[src_node];
    }
    fprintf(stderr, "Writeback from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=%lu].\n",
            gg_num_nodes, gg_num_edges, sizeof(NodeDataType),
            sizeof(EdgeDataType));
  }

  template <typename GaloisGraph>
  void load_from_galois(GaloisGraph& ggraph, int gg_num_nodes, int gg_num_edges,
                        int num_ghosts) {
    typedef typename GaloisGraph::GraphNode GNode;
    //      const size_t gg_num_nodes = ggraph.size();
    //      const size_t gg_num_edges = ggraph.sizeEdges();
    init(gg_num_nodes + num_ghosts, gg_num_edges);
    const int* ptr = (int*)this->gpu_graph->host_ptr();
    fprintf(stderr, "Loading from GaloisGraph [%d,%d,%d].\n", (int)gg_num_nodes,
            (int)gg_num_edges, num_ghosts);
    int edge_counter = 0;
    int node_counter = 0;
    for (auto n = ggraph.begin(); n != ggraph.begin() + gg_num_nodes;
         n++, node_counter++) {
      int src_node               = *n;
      getData()[src_node]        = ggraph.getData(*n);
      outgoing_index()[src_node] = edge_counter;
      for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n);
           ++nbr) {
        GNode dst                     = ggraph.getEdgeDst(*nbr);
        out_neighbors()[edge_counter] = dst;
        out_edge_data()[edge_counter] = ggraph.getEdgeData(*nbr);
        edge_counter++;
      }
    }
    for (; node_counter < gg_num_nodes + num_ghosts; node_counter++) {
      outgoing_index()[node_counter] = edge_counter;
    }
    outgoing_index()[gg_num_nodes] = edge_counter;
    if (node_counter != gg_num_nodes)
      fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %d, %d\n", node_counter,
              gg_num_nodes, num_ghosts);
    assert(edge_counter == gg_num_edges && "Failed to add all edges.");
    fprintf(stderr, "Loaded from GaloisGraph [V=%d,E=%d,ND=%lu,ED=%lu].\n",
            gg_num_nodes, gg_num_edges, sizeof(NodeDataType),
            sizeof(EdgeDataType));
  }
  ~LC_LinearArray_Graph() { deallocate(); }
  void read(const char* filename) { readFromGR(*this, filename); }
  size_t size() { return gpu_graph->size(); }
  NodeDataType* getData() { return (NodeDataType*)(gpu_graph->host_data + 4); }
  NodeDataType& getData(NodeIDType nid) {
    return ((NodeDataType*)(gpu_graph->host_data + 4))[nid];
  }
  unsigned int* outgoing_index() {
    return (unsigned int*)(getData()) + _num_nodes * SizeNodeData;
  }
  unsigned int num_neighbors(NodeIDType node) {
    return outgoing_index()[node + 1] - outgoing_index()[node];
  }
  unsigned int* out_neighbors() {
    return (unsigned int*)outgoing_index() + _num_nodes + 1;
  }
  unsigned int& out_neighbors(NodeIDType node, unsigned int idx) {
    return out_neighbors()[node + idx];
  }

  EdgeDataType* out_edge_data() {
    return (EdgeDataType*)(unsigned int*)(out_neighbors()) + _num_edges;
  }
  GPUType* get_array_ptr(void) { return gpu_graph; }
  size_t num_nodes() { return _num_nodes; }
  size_t num_edges() { return _num_edges; }
  size_t max_degree() { return _max_degree; }
  void init(size_t n_n, size_t n_e) {
    _num_nodes = n_n;
    _num_edges = n_e;
    fprintf(stderr, "Allocating NN: :%d,  , NE %d :\n", (int)_num_nodes,
            (int)_num_edges);
    //      std::cout << "Allocating NN: " << _num_nodes << " , NE :" <<
    //      _num_edges << ". ";
    // Num_nodes, num_edges, [node_data] , [outgoing_index], [out_neighbors],
    // [edge_data]
    const size_t sz = 4 + _num_nodes * SizeNodeData + _num_nodes + 1 +
                      _num_edges + _num_edges * SizeEdgeData;
    gpu_graph = new GPUType(sz);
    memset(gpu_graph->host_ptr(), 0, sizeof(int) * sz);
    gpu_graph->copy_to_device();
    (*gpu_graph)[0] = (int)_num_nodes;
    (*gpu_graph)[1] = (int)_num_edges;
    (*gpu_graph)[2] = (int)SizeNodeData;
    (*gpu_graph)[3] = (int)SizeEdgeData;
    fprintf(stderr, "META :: %zu %d %d %d %d \n", sz, (*gpu_graph)[0],
            (*gpu_graph)[1], (*gpu_graph)[2], (*gpu_graph)[3]);
    // allocate_on_gpu();
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void copy_to_device(void) { gpu_graph->copy_to_device(); }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void copy_to_host(void) { gpu_graph->copy_to_host(); }
  DevicePtrType& device_ptr() { return gpu_graph->device_ptr(); }
  HostPtrType& host_ptr(void) { return gpu_graph->host_ptr(); }
  void print_header(void) {
    std::cout << "Header :: [";
    for (unsigned int i = 0; i < 6; ++i) {
      std::cout << gpu_graph->operator[](i) << ",";
    }
    std::cout << "\n";
    return;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void print_node(unsigned int idx, const char* post = "") {
    if (idx < _num_nodes) {
      std::cout << "N-" << idx << "(" << (getData())[idx] << ")"
                << " :: [";
      for (size_t i = (outgoing_index())[idx]; i < (outgoing_index())[idx + 1];
           ++i) {
        // std::cout << " " << (neighbors())[i] << "(" << (edge_data())[i] <<
        // "), ";
        std::cout << " " << (out_neighbors())[i] << "(" << (out_edge_data())[i]
                  << "<" << getData()[out_neighbors()[i]] << ">"
                  << "), ";
      }
      std::cout << "]" << post;
    }
    return;
  }
  void print_node_nobuff(unsigned int idx, const char* post = "") {
    if (idx < _num_nodes) {
      fprintf(stderr, "N-%d(%d)::[", idx, (getData())[idx]);
      for (size_t i = (outgoing_index())[idx]; i < (outgoing_index())[idx + 1];
           ++i) {
        // std::cout << " " << (neighbors())[i] << "(" << (edge_data())[i] <<
        // "), ";
        fprintf(stderr, "%d ( < %d(%d) > ),  ", (out_neighbors())[i],
                (out_edge_data())[i], getData()[out_neighbors()[i]]);
      }
      fprintf(stderr, "]%s", post);
    }
    return;
  }

  static const char* get_graph_decl(std::string& res) {
    res.append(_str_LC_LinearArray_Graph);
    return _str_LC_LinearArray_Graph;
  }
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void allocate_on_gpu() { return; }
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
  void print_graph_nobuff(void) {
    fprintf(stderr, "\n====Printing graph (%d ,%d )=====\n", (int)_num_nodes,
            (int)_num_edges);
    for (size_t i = 0; i < _num_nodes; ++i) {
      print_node_nobuff(i, "\n");
    }
    return;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  void print_compact(void) {
    std::cout << "\nOut-index [";
    for (size_t i = 0; i < _num_nodes + 1; ++i) {
      std::cout << " " << outgoing_index()[i] << ",";
    }
    std::cout << "]\nNeigh[";
    for (size_t i = 0; i < _num_edges; ++i) {
      std::cout << " " << out_neighbors()[i] << ",";
    }
    std::cout << "]\nEData [";
    for (size_t i = 0; i < _num_edges; ++i) {
      std::cout << " " << out_edge_data()[i] << ",";
    }
    std::cout << "]";
  }

  ////////////##############################################################///////////
  ////////////##############################################################///////////
  void deallocate(void) { delete gpu_graph; }
};
// End LC_Graph
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
} // namespace opencl
} // namespace galois
#include "LC_LinearArray_VoidGraph.h"
#endif /* LC_LinearArray_Graph_H_ */
