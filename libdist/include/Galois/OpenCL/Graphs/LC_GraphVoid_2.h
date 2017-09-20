/*
 * LC_Graph_2.h
 *
 *  Created on: Nov 19, 2015
 *      Author: rashid
 */

#ifndef GDIST_EXP_INCLUDE_OPENCL_LC_GRAPHVOID_2_H_
#define GDIST_EXP_INCLUDE_OPENCL_LC_GRAPHVOID_2_H_

namespace galois {
namespace OpenCL {
namespace Graphs{
static const char * cl_wrapper_str_LC_GraphVoid_2 =
      "\
      typedef struct _GraphType { \n\
   uint _num_nodes;\n\
   uint _num_edges;\n\
    uint _node_data_size;\n\
    uint _edge_data_size;\n\
    __global uint *_node_data;\n\
    __global uint *_out_index;\n\
    __global uint *_out_neighbors;\n\
    }GraphType;\
      ";
static const char * init_kernel_str_LC_GraphVoid_2 =
      "\
      __kernel void initialize_graph_struct(__global uint * res, __global uint * g_meta, __global uint *g_node_data, __global uint * g_out_index, __global uint * g_nbr){ \n \
      __global GraphType * g = (__global GraphType *) res;\n\
      g->_num_nodes = g_meta[0];\n\
      g->_num_edges = g_meta[1];\n\
      g->_node_data_size = g_meta[2];\n\
      g->_edge_data_size= g_meta[3];\n\
      g->_node_data = g_node_data;\n\
      g->_out_index= g_out_index;\n\
      g->_out_neighbors = g_nbr;\n\
      }\n\
      ";

template<template<typename > class GPUWrapper, typename NodeDataTy>
struct LC_Graph_2<GPUWrapper, NodeDataTy, void> {

   //Are you using gcc/4.7+ Error on line below for earlier versions.
#ifdef _WIN32
   typedef GPUWrapper<unsigned int> GPUType;
   typedef typename GPUWrapper<unsigned int>::HostPtrType HostPtrType;
   typedef typename GPUWrapper<unsigned int>::DevicePtrType DevicePtrType;
#else
   template<typename T> using ArrayType = GPUWrapper<T>;
   typedef GPUWrapper<unsigned int> GPUType;
   typedef GPUWrapper<NodeDataTy> GPUNodeType;
   typedef typename GPUWrapper<unsigned int>::HostPtrType HostPtrType;
   typedef typename GPUWrapper<unsigned int>::DevicePtrType DevicePtrType;
#endif
   typedef NodeDataTy NodeDataType;
   typedef unsigned int NodeIDType;
   typedef unsigned int EdgeIDType;
   size_t _num_nodes;
   size_t _num_edges;
   unsigned int _max_degree;
   const size_t SizeEdgeData;
   const size_t SizeNodeData;
   GPUType * gpu_struct_ptr;
   GPUType * gpu_graph;
   GPUNodeType * node_data_ptr;
   GPUType * outgoing_index_ptr;
   GPUType * outgoing_neighbors_ptr;
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
   LC_Graph_2() :
         SizeEdgeData(0), SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
//      fprintf(stderr, "Created LC_LinearArray_Graph with %d node %d edge data.", (int) SizeNodeData, (int) SizeEdgeData);
      _max_degree = _num_nodes = _num_edges = 0;
      gpu_struct_ptr = nullptr;
      gpu_graph = nullptr;
      node_data_ptr = nullptr;
      outgoing_index_ptr = nullptr;
      outgoing_neighbors_ptr = nullptr;
   }
   template<typename GaloisGraph>
   void load_from_galois(GaloisGraph & ggraph) {
      typedef typename GaloisGraph::GraphNode GNode;
      const size_t gg_num_nodes = ggraph.size();
      const size_t gg_num_edges = ggraph.sizeEdges();
      init(gg_num_nodes, gg_num_edges);
      const int * ptr = (int *) this->gpu_graph->host_ptr();
      int edge_counter = 0;
      int node_counter = 0;
      outgoing_index()[0] = 0;
      for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
         int src_node = *n;
         getData()[src_node] = ggraph.getData(*n);
         outgoing_index()[src_node] = edge_counter;
         for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n); ++nbr) {
            GNode dst = ggraph.getEdgeDst(*nbr);
            out_neighbors()[edge_counter] = dst;
//               std::cout<<src_node<<" "<<dst<<" "<<out_edge_data()[edge_counter]<<"\n";
            edge_counter++;
         }

      }
      outgoing_index()[gg_num_nodes] = edge_counter;
//         outgoing_index()[node_counter] = edge_counter;
      fprintf(stderr, "Debug :: %d %d \n", node_counter, edge_counter);
      if (node_counter != gg_num_nodes)
         fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %zu\n", node_counter, gg_num_nodes);
      init_graph_struct();
      assert(edge_counter == gg_num_edges && "Failed to add all edges.");
      fprintf(stderr, "Loaded from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=0].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType));
   }
   template<typename GaloisGraph>
   void writeback_from_galois(GaloisGraph & ggraph) {
      typedef typename GaloisGraph::GraphNode GNode;
      const size_t gg_num_nodes = ggraph.size();
      const size_t gg_num_edges = ggraph.sizeEdges();
      int edge_counter = 0;
      int node_counter = 0;
      outgoing_index()[0] = 0;
      for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
         int src_node = *n;
//               std::cout<<*n<<", "<<ggraph.getData(*n)<<", "<< getData()[src_node]<<"\n";
         ggraph.getData(*n) = getData()[src_node];

      }
      fprintf(stderr, "Writeback from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=0].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType));
   }

   template<typename GaloisGraph>
   void load_from_galois(GaloisGraph & ggraph, int gg_num_nodes, int gg_num_edges, int num_ghosts) {
      typedef typename GaloisGraph::GraphNode GNode;
      //      const size_t gg_num_nodes = ggraph.size();
      //      const size_t gg_num_edges = ggraph.sizeEdges();
      init(gg_num_nodes + num_ghosts, gg_num_edges);
      const int * ptr = (int *) this->gpu_graph->host_ptr();
      fprintf(stderr, "Loading from GaloisGraph [%d,%d,%d].\n", (int) gg_num_nodes, (int) gg_num_edges, num_ghosts);
      int edge_counter = 0;
      int node_counter = 0;
      for (auto n = ggraph.begin(); n != ggraph.begin() + gg_num_nodes; n++, node_counter++) {
         int src_node = *n;
         getData()[src_node] = ggraph.getData(*n);
         outgoing_index()[src_node] = edge_counter;
         for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n); ++nbr) {
            GNode dst = ggraph.getEdgeDst(*nbr);
            out_neighbors()[edge_counter] = dst;
            edge_counter++;
         }
      }
      for (; node_counter < gg_num_nodes + num_ghosts; node_counter++) {
         outgoing_index()[node_counter] = edge_counter;
      }
      outgoing_index()[gg_num_nodes] = edge_counter;
      if (node_counter != gg_num_nodes)
         fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %d, %d\n", node_counter, gg_num_nodes, num_ghosts);
      init_graph_struct();
      assert(edge_counter == gg_num_edges && "Failed to add all edges.");
      fprintf(stderr, "Loaded from GaloisGraph [V=%d,E=%d,ND=%lu,ED=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), 0);
   }
   ~LC_Graph_2() {
      deallocate();
   }
   void read(const char * filename) {
      readFromGR(*this, filename);
   }
   size_t size() {
      return gpu_graph->size();
   }
   NodeDataType * getData() {
      return node_data_ptr->host_data;
   }
   NodeDataType & getData(NodeIDType nid) {
      return getData()[nid];
   }
   unsigned int * outgoing_index() {
      return outgoing_index_ptr->host_data;
   }
   unsigned int num_neighbors(NodeIDType node) {
      return outgoing_index()[node + 1] - outgoing_index()[node];
   }
   unsigned int * out_neighbors() {
      return outgoing_neighbors_ptr->host_data;
   }
   unsigned int & out_neighbors(NodeIDType node, unsigned int idx) {
      return outgoing_neighbors_ptr->host_data[node + idx];
   }
   GPUType * get_array_ptr(void) {
      return gpu_graph;
   }
   size_t num_nodes() {
      return _num_nodes;
   }
   size_t num_edges() {
      return _num_edges;
   }
   size_t max_degree() {
      return _max_degree;
   }
   void init(size_t n_n, size_t n_e) {
      _num_nodes = n_n;
      _num_edges = n_e;
      fprintf(stderr, "Allocating NN: :%d,  , NE %d :\n", (int) _num_nodes, (int) _num_edges);
//      std::cout << "Allocating NN: " << _num_nodes << " , NE :" << _num_edges << ". ";
      //Num_nodes, num_edges, [node_data] , [outgoing_index], [out_neighbors], [edge_data]
//      const size_t sz = 4;
      gpu_struct_ptr = new GPUType(32); // TODO RK make sure the structure size is correct.
      gpu_graph = new GPUType(8);
      node_data_ptr = new GPUNodeType(_num_nodes);
      outgoing_index_ptr = new GPUType(_num_nodes + 1);
      outgoing_neighbors_ptr = new GPUType(_num_edges);
      gpu_graph->copy_to_device();
      (*gpu_graph)[0] = (int) _num_nodes;
      (*gpu_graph)[1] = (int) _num_edges;
      (*gpu_graph)[2] = (int) SizeNodeData;
      (*gpu_graph)[3] = (int) SizeEdgeData;
      fprintf(stderr, "VoidGraph :: META :: %d %d %d %d \n", (*gpu_graph)[0], (*gpu_graph)[1], (*gpu_graph)[2], (*gpu_graph)[3]);
      //allocate_on_gpu();
   }
   void init_graph_struct(){
#if PRE_INIT_STRUCT_ON_DEVICE
      this->copy_to_device();
      CL_Kernel init_kernel;
      size_t kernel_len = strlen(cl_wrapper_str_LC_GraphVoid_2) + strlen(init_kernel_str_LC_GraphVoid_2) + 1;
      char * kernel_src = new char[kernel_len];
      sprintf(kernel_src, "%s\n%s", cl_wrapper_str_LC_GraphVoid_2, init_kernel_str_LC_GraphVoid_2);
      init_kernel.init_string(kernel_src, "initialize_graph_struct");
      init_kernel.set_arg_list(gpu_struct_ptr, gpu_graph, node_data_ptr, outgoing_index_ptr, outgoing_neighbors_ptr);
      init_kernel.run_task();
      gpu_struct_ptr->copy_to_host();
#endif
   }
   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////
   void copy_to_device(void) {
      gpu_graph->copy_to_device();
      gpu_struct_ptr->copy_to_device();
      node_data_ptr->copy_to_device();
      outgoing_index_ptr->copy_to_device();
      outgoing_neighbors_ptr->copy_to_device();
   }
   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////
   void copy_to_host(void) {
      gpu_graph->copy_to_host();
      gpu_struct_ptr->copy_to_host();
      node_data_ptr->copy_to_host();
      outgoing_index_ptr->copy_to_host();
      outgoing_neighbors_ptr->copy_to_host();

   }
   DevicePtrType & device_ptr() {
      return gpu_struct_ptr->device_ptr();
   }
   HostPtrType & host_ptr(void) {
      return gpu_graph->host_ptr();
   }
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
   void print_node(unsigned int idx, const char * post = "") {
      if (idx < _num_nodes) {
         std::cout << "N-" << idx << "(" << (getData())[idx] << ")" << " :: [";
         for (size_t i = (outgoing_index())[idx]; i < (outgoing_index())[idx + 1]; ++i) {
            //std::cout << " " << (neighbors())[i] << "(" << (edge_data())[i] << "), ";
            std::cout << " " << (out_neighbors())[i] << "(" << "<" << getData()[out_neighbors()[i]] << ">" << "), ";
         }
         std::cout << "]" << post;
      }
      return;
   }
   void print_node_nobuff(unsigned int idx, const char * post = "") {
      if (idx < _num_nodes) {
         fprintf(stderr, "N-%d(%d)::[", idx, (getData())[idx]);
         for (size_t i = (outgoing_index())[idx]; i < (outgoing_index())[idx + 1]; ++i) {
            //std::cout << " " << (neighbors())[i] << "(" << (edge_data())[i] << "), ";
            fprintf(stderr, "%d ( < (%d) > ),  ", (out_neighbors())[i], getData()[out_neighbors()[i]]);
         }
         fprintf(stderr, "]%s", post);
      }
      return;
   }

//   static const char * get_graph_decl(std::string &res) {
//      res.append(_str_LC_LinearArray_Graph);
//      return _str_LC_LinearArray_Graph;
//   }
   /////////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////////
   void allocate_on_gpu() {
      return;
   }
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
   void print_graph(void) {
      std::cout << "\n====Printing graph (" << _num_nodes << " , " << _num_edges << ")=====\n";
      for (size_t i = 0; i < _num_nodes; ++i) {
         print_node(i);
         std::cout << "\n";
      }
      return;
   }

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
   void print_graph_nobuff(void) {
      fprintf(stderr, "\n====Printing graph (%d ,%d )=====\n", (int) _num_nodes, (int) _num_edges);
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
      std::cout << "]";
   }

   ////////////##############################################################///////////
   ////////////##############################################################///////////
   void deallocate(void) {
      delete gpu_struct_ptr;
      delete gpu_graph;
      delete node_data_ptr;
      delete outgoing_index_ptr;
      delete outgoing_neighbors_ptr;
      delete gpu_graph;
   }
};
//End LC_Graph_2
}//namespace Graphs
}//Namespace OpenCL
} // Namespace Galois

#endif /* GDIST_EXP_INCLUDE_OPENCL_LC_GRAPH_2_H_ */
