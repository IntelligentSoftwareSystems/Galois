/** SSSP application using Galois-CL API -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */

#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Graphs/LC_CSR_Graph.h"
#include "Galois/Graphs/Util.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>

#include "CL_Header.h"

#define _HETERO_DEBUG_ 0

static const char* const name_short = "OpenCL Graph test";
static const char* const name = "OpenCL API test with Graph";
static const char* const desc = "OpenCL API test with Graph.";
static const char* const url = 0;

enum Personality {
   CPU, GPU_CUDA, GPU_OPENCL
};
std::string personality_str(Personality p) {
   switch (p) {
   case CPU:
      return "CPU";
   case GPU_CUDA:
      return "GPU_CUDA";
   case GPU_OPENCL:
      return "GPU_OPENCL";
   }
   assert(false && "Invalid personality");
   return "";
}

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file (transpose)>"), cll::Required);
static cll::opt<float> cldevice("cldevice", cll::desc("Select OpenCL device to run on , default is 0.0 (OpenCL backend)"), cll::init(0.0));
////////////////////////////////////////////
typedef Galois::Graph::LC_CSR_Graph<unsigned int, unsigned int> Graph;
typedef Galois::OpenCL::Graphs::CL_LC_Graph<unsigned int, unsigned int> DevGraph;
typedef typename Graph::GraphNode GNode;
using namespace Galois::OpenCL;

template<typename ItTy, typename OpType>
void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f);
/********************************************************************************************
 *
 ********************************************************************************************/
/*
 * Initialization routine.
 * */
struct InitializeGraph {
   DevGraph* g;
private:
   const char * get_kstring() const {
      const char * g_struct =
            "\
      typedef struct _GraphType { \n\
   uint _num_nodes;\n\
   uint _num_edges;\n\
    uint _node_data_size;\n\
    uint _edge_data_size;\n\
    __global uint *_node_data;\n\
    __global uint *_out_index;\n\
    __global uint *_out_neighbors;\n\
    __global uint *_out_edge_data;\n\
    }GraphType;\n\
      __kernel void init_kernel(__global GraphType * g, int num){ const uint my_id = get_global_id(0); if(my_id<g->_num_nodes){g->_node_data[my_id]=my_id==0?0:1024*1024;} }";
      return g_struct;
   }
   const char * get_kname() const {
      return "init_kernel";
   }
public:
   CL_Kernel get_kernel(int num_items) const {
      CL_Kernel k(getCLContext()->get_default_device());
      k.init_string(get_kstring(), get_kname());
//      k.set_arg_list(g);
      k.set_arg(0, sizeof(cl_mem), &g->device_ptr());
      k.set_arg(1, sizeof(int), &num_items);
      return k;
   }
   void static go(DevGraph& _g, int num) {
      fprintf(stderr, "Calling InitializeGraph with %u\n", num);
      do_all_cl(_g.begin(), _g.end(), InitializeGraph { &_g });
   }
   void copy_to_device() const {
      fprintf(stderr, "Pre copy to device ::[");
      for (auto i = g->begin(); i != g->end(); ++i)
         fprintf(stderr, "%d, ", g->getData(i));
      fprintf(stderr, "]\n");
//      g->copy_to_device();
   }
   void copy_to_host() const {
      g->copy_to_host(Galois::OpenCL::Graphs::GRAPH_FIELD_FLAGS::NODE_DATA);
      fprintf(stderr, "Post copy to host ::[");
      for (auto i = g->begin(); i != g->end(); ++i)
         fprintf(stderr, "%d, ", g->getData(i));
      fprintf(stderr, "]\n");
   }
   void operator()(GNode src) {
      g->getData(src) = 5;
   }
};

/********************************************************************************************
 *
 ********************************************************************************************/
/*
 * Kernel routine.
 * */
struct SSSPKernel {
   DevGraph* g;
   Array<int> * term;
private:
   const char * get_kstring() const {
      const char * g_struct =
            "\
      typedef struct _GraphType { \n\
   uint _num_nodes;\n\
   uint _num_edges;\n\
    uint _node_data_size;\n\
    uint _edge_data_size;\n\
    __global uint *_node_data;\n\
    __global uint *_out_index;\n\
    __global uint *_out_neighbors;\n\
    __global uint *_out_edge_data;\n\
    }GraphType;\n\
__kernel void sssp_kernel(__global GraphType * g, __global int * term,  int num){const uint my_id = get_global_id(0); if(my_id<g->_num_nodes){\ for(int idx= g->_out_index[my_id]; idx<g->_out_index[my_id+1]; ++idx){ \ if (g->_node_data[g->_out_neighbors[idx]]>g->_node_data[my_id]+g->_out_edge_data[idx]){\ *term=1;atomic_min(&g->_node_data[g->_out_neighbors[idx]],g->_node_data[my_id]+g->_out_edge_data[idx]);}\ /*End if*/}/*End for*/\} }";
      return g_struct;
   }
   const char * get_kname() const {
      return "sssp_kernel";
   }
public:
   CL_Kernel get_kernel(int num_items) const {
      CL_Kernel k(getCLContext()->get_default_device());
      k.init_string(get_kstring(), get_kname());
      k.set_arg_list(g, term);
      k.set_arg(2, sizeof(int), &num_items);
      return k;
   }
   void static go(DevGraph& _g, Array<int> & term, int num) {
//      term.host_ptr()[0] = 0;
//      fprintf(stderr, "Calling SSSP with %u\n", num);
//      for(int i=0; i<100; i+=1)
      do_all_cl(_g.begin(), _g.end(), SSSPKernel { &_g, &term });
   }
   void copy_to_device() const {
//      fprintf(stderr, "Pre copy to device ::[");
      /*for(auto i = g->begin(); i!=g->end(); ++i)
       fprintf(stderr, "%d, ", g->getData(i));
       fprintf(stderr, "]\n");*/
      term->copy_to_device();
//      g->copy_to_device();
   }
   void copy_to_host() const {
      term->copy_to_host();
//      g->copy_to_host(Galois::OpenCL::Graphs::GRAPH_FIELD_FLAGS::NODE_DATA);
      /*      fprintf(stderr, "Post copy to host ::[");
       for(auto i = g->begin(); i!=g->end(); ++i)
       fprintf(stderr, "%d, ", g->getData(i));
       fprintf(stderr, "]\n");*/
   }
   void operator()(GNode src) {
//      g->getData(src)=5;
   }
};
/*
 * Verification routine.
 * */
struct VerifyArray {
   Array<int> * g;
   Array<int> * fail_counter;
   char * get_kstring() const {
      return "__kernel void verify_kernel(__global int * arr, __global int * fcounter, int num){ const uint my_id = get_global_id(0); if(my_id<num){atomic_add(fcounter, arr[my_id]!=0);} }";
   }
   char * get_kname() const {
      return "verify_kernel";
   }
   CL_Kernel get_kernel(int num_items) const {
      CL_Kernel k(getCLContext()->get_default_device());
      k.init_string(get_kstring(), get_kname());
      k.set_arg_list(g, fail_counter);
      k.set_arg(2, sizeof(int), &num_items);
      return k;
   }
   void static go(Array<int>& _g, Array<int> & f, unsigned num) {
      fprintf(stderr, "Calling Verify with %u\n", num);
      std::vector<int> it(1024);
      int c = 0;
      for (auto i : it)
         i = c++;
      do_all_cl(it.begin(), it.end(), VerifyArray { &_g, &f });
   }
   void copy_to_device() const {
      fprintf(stderr, "Pre copy to device :: %d \n", fail_counter->host_ptr()[0]);
      fail_counter->copy_to_device();
   }
   void copy_to_host() const {
      fail_counter->copy_to_host();
      fprintf(stderr, "Post copy to host :: %d \n", fail_counter->host_ptr()[0]);

   }
   ~VerifyArray() {
      fprintf(stderr, "Fail Counter =%d\n", fail_counter->host_ptr()[0]);
   }
   void operator()(GNode src) {
      if (g->host_ptr()[src] != 5)
         fail_counter++;
   }
};

/*
 * Template wrapper for do_all_cl implementation.
 * */
template<typename ItTy, typename OpType>
void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f) {
   auto num_items = std::distance(s, e);
   CL_Kernel kernel = f.get_kernel(num_items); //(getCLContext()->get_default_device());
   f.copy_to_device();
   fprintf(stderr, "Launching kernel with %d items \n", num_items);
   kernel.set_work_size(num_items);
   kernel();
   f.copy_to_host();
}
/*
 * Main. Create an array, initialize it and verify it.
 * */
int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   CLContext * ctx = getCLContext();
   auto& net = Galois::Runtime::getSystemNetworkInterface();
   Galois::StatManager statManager;
   auto& barrier = Galois::Runtime::getHostBarrier();
   const unsigned my_host_id = Galois::Runtime::NetworkInterface::ID;
   fprintf(stderr, "Starting OpenCL Test APP \n");
   Graph g;
   Galois::Graph::readGraph(g, inputFile);
   DevGraph cl_graph;
   cl_graph.load_from_galois(g);
   InitializeGraph::go(cl_graph, cl_graph.size());
   Array<int> term(1, ctx->get_default_device());
   do {
      term.host_ptr()[0]=0;
      SSSPKernel::go(cl_graph, term, cl_graph.size());
   } while (term.host_ptr()[0] != 0);
   cl_graph.copy_to_host(Galois::OpenCL::Graphs::GRAPH_FIELD_FLAGS::NODE_DATA);
   for (auto i = cl_graph.begin(); i != cl_graph.end(); ++i)
      fprintf(stderr, "%d, ", cl_graph.getData(i));
   fprintf(stderr, "]\n");
   //Parse arg string when running on multiple hosts and update/override personality
   //with corresponding value.
   fprintf(stderr, "Finishing\n");
   std::cout.flush();
   return 0;
}
