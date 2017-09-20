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

#include "Galois/OpenCL/CL_Header.h"

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
typedef galois::graphs::LC_CSR_Graph<unsigned int, unsigned int> Graph;
typedef galois::OpenCL::Graphs::CL_LC_Graph<unsigned int, unsigned int> DevGraph;
typedef typename Graph::GraphNode GNode;
using namespace galois::OpenCL;

//template<typename ItTy, typename OpType, typename ... Args>
//void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f, const Args & ... args);
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
      __kernel void init_kernel(__global GraphType * g, int num){ const uint my_id = get_global_id(0); if(my_id<g->_num_nodes){g->_node_data[my_id]=1024*1024;} }";
      return g_struct;
   }
   const char * get_kname() const {
      return "init_kernel";
   }
public:
   CL_Kernel * get_kernel(int num_items) const {
      static CL_Kernel k(getCLContext()->get_default_device(),get_kstring(), get_kname(), true);
      k.set_arg(0, sizeof(cl_mem), &g->device_ptr());
      k.set_arg(1, sizeof(int), &num_items);
      return &k;
   }
   void static go(DevGraph& _g, int num) {
      fprintf(stderr, "Calling InitializeGraph with %u\n", num);
      do_all_cl(_g.begin(), _g.end(), InitializeGraph { &_g });
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
   CL_Kernel * get_kernel(int num_items) const {
      static CL_Kernel k(getCLContext()->get_default_device(),get_kstring(), get_kname(), true);
      k.set_arg_list(g, term);
      k.set_arg(2, sizeof(int), &num_items);
      return &k;
   }
   void static go(DevGraph& _g, Array<int> & term, int num) {
      do_all_cl(_g.begin(), _g.end(), SSSPKernel { &_g, &term });
   }
   void operator()(GNode src) {
//      g->getData(src)=5;
   }
};

/*
 * Template wrapper for do_all_cl implementation.
 *
template<typename ItTy, typename OpType, typename ... Args>
void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f, const Args & ... args) {
   auto num_items = std::distance(s, e);
   CL_Kernel * kernel = f.get_kernel(num_items); //(getCLContext()->get_default_device());
   kernel->set_work_size(num_items);
   (*kernel)();
}*/
/*
 * Main. Create an array, initialize it and verify it.
 * */
int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   CLContext * ctx = getCLContext();
   auto& net = galois::runtime::getSystemNetworkInterface();
   galois::StatManager statManager;
   auto& barrier = galois::runtime::getHostBarrier();
   const unsigned my_host_id = galois::runtime::NetworkInterface::ID;
   fprintf(stderr, "Starting OpenCL Test APP \n");
   Graph g;
   galois::graphs::readGraph(g, inputFile);
   DevGraph cl_graph;
   cl_graph.load_from_galois(g);
   InitializeGraph::go(cl_graph, cl_graph.size());

   cl_graph.getDataP(0)=0;
   cl_graph.sync_outstanding_data();
   Array<int> term(1, ctx->get_default_device());
   do {
      term.host_ptr()[0]=0;
      term.copy_to_device();
      SSSPKernel::go(cl_graph, term, cl_graph.size());
      term.copy_to_host();
   } while (term.host_ptr()[0] != 0);
   fprintf(stderr, "Finishing\n");
   {
     cl_graph.copy_to_host(galois::OpenCL::Graphs::GRAPH_FIELD_FLAGS::NODE_DATA);
      fprintf(stderr, "Post operator ::[");
      for (auto i = cl_graph.begin(); i != cl_graph.end(); ++i)
         fprintf(stderr, "%d, ", cl_graph.getData(i));
      fprintf(stderr, "]\n");
   }
   std::cout.flush();
   return 0;
}
