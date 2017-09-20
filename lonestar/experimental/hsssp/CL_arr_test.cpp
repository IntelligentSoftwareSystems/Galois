/** SSSP application -*- C++ -*-
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
 * @author Andrew Lenharth <andrew@lenharth.org>
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/graphs/Util.h"
#include "Lonestar/BoilerPlate.h"
#include "PGraph.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>

#include "CL_Header.h"

#define _HETERO_DEBUG_ 0

static const char* const name_short = "OpenCL Array test";
static const char* const name = "OpenCL API test with arrays";
static const char* const desc = "OpenCL API test with arrays.";
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
   assert(false&& "Invalid personality");
   return "";
}

namespace cll = llvm::cl;
static cll::opt<float> cldevice("cldevice", cll::desc("Select OpenCL device to run on , default is 0.0 (OpenCL backend)"), cll::init(0.0));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));
////////////////////////////////////////////
struct NodeData {
   int dist;/*ID=0*/
};
////////////////////////////////////////////
typedef NodeData NodeDataType;
typedef galois::graphs::LC_CSR_Graph<NodeDataType, unsigned int> Graph;
typedef pGraph<Graph> PGraph;
typedef typename Graph::GraphNode GNode;
bool hasChanged = false;
using namespace galois::opencl;

template<typename ItTy, typename OpType>
void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f);
/********************************************************************************************
 *
 ********************************************************************************************/
/*
 * Initialization routine.
 * */
struct InitializeArray{
   Array<int>* g;
   int counter;
   char * get_kstring ()const { return
         "__kernel void init_kernel(__global int * arr, int num){ const uint my_id = get_global_id(0); if(my_id<num){arr[my_id]=my_id;} }";
   }
   char * get_kname()const{
      return "init_kernel";
   }
   CL_Kernel get_kernel(int num_items)const{
      CL_Kernel k(getCLContext()->get_default_device());
      k.init_string(get_kstring(), get_kname());
      k.set_arg_list(g);
      k.set_arg(1,sizeof(int), &num_items);
      return k;
   }
   void static go(Array<int>& _g, int num) {
      fprintf(stderr, "Calling Initialize with %u\n", num);
      std::vector<int> it(1024);
      int c=0;
      for(auto  i : it)i=c++;
      do_all_cl(it.begin(),it.end(), InitializeArray { &_g,0 });
   }
   void copy_to_device()const{
      fprintf(stderr, "Pre copy to device ::[");
      for(int i=0; i<1024; ++i)
         fprintf(stderr, "%d, ", g->host_ptr()[i]);
      fprintf(stderr, "]\n");
      g->copy_to_device();
   }
   void copy_to_host()const{
      g->copy_to_host();
      fprintf(stderr, "Post copy to host ::[");
      for(int i=0; i<1024; ++i)
         fprintf(stderr, "%d, ", g->host_ptr()[i]);
      fprintf(stderr, "]\n");
   }
   void operator()(GNode src){
      counter++;
      g->host_ptr()[src]=5;
   }
};
/*
 * Verification routine.
 * */
struct VerifyArray{
   Array<int> * g;
   Array<int> * fail_counter;
   char * get_kstring()const{
      return "__kernel void verify_kernel(__global int * arr, __global int * fcounter, int num){ const uint my_id = get_global_id(0); if(my_id<num){atomic_add(fcounter, arr[my_id]!=0);} }";
   }
   char * get_kname()const{
      return "verify_kernel";
   }
   CL_Kernel get_kernel(int num_items)const{
      CL_Kernel k(getCLContext()->get_default_device());
      k.init_string(get_kstring(), get_kname());
      k.set_arg_list(g, fail_counter);
      k.set_arg(2,sizeof(int), &num_items);
      return k;
   }
   void static go(Array<int>& _g, Array<int> & f, unsigned num) {
      fprintf(stderr, "Calling Verify with %u\n", num);
      std::vector<int> it(1024);
      int c=0;
      for(auto  i : it)i=c++;
      do_all_cl(it.begin(), it.end(), VerifyArray { &_g,&f });
   }
   void copy_to_device()const{
      fprintf(stderr, "Pre copy to device :: %d \n", fail_counter->host_ptr()[0]);
      fail_counter->copy_to_device();
   }
   void copy_to_host()const{
      fail_counter->copy_to_host();
      fprintf(stderr, "Post copy to host :: %d \n", fail_counter->host_ptr()[0]);

   }
   ~VerifyArray(){
      fprintf(stderr, "Fail Counter =%d\n", fail_counter->host_ptr()[0]);
   }
   void operator()(GNode src) {
      if(g->host_ptr()[src]!=5)
         fail_counter++;
   }
};

/*
 * Template wrapper for do_all_cl implementation.
 * */
template<typename ItTy, typename OpType>
void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f){
   auto num_items = std::distance(s,e);
   CL_Kernel kernel = f.get_kernel(num_items);//(getCLContext()->get_default_device());
   f.copy_to_device();
   fprintf(stderr, "Launching kernel with %d items \n", num_items );
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
   auto& net = galois::runtime::getSystemNetworkInterface();
   galois::StatManager statManager;

   fprintf(stderr, "Starting OpenCL Test APP \n");
   auto& barrier = galois::runtime::getHostBarrier() ;//getSystemBarrier();
   const unsigned my_host_id = galois::runtime::NetworkInterface::ID;
   auto * d = ctx->get_default_device();
   Array<int> arr(1024,d);
   Array<int> f(1,d);
   f.host_ptr()[0]=0;
   InitializeArray::go(arr,1024);
   //Parse arg string when running on multiple hosts and update/override personality
   //with corresponding value.
   VerifyArray::go(arr,f,1024);
   fprintf(stderr, "Finishing\n");
   std::cout.flush();
   return 0;
}
