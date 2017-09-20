/** Page rank application -*- C++ -*-
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
 * @author Gurbinder Gill <gill@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/graphs/FileGraph.h"
#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/graphs/Util.h"
#include "Lonestar/BoilerPlate.h"

#include "PGraph.h"
#include "hpr.h"

#include <iostream>
#include <fstream>
#include <typeinfo>
#include <algorithm>
#include <atomic>

#define _HETERO_DEBUG_ 0

static const char* const name = "Page Rank - Distributed Heterogeneous";
static const char* const desc = "Computes PageRank on Distributed Galois.  Uses pull algorithm, takes the pre-transposed graph.";
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
static cll::opt<Personality> personality("personality", cll::desc("Personality"),
      cll::values(clEnumValN(CPU, "cpu", "Galois CPU"), clEnumValN(GPU_CUDA, "gpu/cuda", "GPU/CUDA"), clEnumValN(GPU_OPENCL, "gpu/opencl", "GPU/OpenCL"), clEnumValEnd),
      cll::init(CPU));
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file (transpose)>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(4));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));
static cll::opt<int> gpudevice("gpu", cll::desc("Select GPU to run on, default is to choose automatically"), cll::init(-1));
static cll::opt<float> cldevice("cldevice", cll::desc("Select OpenCL device to run on , default is 0.0 (OpenCL backend)"), cll::init(0.0));
static cll::opt<std::string> personality_set("pset", cll::desc("String specifying personality for each host. 'c'=CPU,'g'=GPU/CUDA and 'o'=GPU/OpenCL"), cll::init(""));


//typedef double PRTy;
PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta){
  PRTy old;
  do {
    old = v;
  }while(!v.compare_exchange_strong(old, old + delta));

  return old;
}

PRTy atomicSub(std::atomic<PRTy>& v, PRTy delta){
  PRTy old;
  do {
    old = v;
  }while(!v.compare_exchange_strong(old, old - delta));

  return old;
}


struct LNode {
  PRTy value;
  std::atomic<PRTy> residual;
  //PRTy residual;
  unsigned int nout;

  LNode(){
    value = 1.0 - alpha;
    residual = 0.0;
    nout = 0.0;
  }
  LNode(const LNode& rhs){
    value = rhs.value;
    nout = rhs.nout;
    residual = rhs.residual.load();
  }

  LNode& operator=(const LNode& rhs){
    value = rhs.value;
    nout = rhs.nout;
    residual = rhs.residual.load();
    return *this;
  }
  //void init(){value = 1.0 - alpha; residual = 0.0; nout = 0;}
  PRTy getPagerank(int x = 0){ return value;}
};

//////////////////////////////////////////////////////////////////////////////////////
typedef galois::graphs::LC_CSR_Graph<LNode, void> Graph;
typedef typename Graph::GraphNode GNode;
//////////////////////////////////////////////////////////////////////////////////////

//typedef galois::opencl::LC_LinearArray_Graph<galois::opencl::Array, LNode, void> DeviceGraph;

//struct CUDA_Context *cuda_ctx;
//struct OPENCL_Context<DeviceGraph> cl_ctx;

/*********************************************************************************
 *
 **********************************************************************************/
struct InitializeGraph {
   pGraph<Graph>* g;
   void static go(pGraph<Graph>& _g) {
      galois::do_all(_g.g.begin(), _g.g.begin() + _g.numOwned, InitializeGraph { &_g }, galois::loopname("init"));
   }
   void operator()(GNode src) const {
      LNode& sdata = g->g.getData(src);
      sdata.value = 1.0 - alpha;
      sdata.nout = std::distance(g->g.edge_begin(src), g->g.edge_end(src));
#if _HETERO_DEBUG_
      if(galois::runtime::NetworkInterface::ID == 0)
        std::cout << "Src : " << src << " nout : " <<sdata.nout << "\n"; 
#endif
      //Initializing Residual is like running one round of pagerank
      if(sdata.nout > 0) {
        PRTy delta = sdata.value*alpha/sdata.nout;
        for(auto nbr = g->g.edge_begin(src); nbr != g->g.edge_end(src); ++nbr){
          GNode dst = g->g.getEdgeDst(*nbr);
          LNode& ddata = g->g.getData(dst);
          atomicAdd(ddata.residual, delta);
        }
      }
   }
};

struct InitializeGhostCells {
  pGraph<Graph>* g;
  void static go(pGraph<Graph>& _g){
      galois::do_all(_g.g.begin() + _g.numOwned, _g.g.begin() + _g.numNodes, InitializeGhostCells { &_g }, galois::loopname("init ghost cells"));
  }
  void operator()(GNode src) const {
    //Residual for ghost cells should start from zero.
#if _HETERO_DEBUG_
      if (galois::runtime::NetworkInterface::ID == 0)
        std::cout << " Ghost : " << src << "\n";
#endif
      LNode& sdata = g->g.getData(src);
      sdata.residual = 0.0;
  }
};

 /*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct PageRank_push {
   pGraph<Graph>* g;

   void static go(pGraph<Graph>& _g) {
      galois::do_all(_g.g.begin(), _g.g.begin() + _g.numOwned, PageRank_push { &_g }, galois::loopname("Page Rank (push)"));
   }

   void operator()(GNode src) const {
      LNode& sdata = g->g.getData(src);
      PRTy oldResidual = sdata.residual.exchange(0.0);
      sdata.value += oldResidual;
      PRTy delta = oldResidual*alpha/sdata.nout;
      for (auto jj = g->g.edge_begin(src), ej = g->g.edge_end(src); jj != ej; ++jj) {
         GNode dst = g->g.getEdgeDst(jj);
         LNode& ddata = g->g.getData(dst);
         PRTy old = atomicAdd(ddata.residual, delta);
      }
   }
};

/*********************************************************************************
 *
 **********************************************************************************/

// [hostid] -> vector of GID that host has replicas of
std::vector<std::vector<unsigned> > remoteReplicas;
// [hostid] -> vector of LID that host has replicas of
std::vector<std::vector<unsigned> > remoteReplicas_L;
pGraph<Graph>* g_Local;
// [hostid] -> remote pGraph Structure (locally invalid)
std::vector<pGraph<Graph>*> magicPointer;
//[hostID] -> vector of replica residual
std::vector<std::vector<PRTy>> remoteReplicas_residual;
/*********************************************************************************
 *
 **********************************************************************************/

void setRemotePtr(uint32_t hostID, pGraph<Graph>* p) {
   if (hostID >= magicPointer.size())
      magicPointer.resize(hostID + 1);
   magicPointer[hostID] = p;
}
/*********************************************************************************
 *
 **********************************************************************************/

void recvNodeStatic(unsigned GID, uint32_t hostID) {
   if (hostID >= remoteReplicas.size()){
      remoteReplicas.resize(hostID + 1);
      remoteReplicas_L.resize(hostID + 1);
   }
   remoteReplicas[hostID].push_back(GID);
   remoteReplicas_L[hostID].push_back(g_Local->G2L_Local(GID));
}

/*********************************************************************************
 * Node value reduction.
 **********************************************************************************/
void reduceNodeValue(pGraph<Graph>* p, unsigned GID, PRTy residual) {
   switch (personality) {
   case CPU:
      atomicAdd(p->g.getData(p->G2L(GID)).residual, residual);
      break;
   default:
      break;
   }
}

/*********************************************************************************
 *Reduce all the values a ghost cells
 **********************************************************************************/
struct reduceGhostCells_struct {
  pGraph<Graph>* g;

  void static go(pGraph<Graph>& _g){
      galois::do_all(_g.g.begin() + _g.numOwned, _g.g.begin() + _g.numNodes, reduceGhostCells_struct { &_g }, galois::loopname("ReduceGhost Cells"));
  }
  void operator()(GNode src) const {
     galois::runtime::NetworkInterface& net = galois::runtime::getSystemNetworkInterface();

     LNode& sdata = g->g.getData(src);

     auto n = g->uid(src);
     auto x = g->getHost(n);

     if (x >= remoteReplicas_residual.size())
       remoteReplicas_residual.resize(x+1);

     remoteReplicas_residual[x].push_back(sdata.residual.exchange(0.0));
  }
};


struct applyResidual_struct {
  std::vector<unsigned>* RR_vec;
  std::vector<PRTy>* RR_residual_vec;
  pGraph<Graph>* g;

  void static go(std::vector<unsigned>& _remoteReplicas_vec, std::vector<PRTy>& _remoteReplicas_residual_vec, pGraph<Graph>& _g){
    //galois::do_all(_remoteReplicas_vec.begin(), _remoteReplicas_vec.end(), applyResidual_struct { &_remoteReplicas_vec, &_remoteReplicas_residual_vec, &_g }, galois::loopname("Apply residual struct"));
    galois::do_all(boost::counting_iterator<unsigned>(0), boost::counting_iterator<unsigned>(_remoteReplicas_vec.size()) , applyResidual_struct { &_remoteReplicas_vec, &_remoteReplicas_residual_vec, &_g }, galois::loopname("Apply residual struct"));
  }

  void operator()(unsigned i) const {
    atomicAdd(g->g.getData((*RR_vec)[i]).residual, (*RR_residual_vec)[i]);
  }

};
void receiveGhostCellVectors(galois::runtime::RecvBuffer& buff){
  std::vector<PRTy> residual_vec;
  unsigned fromHostID;
  gDeserialize(buff, fromHostID, residual_vec);
  auto& net = galois::runtime::getSystemNetworkInterface();
  pGraph<Graph>* p = magicPointer[net.ID];

#if _HETERO_DEBUG_
  std::cout << "RECEIVED from " << fromHostID << " ON : " << net.ID << "\n";
  if(galois::runtime::NetworkInterface::ID == 1)
    for(auto x : residual_vec)
      std::cout << x << "\n";
#endif

  if(residual_vec.size() > 0)
    applyResidual_struct::go(remoteReplicas_L[fromHostID], residual_vec, *p);

  /**** For sequential apply ********/
  /*
  unsigned i = 0;
  for(auto GID = remoteReplicas[fromHostID].begin(); GID != remoteReplicas[fromHostID].end(); ++GID, ++i){
    //if(net.ID == 1)
      //std::cout << *GID << " " << residual_vec[i] << "\n";
    atomicAdd(p->g.getData(p->G2L(*GID)).residual, residual_vec[i]);
  }
  */
}

void sendGhostCellVectors(galois::runtime::NetworkInterface& net, pGraph<Graph>& g) {

  unsigned remoteHostID = 0;
  for(auto x = remoteReplicas_residual.begin(); x != remoteReplicas_residual.end(); ++x, ++remoteHostID){
    if(remoteHostID == net.ID)
      continue;

    galois::runtime::SendBuffer buff;
    gSerialize(buff, net.ID, *x);
    net.send(remoteHostID, receiveGhostCellVectors, buff);
  }
  //std::cout << " SENT from " << net.ID << "\n";
}

void reduceGhostCells(galois::runtime::NetworkInterface& net, pGraph<Graph>& g) {

  for(auto ii = g.g.begin() + g.numOwned; ii != g.g.begin() + g.numNodes; ++ii){
    auto n = g.uid(*ii);
    auto x = g.getHost(n);
    LNode& sdata = g.g.getData(*ii);

#if _HETERO_DEBUG_
    if (net.ID == 0)
      std::cout << "src : " << *ii << " , n " << n << ", x "<< x << "\n";
#endif
    switch (personality) {
      case CPU:
        net.sendAlt(x, reduceNodeValue, magicPointer[x], n, sdata.residual.exchange(0.0));
        break;
      default:
        assert(false);
        break;
    }
  }
}

/*********************************************************************************
 *Make Ghost cells residual zero
 **********************************************************************************/
struct clearGhostCells {
  pGraph<Graph>* g;

  void static go(pGraph<Graph>& _g){
    galois::do_all(_g.g.begin() + _g.numOwned, _g.g.begin() + _g.numNodes, InitializeGhostCells { &_g }, galois::loopname("Clear residual on Ghost cells"));
  }

  void operator()(GNode src){
    PRTy oldResidual = g->g.getData(src).residual.exchange(0.0);
  }
};
/*********************************************************************************
 *
 **********************************************************************************/
void inner_main() {
   auto& net = galois::runtime::getSystemNetworkInterface();
   galois::StatManager statManager;
   auto& barrier = galois::runtime::getBarrier(galois::getActiveThreads());
   const unsigned my_host_id = galois::runtime::NetworkInterface::ID;
   galois::Timer T_total, T_graph_load, T_pagerank, T_pagerank_perIter, T_graph_init;
   T_total.start();
   //Parse arg string when running on multiple hosts and update/override personality
   //with corresponding value.
   if (personality_set.length() == galois::runtime::NetworkInterface::Num) {
      switch (personality_set.c_str()[galois::runtime::NetworkInterface::ID]) {
      case 'g':
         personality = GPU_CUDA;
         break;
      case 'o':
         personality = GPU_OPENCL;
         break;
      case 'c':
      default:
         personality = CPU;
         break;
      }
   }
   fprintf(stderr, "Pre-barrier - Host: %d, Personality %s\n",galois::runtime::NetworkInterface::ID ,personality_str(personality).c_str());
   barrier.wait();
   fprintf(stderr, "Post-barrier - Host: %d, Personality %s\n",galois::runtime::NetworkInterface::ID ,personality_str(personality).c_str());
   T_graph_load.start();
   pGraph<Graph> g;
   g_Local = &g;
   g.loadGraph(inputFile);

#if _HETERO_DEBUG_
   std::cout << g.id << " graph loaded\n";
#endif

   T_graph_load.stop();

   T_graph_init.start();

   //send pGraph pointers
   for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, setRemotePtr, galois::runtime::NetworkInterface::ID, &g);

   //Ask for cells
   for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, galois::runtime::NetworkInterface::ID);

#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for remote replicas\n";
#endif
   barrier.wait();

   //local initialization
   if (personality == CPU) {
     InitializeGraph::go(g);
     // Propagate the residual collected in initialization phase.
     reduceGhostCells_struct::go(g);
     sendGhostCellVectors(net,g);
     for(auto x = remoteReplicas_residual.begin(); x != remoteReplicas_residual.end(); ++x){
       (*x).clear();
     }
   }

#if _HETERO_DEBUG_
   std::cout << g.id << " initialized\n";
#endif
   barrier.wait();
   T_graph_init.stop();

   std::cout << "[" << my_host_id << "] Starting PageRank" << "\n";
   T_pagerank.start();
   for (int i = 0; i < maxIterations; ++i) {
      //Do pagerank
      switch (personality) {
      case CPU:
        PageRank_push::go(g);
        reduceGhostCells_struct::go(g);
        sendGhostCellVectors(net,g);
        for(auto x = remoteReplicas_residual.begin(); x != remoteReplicas_residual.end(); ++x){
          (*x).clear();
        }
#if _HETERO_DEBUG_
        if(my_host_id == 1)
        {
           for(auto x = remoteReplicas.begin(); x != remoteReplicas.end(); ++x)
            for(auto y = x->begin(); y != x->end(); ++y)
              std::cout << "remote on 1 - > " << *y << " : " << g.G2L_Local(*y) << "\n";

           for(auto x = remoteReplicas_L.begin(); x != remoteReplicas_L.end(); ++x)
            for(auto y = x->begin(); y != x->end(); ++y)
              std::cout << "remote LID on 1 - > " << *y << "\n";

        }
        if(my_host_id == 0)
        {
          for(auto x = remoteReplicas_residual.begin(); x != remoteReplicas_residual.end(); ++x)
            for(auto y = x->begin(); y != x->end(); ++y)
              std::cout << "R for 1 - >"<< *y << "\n";

          for(auto  x : g.L2G){
            std::cout << "L2G - > " << x << "\n";
          }

          int i = 0;
          for (auto n = g.g.begin() + g.numOwned; n != g.g.begin() + g.numNodes; ++n, ++i)
          {
            LNode& sdata = g.g.getData(*n);

            std::cout << *n << " R : " << sdata.residual <<" L2G vec : " << g.L2G[*n - g.numOwned] <<" From vec : " << sdata.residual << " \n";
          }
        }
#endif
        break;
      default:
         break;
      }

      barrier.wait();
   }

   T_pagerank.stop();

   if (verify) {
      std::stringstream ss;
      ss << personality_str(personality) << "_" << my_host_id << "_of_" << galois::runtime::NetworkInterface::Num << "_page_ranks.csv";
      std::ofstream out_file(ss.str());
      switch (personality) {
      case CPU: {
         for (auto n = g.g.begin(); n != g.g.begin() + g.numOwned; ++n) {
            out_file << *n + g.g_offset << ", " << g.g.getData(*n).value<< ", " << g.g.getData(*n).nout << "\n";
         }
         break;
      }
      }
      out_file.close();
   }

   T_total.stop();
   std::cout << "[" << galois::runtime::NetworkInterface::ID << "]" << " Total : " << T_total.get() << " Loading : " << T_graph_load.get() << " Init : " << T_graph_init.get() << " PageRank (" << maxIterations << " iteration) : " << T_pagerank.get() << " (msec)\n";

   std::cout << "Terminated on [ " << my_host_id << " ]\n";
   std::cout.flush();

}

int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   //auto& net = galois::runtime::getSystemNetworkInterface();
   inner_main();
   return 0;
}
