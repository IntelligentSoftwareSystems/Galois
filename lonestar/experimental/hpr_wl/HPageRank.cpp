#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Graph/FileGraph.h"
#include "galois/Graph/LC_CSR_Graph.h"
#include "galois/Graph/Util.h"
#include "Lonestar/BoilerPlate.h"
#include "PGraph.h"

#include "cuda/cuda_mtypes.h"
#include "hpr.h"

#include "opencl/OpenCLPrBackend.h"

#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <queue>
#include <list>
#include <time.h>
#include <chrono>
#include "cuda/hpr_wl_cuda.h"

#define _HETERO_DEBUG_ 1
#define _MICRO_CONV_ 1000000.0 //to convert microsec to sec

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
   assert(false && "Invalid personality");
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

struct LNode {
   float value;
   unsigned int nout;
};

//to control the work
struct WorkManager {
   int active_workers;bool first_time;
   int my_amount_of_work;
   int barrier_counter; // to test a different barrier
};

typedef galois::graphs::LC_CSR_Graph<LNode, void> Graph;
typedef pGraph<Graph> PGraph;
typedef typename Graph::GraphNode GNode;
std::map<GNode, float> buffered_updates;

WorkManager MWork;

//////////////////////////////////////////////////////////////////////////////////////
typedef galois::opencl::LC_LinearArray_Graph<galois::opencl::Array, LNode, void> DeviceGraph;

struct CUDA_Context *cuda_ctx;
struct OPENCL_Context<DeviceGraph> cl_ctx;

/*********************************************************************************
 *
 **********************************************************************************/
struct InitializeGraph {
   Graph* g;

   void static go(Graph& _g, unsigned num) {

      galois::do_all(_g.begin(), _g.begin() + num, InitializeGraph { &_g }, galois::loopname("init"));
   }

   void operator()(GNode src) const {
      LNode& sdata = g->getData(src);
      //sdata.value = (alpha / g->size()) + ((1.0 - alpha)/g->size());
      sdata.value = 1.0 - alpha;
      buffered_updates[src] = 0;
      for (auto nbr = g->edge_begin(src); nbr != g->edge_end(src); ++nbr) {
         __sync_fetch_and_add(&g->getData(g->getEdgeDst(*nbr)).nout, 1);
      }
   }
};

/*********************************************************************************
 * CPU PageRank operator implementation.
 **********************************************************************************/
struct WriteBack {
   Graph * g;

   void static go(Graph& _g, unsigned num) {
      galois::do_all(_g.begin(), _g.begin() + num, WriteBack { &_g }, galois::loopname("Writeback"));
   }

   void operator()(GNode src) const {
      LNode& sdata = g->getData(src);
      sdata.value = buffered_updates[src];
      buffered_updates[src] = 0;
   }
};

struct PageRank {
   PGraph* g;

   //void static go(Graph& _g, unsigned num) {
   void static go(PGraph& pg) {
      pg.work[(pg.active_bag + 1) % 2].clear();
      galois::do_all(pg.work[pg.active_bag % 2].begin(), pg.work[pg.active_bag % 2].end(), PageRank { &pg }, galois::loopname("PageRank-WL"));
      pg.work[pg.active_bag % 2].clear();
      pg.active_bag++;
   }
   void operator()(GNode & node) {
      LNode& sdata = g->g.getData(node);
      double sum = 0;
      for (auto jj = g->g.edge_begin(node); jj != g->g.edge_end(node); ++jj) {
         GNode dst = g->g.getEdgeDst(jj);
         LNode& ddata = g->g.getData(dst);
         sum += ddata.value / ddata.nout;
      }
      double old_value = sdata.value;
      //sdata.value = (1.0 - alpha) * sum + (alpha / pg.g.size());
      sdata.value = (1.0 - alpha) * sum + alpha;
      //update the value right way, instead of using auxiliary vector. Need to check if this has some kind of strange effect

      double diff = std::fabs(sdata.value - old_value);
      //std::cout << "node: " << node << " score " << old_value << " changed to: " << sdata.value << " diff: " << diff << " tolerance " << ERROR_THRESHOLD <<"\n";
      if (diff > ERROR_THRESHOLD) { //&& (std::find(pg.myFutureWork.begin(), pg.myFutureWork.end(), node) != node)) {
         //if strategy is changed that you may have to take into account if a node is already on the second list
         g->work[(g->active_bag + 1) % 2].push(node);
      }
   }
#if 0
   void static old_go(PGraph& pg) {
      //std::cout << pg.id << " List has " << pg.myWork.size() << " elements\n";
      while (!pg.myWork.empty()) { //process all nodes on the list
         unsigned node = pg.myWork.front();//get the first element
         pg.myWork.pop_front();//remove element
         LNode& sdata = pg.g.getData(node);
         double sum = 0;
         for (auto jj = pg.g.edge_begin(node); jj != pg.g.edge_end(node); ++jj) {
            GNode dst = pg.g.getEdgeDst(jj);
            LNode& ddata = pg.g.getData(dst);
            sum += ddata.value / ddata.nout;
         }
         double old_value = sdata.value;
         //sdata.value = (1.0 - alpha) * sum + (alpha / pg.g.size());
         sdata.value = (1.0 - alpha) * sum + alpha;
         //update the value right way, instead of using auxiliary vector. Need to check if this has some kind of strange effect

         double diff = std::fabs(sdata.value - old_value);
         //std::cout << "node: " << node << " score " << old_value << " changed to: " << sdata.value << " diff: " << diff << " tolerance " << ERROR_THRESHOLD <<"\n";
         if (diff > ERROR_THRESHOLD) { //&& (std::find(pg.myFutureWork.begin(), pg.myFutureWork.end(), node) != node)) {
            //if strategy is changed that you may have to take into account if a node is already on the second list
            pg.myFutureWork.push_back(node);
         }
      }
      //std::cout << "size old work " << pg.myWork.size() << " new work " << pg.myFutureWork.size() << "\n";
      pg.myWork.swap(pg.myFutureWork);//swap lists
   }
#endif
};
/*********************************************************************************
 *
 **********************************************************************************/

// [hostid] -> vector of GID that host has replicas of
std::vector<std::vector<unsigned> > remoteReplicas;
// [hostid] -> remote pGraph Structure (locally invalid)
std::vector<PGraph*> magicPointer;

/*********************************************************************************
 *
 **********************************************************************************/

//An active work just ended its work
void decreaseActiveWorkers(uint32_t hostID) {
   //std::cout <<  galois::runtime::NetworkInterface::ID << " received a notice that " << hostID << " stopped working, alive workers: " << MWork.active_workers << "\n";
   MWork.active_workers--;
}

void decreaseBarrier(uint32_t hostID) { //creating my own barrier
   MWork.barrier_counter--;
   std::cout << galois::runtime::NetworkInterface::ID << " received a notice that " << hostID << " reached barrier " << MWork.barrier_counter << "\n";
}

void setRemotePtr(uint32_t hostID, PGraph* p) {
   if (hostID >= magicPointer.size())
      magicPointer.resize(hostID + 1);
   magicPointer[hostID] = p;
}

/*********************************************************************************
 *
 **********************************************************************************/

void recvNodeStatic(unsigned GID, uint32_t hostID) {
   if (hostID >= remoteReplicas.size())
      remoteReplicas.resize(hostID + 1);
   remoteReplicas[hostID].push_back(GID);
}

/*********************************************************************************
 *
 **********************************************************************************/

void setNodeValue(PGraph* p, unsigned GID, float v) {
   switch (personality) {
   case CPU:
      p->g.getData(p->G2L(GID)).value = v;
      break;
   case GPU_CUDA:
      setNodeValue_CUDA(cuda_ctx, p->G2L(GID), v);
      break;
   case GPU_OPENCL:
      cl_ctx.getData(p->G2L(GID)).value = v;
      break;
   default:
      break;
   }
}
/*********************************************************************************
 *
 **********************************************************************************/
// could be merged with setNodeValue, but this is one-time only ...
void setNodeAttr(PGraph *p, unsigned GID, unsigned nout) {
   switch (personality) {
   case CPU:
      p->g.getData(p->G2L(GID)).nout = nout;
      break;
   case GPU_CUDA:
      setNodeAttr_CUDA(cuda_ctx, p->G2L(GID), nout);
      break;
   case GPU_OPENCL:
      cl_ctx.getData(p->G2L(GID)).nout = nout;
      break;
   default:
      break;
   }
}
/*********************************************************************************
 *
 **********************************************************************************/
// send values for nout calculated on my node
void setNodeAttr2(PGraph *p, unsigned GID, unsigned nout) {
   auto LID = GID - p->g_offset;
   //printf("%d setNodeAttrs2 GID: %u nout: %u LID: %u\n", p->id, GID, nout, LID);
   switch (personality) {
   case CPU:
      p->g.getData(LID).nout += nout;
      break;
   case GPU_CUDA:
      setNodeAttr2_CUDA(cuda_ctx, LID, nout);
      break;
   case GPU_OPENCL:
      cl_ctx.getData(LID).nout += nout;
      break;
   default:
      break;
   }
}

/*********************************************************************************
 *
 **********************************************************************************/
void sendGhostCellAttrs2(galois::runtime::NetworkInterface& net, PGraph& g) {
   for (auto n = g.g.begin() + g.numOwned; n != g.g.begin() + g.numNodes; ++n) {
      auto l2g_ndx = std::distance(g.g.begin(), n) - g.numOwned;
      auto x = g.getHost(g.L2G[l2g_ndx]);
      //printf("%d: sendAttr2 GID: %d own: %d\n", g.id, g.L2G[l2g_ndx], x);
      switch (personality) {
      case CPU:
         net.sendAlt(x, setNodeAttr2, magicPointer[x], g.L2G[l2g_ndx], g.g.getData(*n).nout);
         break;
      case GPU_CUDA:
         net.sendAlt(x, setNodeAttr2, magicPointer[x], g.L2G[l2g_ndx], getNodeAttr2_CUDA(cuda_ctx, *n));
         break;
      case GPU_OPENCL:
         net.sendAlt(x, setNodeAttr2, magicPointer[x], g.L2G[l2g_ndx], cl_ctx.getData(*n).nout);
         break;
      default:
         assert(false);
         break;
      }
   }
}

/*********************************************************************************
 *
 * 
 * 
 * 
 **********************************************************************************/

void sendGhostCellAttrs(galois::runtime::NetworkInterface& net, PGraph& g) {
   for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
      for (auto n : remoteReplicas[x]) {
         /* no per-personality needed but until nout is
          fixed for CPU and OpenCL ... */

         switch (personality) {
         case CPU:
            net.sendAlt(x, setNodeAttr, magicPointer[x], n, g.g.getData(n - g.g_offset).nout);
            break;
         case GPU_CUDA:
            net.sendAlt(x, setNodeAttr, magicPointer[x], n, getNodeAttr_CUDA(cuda_ctx, n - g.g_offset));
            break;
         case GPU_OPENCL:
            net.sendAlt(x, setNodeAttr, magicPointer[x], n, cl_ctx.getData(n - g.g_offset).nout);
            break;
         default:
            assert(false);
            break;
         }
      }
   }
}

/*********************************************************************************
 * Send ghost-cell updates to all hosts that require it. Go over all the remotereplica
 * arrays, and for each array, go over all the elements and send it to the host 'x'.
 * Note that we use the magicPointer array to obtain the reference of the graph object
 * where the node data is to be set.
 **********************************************************************************/
void sendGhostCells(galois::runtime::NetworkInterface& net, PGraph& g) {
   for (unsigned x = 0; x < remoteReplicas.size(); ++x) {
      for (auto n : remoteReplicas[x]) {
         switch (personality) {
         case CPU:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, g.g.getData(n - g.g_offset).value);
            break;
         case GPU_CUDA:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, getNodeValue_CUDA(cuda_ctx, n - g.g_offset));
            break;
         case GPU_OPENCL:
            net.sendAlt(x, setNodeValue, magicPointer[x], n, cl_ctx.getData((n - g.g_offset)).value);
            break;
         default:
            assert(false);
            break;
         }
      }
   }
}

/*********************************************************************************
 *
 **********************************************************************************/

MarshalGraph pGraph2MGraph(PGraph &g) {
   MarshalGraph m;

   m.nnodes = g.numNodes;
   m.nedges = g.numEdges;
   m.nowned = g.numOwned;
   m.g_offset = g.g_offset;
   m.id = g.id;
   m.row_start = (index_type *) calloc(m.nnodes + 1, sizeof(index_type));
   m.edge_dst = (index_type *) calloc(m.nedges, sizeof(index_type));

   // TODO: initialize node_data and edge_data
   m.node_data = NULL;
   m.edge_data = NULL;

   // pinched from Rashid's LC_LinearArray_Graph.h

   size_t edge_counter = 0, node_counter = 0;
   for (auto n = g.g.begin(); n != g.g.end() && *n != m.nnodes; n++, node_counter++) {
      m.row_start[node_counter] = edge_counter;
      if (*n < g.numOwned) {
         for (auto e = g.g.edge_begin(*n); e != g.g.edge_end(*n); e++) {
            if (g.g.getEdgeDst(e) < g.numNodes)
               m.edge_dst[edge_counter++] = g.g.getEdgeDst(e);
         }
      }
   }

   m.row_start[node_counter] = edge_counter;
   m.nedges = edge_counter;
   //   printf("dropped %u edges (g->r, g->l, g->g)\n", g.numEdges - m.nedges);
   // for(int i = 0; i < node_counter; i++) {
   //   printf("%u ", m.row_start[i]);
   // }

   // for(int i = 0; i < edge_counter; i++) {
   //   printf("%u ", m.edge_dst[i]);
   // }

   // printf("nc: %zu ec: %zu\n", node_counter, edge_counter);
   return m;
}

/*********************************************************************************
 *
 **********************************************************************************/

void loadGraphNonCPU(PGraph &g) {
   MarshalGraph m;
   assert(personality != CPU);
   switch (personality) {
   case GPU_CUDA:
      m = pGraph2MGraph(g);
      load_graph_CUDA(cuda_ctx, m);
      break;
   case GPU_OPENCL:
      cl_ctx.loadGraphNonCPU(g.g, g.numOwned, g.numEdges, g.numNodes - g.numOwned);
      break;
   default:
      assert(false);
      break;
   }
   // TODO cleanup marshalgraph, leaks memory!
}

void myBarrier(int myID) { //very lazy implementation of a barrier just to compare against the one that is cause massive negative impact on parallel performance
   //Sending messages is the problem, this barrier didn't improve nothing at all!!
   /*MWork.barrier_counter--; //decrease own
    for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; ++x) { //Notify everyone that it has reached the barrier
    if (x==myID){ //avoid sending to itself
    continue;
    }
    std::cout << myID << " - " << galois::runtime::NetworkInterface::Num << " - " <<
    galois::runtime::NetworkInterface::ID << " - " << x << "\n";
    net.sendAlt(x, decreaseBarrier, galois::runtime::NetworkInterface::ID);
    }
    //std::cout << myID << "reached barrier after decreasing its own " << MWork.barrier_counter << "\n";
    while (MWork.barrier_counter > 0) { //while others don't reach the barrie
    * //should change this to a fast sleep or something
    myID = myID;
    }
    MWork.barrier_counter = galois::runtime::NetworkInterface::Num; //setup barrier for the next time*/
   return;
}

/*********************************************************************************
 *
 **********************************************************************************/
void inner_main() {
   auto& net = galois::runtime::getSystemNetworkInterface();
   galois::StatManager statManager;
   auto& barrier = galois::runtime::getSystemBarrier();
   const unsigned my_host_id = galois::runtime::NetworkInterface::ID;
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

   fprintf(stderr, "Pre-barrier - Host: %d, Personality %s\n", galois::runtime::NetworkInterface::ID, personality_str(personality).c_str());
   barrier.wait();
   fprintf(stderr, "Post-barrier - Host: %d, Personality %s\n", galois::runtime::NetworkInterface::ID, personality_str(personality).c_str());

   PGraph g;
   g.loadGraph(inputFile);

   if (personality == GPU_CUDA) {
      cuda_ctx = get_CUDA_context(galois::runtime::NetworkInterface::ID);
      if (!init_CUDA_context(cuda_ctx, gpudevice))
         return;
   } else if (personality == GPU_OPENCL) {
      galois::opencl::cl_env.init(cldevice);
   }
   if (personality != CPU)
      loadGraphNonCPU(g);
#if _HETERO_DEBUG_
   std::cout << g.id << " graph loaded\n";
#endif

   //local initialization
   if (personality == CPU) {
      InitializeGraph::go(g.g, g.numOwned);
   } else if (personality == GPU_CUDA) {
      initialize_graph_cuda(cuda_ctx);
   } else if (personality == GPU_OPENCL) {
      cl_ctx.init(g.numOwned, g.numNodes);
   }
#if _HETERO_DEBUG_
   std::cout << g.id << " initialized\n";
#endif
   barrier.wait();
   //send pGraph pointers
   for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; ++x)
      net.sendAlt(x, setRemotePtr, galois::runtime::NetworkInterface::ID, &g);
   //Ask for cells
   for (auto GID : g.L2G)
      net.sendAlt(g.getHost(GID), recvNodeStatic, GID, galois::runtime::NetworkInterface::ID);
#if _HETERO_DEBUG_
   std::cout << "[" << my_host_id << "]:ask for remote replicas\n";
#endif
   barrier.wait();

   // send out partial contributions for nout from local -> ghost
   sendGhostCellAttrs2(net, g);
   barrier.wait();

   //initialize active workers
   MWork.active_workers = galois::runtime::NetworkInterface::Num;
   MWork.barrier_counter = galois::runtime::NetworkInterface::Num;
   MWork.first_time = true;

   //initialize work list in case its a CPU
   if (personality == CPU) {
      for (auto i = g.g.begin(); i != g.g.begin() + g.numOwned; ++i) {
//         g.myWork.push_back(*i);
         g.work[g.active_bag % 2].push(*i);
      }
   }

   // send final nout values to remote replicas
#if _HETERO_DEBUG_
   std::cout << "[" << my_host_id << "]:ask for ghost cell attrs\n";
#endif
   sendGhostCellAttrs(net, g);
   barrier.wait();
   int counter = 0, total_nodes = g.numOwned;
   MWork.my_amount_of_work = g.numOwned;

//   {
//      galois::Timer timer;
//      unsigned long int barrier_time = 0;
//      for (int i = 0; i < 10; ++i) {
//         timer.start();
//         barrier.wait();
//         timer.stop();
//         barrier_time += timer.get();
//      }
//      fprintf(stderr, "Barrier time :: %u \n", (barrier_time));
//   }
   ////////////////BeginLoop/////////////////////////////////
   //Even after the work has being done on this device, it will still have to go thourgh the barrier.wait otherwise the other processes won't finish.
   while (MWork.active_workers > 0) {
#if _HETERO_DEBUG_
      std::cout << "debug," << g.id << "," << counter << "," << MWork.my_amount_of_work << "," << MWork.active_workers << "\n";
#endif
      counter++;
      if (counter > 50) { //in case something goes wrong it won't cycle forever
         break;
      }

      //send the updates
      //maybe not all nodes should be updated. I could change the Node struct to store the value of previous iteration and check if it changed. If it didn't, then sending a ghost cell is useless.
      sendGhostCells(net, g);
      //only nodes that actually changed should be sent.
      //Also, wouldn't be more efficient if you go for each device, collect all nodes that you have to send and then send them all together

      barrier.wait();
      switch (personality) {
      case CPU:
         PageRank::go(g);
         MWork.my_amount_of_work = std::distance(g.work[g.current()].begin(), g.work[g.current()].end());      //g.myWork.size();
         total_nodes += MWork.my_amount_of_work;
         break;
      case GPU_OPENCL:
         MWork.my_amount_of_work = cl_ctx(g.numOwned);
         break;
      case GPU_CUDA:
         MWork.my_amount_of_work = pagerank_cuda(cuda_ctx);
         total_nodes += MWork.my_amount_of_work;
         break;
      default:
         std::cout << "Wrong choice\n";
         break;
      }
      if (MWork.my_amount_of_work == 0 && MWork.first_time) { //Check if node ran out of work. Need the first_time otherwise at a certain point we are always decreasing the number of workrers
         MWork.first_time = false;
         MWork.active_workers--;
         for (uint32_t x = 0; x < galois::runtime::NetworkInterface::Num; x++) { //Notify everyone that this device work has ended
            if (x == g.id) { //this must be added otherwise some processes will be stuck because of the barrier and the net.sendAlt to itself
               continue;
            }
            net.sendAlt(x, decreaseActiveWorkers, g.id);
         }
      }
      ////////////////EndLoop/////////////////////////////////
      barrier.wait();
   }

   //Final synchronization to ensure that all the nodes are updated.
   sendGhostCells(net, g);
   barrier.wait();
   //myBarrier(g.id);

   if (verify) {
      //printing final results
      //this debug won't work when using GPU's
      /*std::cout << my_host_id <<" PRINTING FINAL PR AFTER " << counter << " ITERATIONS\n";
       for (auto i = g.g.begin(); i != g.g.begin() + g.numOwned; ++i) {
       //LNode n = g.g.getData(i);
       std::cout << "NODE " << *i + g.g_offset << " has PR: " << g.g.getData(*i).value << "\n";
       }*/
      std::stringstream ss;
      ss << personality_str(personality) << "_" << my_host_id << "_of_" << galois::runtime::NetworkInterface::Num << "_page_ranks.csv";
      std::ofstream out_file(ss.str());
      switch (personality) {
      case CPU: {
         for (auto n = g.g.begin(); n != g.g.begin() + g.numOwned; ++n) {
            out_file << *n + g.g_offset << ", " << g.g.getData(*n).value << ", " << g.g.getData(*n).nout << "\n";
         }
         break;
      }
      case GPU_OPENCL: {
         for (int n = 0; n < g.numOwned; ++n) {
            out_file << n + g.g_offset << ", " << cl_ctx.getData(n).value << ", " << cl_ctx.getData(n).nout << "\n";
         }
         break;
      }
      case GPU_CUDA:
         for (int n = 0; n < g.numOwned; n++) {
            out_file << n + g.g_offset << ", " << getNodeValue_CUDA(cuda_ctx, n) << ", " << getNodeAttr_CUDA(cuda_ctx, n) << "\n";
         }
         break;
      }
      out_file.close();
   }
   std::cout.flush();

}

int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   auto& net = galois::runtime::getSystemNetworkInterface();
   inner_main();
   return 0;
}
