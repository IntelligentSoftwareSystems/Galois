/**Test program for vertex  cut **/
/**@Author: Gurbinder Gill (gurbinder533@gmail.com)**/

#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/CompilerHelperFunctions.h"

#include "Galois/Dist/OfflineGraph.h"
#include "Galois/Dist/vGraph.h"
#include "Galois/DistAccumulator.h"


static const char* const name = "SSSP - Distributed Heterogeneous";
static const char* const desc = "Bellman-Ford SSSP on Distributed Galois.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations", cll::desc("Maximum iterations"), cll::init(100));
static cll::opt<unsigned int> src_node("srcNodeId", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool> verify("verify", cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"), cll::init(false));

static cll::opt<std::string> partFolder("partFolder", cll::desc("path to partitionFolder"), cll::init(""));

struct NodeData {
  std::atomic<int> dist_current;
};

typedef vGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;


galois::DGAccumulator<int> DGAccumulator_accum;

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::Runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_hGraph_init, T_init, T_HSSSP;

    Graph hg(inputFile, partFolder, net.ID, net.Num);

    std::cout << "SIZE : " << hg.size() << "\n";

      //if(net.ID == 0) {
        //for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
          //std::cout << "[" << *ii << "]  " << hg.getData(*ii).dist_current << "\n";
        //}
      //}

    return 0;
  } catch(const char* c) {
    std::cerr << "Error: " << c << "\n";
      return 1;
  }
}
