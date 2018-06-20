/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

/**Test program for vertex  cut **/
/**@Author: Gurbinder Gill (gurbinder533@gmail.com)**/

#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/CompilerHelperFunctions.h"

#include "galois/graphs/OfflineGraph.h"
#include "galois/Dist/vGraph.h"
#include "galois/DistAccumulator.h"

static const char* const name = "SSSP - Distributed Heterogeneous";
static const char* const desc = "Bellman-Ford SSSP on Distributed Galois.";
static const char* const url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> maxIterations("maxIterations",
                                            cll::desc("Maximum iterations"),
                                            cll::init(100));
static cll::opt<unsigned int>
    src_node("startNode", cll::desc("ID of the source node"), cll::init(0));
static cll::opt<bool>
    verify("verify",
           cll::desc("Verify ranks by printing to 'page_ranks.#hid.csv' file"),
           cll::init(false));

static cll::opt<std::string> partFolder("partFolder",
                                        cll::desc("path to partitionFolder"),
                                        cll::init(""));

struct NodeData {
  std::atomic<int> dist_current;
};

typedef vGraph<NodeData, unsigned int> Graph;
typedef typename Graph::GraphNode GNode;

galois::DGAccumulator<int> DGAccumulator_accum;

int main(int argc, char** argv) {
  try {
    LonestarStart(argc, argv, name, desc, url);
    auto& net = galois::runtime::getSystemNetworkInterface();
    galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init,
        T_HSSSP;

    Graph hg(inputFile, partFolder, net.ID, net.Num);

    std::cout << "SIZE : " << hg.size() << "\n";

    // if(net.ID == 0) {
    // for(auto ii = hg.begin(); ii != hg.end(); ++ii) {
    // std::cout << "[" << *ii << "]  " << hg.getData(*ii).dist_current << "\n";
    //}
    //}

    return 0;
  } catch (const char* c) {
    std::cerr << "Error: " << c << "\n";
    return 1;
  }
}
