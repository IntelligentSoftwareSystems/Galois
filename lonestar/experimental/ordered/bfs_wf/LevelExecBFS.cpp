/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include <vector>
#include <functional>

#include "galois/runtime/LevelExecutor.h"
#include "galois/worklists/WorkList.h"

#include "bfs.h"
#include "bfsParallel.h"

class LevelExecBFS : public BFS {

public:
  virtual const std::string getVersion() const {
    return "using Level-by-Level executor";
  }

  virtual size_t runBFS(Graph& graph, GNode& startNode) {

    ParCounter numIter;

    // update request for root
    Update first(startNode, 0);

    std::vector<Update> wl;
    wl.push_back(first);

    typedef galois::worklists::PerSocketChunkFIFO<OpFunc::CHUNK_SIZE, Update> C;
    typedef galois::worklists::OrderedByIntegerMetric<
        GetLevel, C>::with_barrier<true>::type WL_ty;

    galois::runtime::for_each_ordered_level(
        galois::runtime::makeStandardRange(wl.begin(), wl.end()), GetLevel(),
        std::less<unsigned>(), VisitNhood(graph), OpFunc(graph, numIter));

    // galois::for_each (first,
    // OpFunc (graph, numIter),
    // galois::loopname ("bfs-level-exec"),
    // galois::wl<WL_ty> ());

    std::cout << "number of iterations: " << numIter.reduce() << std::endl;

    return numIter.reduce();
  }
};

int main(int argc, char* argv[]) {
  LevelExecBFS b;
  b.run(argc, argv);
  return 0;
}
