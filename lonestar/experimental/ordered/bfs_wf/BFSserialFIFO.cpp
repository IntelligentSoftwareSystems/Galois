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

#include <deque>

#include "bfs.h"

class BFSserialFIFO : public BFS {

  typedef std::deque<GNode> WL_ty;
  typedef BFS Super_ty;

public:
  virtual const std::string getVersion() const { return "Serial FIFO"; }

  virtual size_t runBFS(Graph& graph, GNode& startNode) {

    WL_ty worklist;
    size_t niter = 0;

    graph.getData(startNode, galois::MethodFlag::UNPROTECTED) = 0;
    worklist.push_back(startNode);

    while (!worklist.empty()) {

      GNode src = worklist.front();
      worklist.pop_front();

      Super_ty::bfsOperator<false>(graph, src, worklist);

      ++niter;
    }

    return niter;
  }
};
int main(int argc, char* argv[]) {
  BFSserialFIFO sf;
  sf.run(argc, argv);
  return 0;
}
