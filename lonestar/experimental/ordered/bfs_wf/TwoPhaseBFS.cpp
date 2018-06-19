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

#include <vector>
#include <functional>

#include "galois/runtime/KDGtwoPhase.h"

#include "bfs.h"
#include "bfsParallel.h"

class TwoPhaseBFS : public BFS {

  struct GlobalLevel {
    unsigned value = 1;
    bool updated   = false;
  };

  // relies on round based execution of IKDG executor
  struct VisitNhoodSafety : public VisitNhood {
    GlobalLevel& globalLevel;

    VisitNhoodSafety(Graph& graph, GlobalLevel& globalLevel)
        : VisitNhood(graph), globalLevel(globalLevel) {}

    template <typename C>
    void operator()(const Update& up, C& ctx) {
      if (up.level <= globalLevel.value) {
        VisitNhood::operator()(up, ctx);
      } else {
        galois::runtime::signalConflict();
      }

      if (galois::substrate::ThreadPool::getTID() == 0 && globalLevel.updated) {
        globalLevel.updated = false;
      }
    }
  };

  // relies on round based execution of IKDG executor
  struct OpFuncSafety : public OpFunc {
    GlobalLevel& globalLevel;

    OpFuncSafety(Graph& graph, ParCounter& numIter, GlobalLevel& globalLevel)
        : OpFunc(graph, numIter), globalLevel(globalLevel) {}

    template <typename C>
    void operator()(const Update& up, C& ctx) {
      if (galois::substrate::ThreadPool::getTID() == 0 &&
          !globalLevel.updated) {
        globalLevel.updated = true;
        ++globalLevel.value;
      }

      OpFunc::operator()(up, ctx);
    }
  };

public:
  virtual const std::string getVersion() const { return "Two Phase ordered"; }

  virtual size_t runBFS(Graph& graph, GNode& startNode) {

    ParCounter numIter;

    GlobalLevel globalLevel;

    // update request for root
    Update first(startNode, 0);

    std::vector<Update> wl;
    wl.push_back(first);

    galois::runtime::for_each_ordered_ikdg(
        galois::runtime::makeStandardRange(wl.begin(), wl.end()), Comparator(),
        VisitNhoodSafety(graph, globalLevel),
        OpFuncSafety(graph, numIter, globalLevel),
        std::make_tuple(galois::loopname("bfs_two_phase_safety")));

    std::cout << "number of iterations: " << numIter.reduce() << std::endl;

    return numIter.reduce();
  }
};

int main(int argc, char* argv[]) {
  TwoPhaseBFS wf;
  wf.run(argc, argv);
  return 0;
}
