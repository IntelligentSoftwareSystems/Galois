/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef KRUSKAL_HAND_H
#define KRUSKAL_HAND_H

#include "Kruskal.h"
#include "KruskalParallel.h"

namespace kruskal {

class KruskalHand : public Kruskal {
protected:
  virtual const std::string getVersion() const {
    return "Handwritten using window-based two-phase union-find";
  }

  virtual void runMST(const size_t numNodes, VecEdge& edges, size_t& mstWeight,
                      size_t& totalIter) {

    if (edges.size() >= 2 * numNodes) {
      runMSTfilter(numNodes, edges, mstWeight, totalIter, UnionFindWindow());

    } else {

      runMSTsimple(numNodes, edges, mstWeight, totalIter, UnionFindWindow());
    }
  }
};

} // end namespace kruskal

#endif //  KRUSKAL_HAND_H
