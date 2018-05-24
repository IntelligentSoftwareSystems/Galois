#ifndef KRUSKAL_HAND_H
#define KRUSKAL_HAND_H

#include "Kruskal.h"
#include "KruskalParallel.h"


namespace kruskal {


class KruskalHand: public Kruskal {
  protected:

  virtual const std::string getVersion () const { return "Handwritten using window-based two-phase union-find"; }

  virtual void runMST (const size_t numNodes, VecEdge& edges,
      size_t& mstWeight, size_t& totalIter) {

    if (edges.size () >= 2 * numNodes) {
      runMSTfilter (numNodes, edges, mstWeight, totalIter, UnionFindWindow ());

    } else {

      runMSTsimple (numNodes, edges, mstWeight, totalIter, UnionFindWindow ());
    }
  }
};


}// end namespace kruskal

#endif //  KRUSKAL_HAND_H

