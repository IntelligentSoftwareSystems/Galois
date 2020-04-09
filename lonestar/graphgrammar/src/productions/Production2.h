#ifndef GALOIS_PRODUCTION2_H
#define GALOIS_PRODUCTION2_H

#include "Production.h"
#include "../utils/ConnectivityManager.h"
#include "../utils/utils.h"

class Production2 : public Production {
private:
  bool checkApplicabilityCondition(
      const std::vector<optional<EdgeIterator>>& edgesIterators) const {
    return connManager.countBrokenEdges(edgesIterators) == 1;
  }

public:
  using Production::Production;

  bool execute(ProductionState& pState,
               galois::UserContext<GNode>& ctx) override {
    if (!checkApplicabilityCondition(pState.getEdgesIterators())) {
      return false;
    }

    //        logg(pState.getInteriorData(), pState.getVerticesData());

    int brokenEdge = pState.getAnyBrokenEdge();
    assert(brokenEdge != -1);

    if (!checkIfBrokenEdgeIsTheLongest(brokenEdge, pState.getEdgesIterators(),
                                       pState.getVertices(),
                                       pState.getVerticesData())) {
      return false;
    }

    breakElementWithHangingNode(brokenEdge, pState, ctx);
    //        std::cout << "P2 executed ";
    return true;
  }
};

#endif // GALOIS_PRODUCTION2_H
