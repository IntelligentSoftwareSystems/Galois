#ifndef GALOIS_PRODUCTION3_H
#define GALOIS_PRODUCTION3_H

#include "Production.h"
#include "../utils/ConnectivityManager.h"
#include "../utils/GaloisUtils.h"

class Production3 : public Production {
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

    int brokenEdge = pState.getAnyBrokenEdge();
    assert(brokenEdge != -1);

    if (checkIfBrokenEdgeIsTheLongest(brokenEdge, pState.getEdgesIterators(),
                                      pState.getVertices(),
                                      pState.getVerticesData())) {
      return false;
    }

    //        logg(pState.getInteriorData(), pState.getVerticesData());

    const vector<int>& longestEdges = pState.getLongestEdges();

    for (int longest : longestEdges) {
      if (pState.getEdgesData()[longest].get().isBorder()) {
        breakElementWithoutHangingNode(longest, pState, ctx);
        //                std::cout << "P3 executed ";
        return true;
      }
    }
    for (int longest : longestEdges) {
      if (!pState.getVerticesData()[getEdgeVertices(longest).first]
               .isHanging() &&
          !pState.getVerticesData()[getEdgeVertices(longest).second]
               .isHanging()) {

        breakElementWithoutHangingNode(longest, pState, ctx);
        //                std::cout << "P3 executed ";
        return true;
      }
    }
    return false;
  }
};

#endif // GALOIS_PRODUCTION3_H
