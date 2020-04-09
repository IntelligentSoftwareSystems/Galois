#ifndef GALOIS_PRODUCTION6_H
#define GALOIS_PRODUCTION6_H

#include "Production.h"
#include "../utils/ConnectivityManager.h"
#include "../utils/utils.h"

class Production6 : public Production {
private:
  bool checkApplicabilityCondition(
      const std::vector<optional<EdgeIterator>>& edgesIterators) const {
    return connManager.countBrokenEdges(edgesIterators) == 3;
  }

public:
  using Production::Production;

  bool execute(ProductionState& pState,
               galois::UserContext<GNode>& ctx) override {
    if (!checkApplicabilityCondition(pState.getEdgesIterators())) {
      return false;
    }

    const vector<int>& longestEdges =
        pState.getLongestEdgesIncludingBrokenOnes();

    //        logg(pState.getInteriorData(), pState.getVerticesData());

    breakElementWithHangingNode(longestEdges[0], pState, ctx);
    //        std::cout << "P5 executed ";
    return true;
  }
};

#endif // GALOIS_PRODUCTION2_H
