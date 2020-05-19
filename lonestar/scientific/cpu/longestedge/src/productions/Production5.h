#ifndef GALOIS_PRODUCTION5_H
#define GALOIS_PRODUCTION5_H

#include "Production.h"
#include "../utils/ConnectivityManager.h"
#include "../utils/GaloisUtils.h"

class Production5 : public Production {
private:
  bool checkApplicabilityCondition(
      const std::vector<optional<EdgeIterator>>& edgesIterators) const {
    return connManager.countBrokenEdges(edgesIterators) == 2;
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
    if (longestEdges.size() > 1) {
      return false;
    }

    //        logg(pState.getInteriorData(), pState.getVerticesData());
    const vector<int>& brokenEdges = pState.getBrokenEdges();
    if (std::find(brokenEdges.begin(), brokenEdges.end(), longestEdges[0]) ==
        brokenEdges.end()) {
      breakElementWithoutHangingNode(longestEdges[0], pState, ctx);
      //            std::cout << "P5 executed ";
      return true;
    }
    return false;
  }
};

#endif // GALOIS_PRODUCTION2_H
