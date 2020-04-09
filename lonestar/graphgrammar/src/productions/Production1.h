#ifndef GALOIS_PRODUCTION1_H
#define GALOIS_PRODUCTION1_H

#include "Production.h"
#include "../utils/ConnectivityManager.h"
#include "../utils/utils.h"

class Production1 : public Production {
private:
  bool checkApplicabilityCondition(
      const NodeData& nodeData,
      const std::vector<optional<EdgeIterator>>& edgesIterators) const {
    return nodeData.isToRefine() && !connManager.hasBrokenEdge(edgesIterators);
  }

  int getEdgeToBreak(const ProductionState& pState) const {
    const vector<NodeData>& verticesData = pState.getVerticesData();
    for (int longest : pState.getLongestEdges()) {
      if (pState.getEdgesData()[longest].get().isBorder()) {
        return longest;
      }
      if (!verticesData[getEdgeVertices(longest).first].isHanging() &&
          !verticesData[getEdgeVertices(longest).second].isHanging()) {

        return longest;
      }
    }
    return -1;
  }

public:
  using Production::Production;

  bool execute(ProductionState& pState,
               galois::UserContext<GNode>& ctx) override {
    if (!checkApplicabilityCondition(pState.getInteriorData(),
                                     pState.getEdgesIterators())) {
      return false;
    }

    //        logg(pState.getInteriorData(), pState.getVerticesData());

    int edgeToBreak = getEdgeToBreak(pState);
    if (edgeToBreak == -1) {
      return false;
    }

    breakElementWithoutHangingNode(edgeToBreak, pState, ctx);
    //        std::cout << "P1 executed ";

    return true;
  }
};

#endif // GALOIS_PRODUCTION1_H
