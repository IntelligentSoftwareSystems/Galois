#ifndef GALOIS_DUMMYCONDITIONCHECKER_H
#define GALOIS_DUMMYCONDITIONCHECKER_H

#include "ConditionChecker.h"

//! This condition checker always sets a hyperedge node to be refined and
//! returns true
class DummyConditionChecker : ConditionChecker {
public:

  //! Sets refinement and returns true for hyperedge nodes
  bool execute(GNode& node) override {
    NodeData& nodeData = node->getData();
    if (!nodeData.isHyperEdge()) {
      return false;
    }
    nodeData.setToRefine(true);
    return true;
  }
};

#endif // GALOIS_DUMMYCONDITIONCHECKER_H
