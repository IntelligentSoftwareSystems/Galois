#ifndef GALOIS_CONDITIONCHECKER_H
#define GALOIS_CONDITIONCHECKER_H

#include "../model/Graph.h"

//! An implementation of a condition checker just needs to implement
//! the execute function to check the condition on some node
class ConditionChecker {
public:
  virtual bool execute(GNode& node) = 0;
};

#endif // GALOIS_CONDITIONCHECKER_H
