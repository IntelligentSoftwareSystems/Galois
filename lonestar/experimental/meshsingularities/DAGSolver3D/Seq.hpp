#ifndef SEQ_HPP
#define SEQ_HPP

#include "Node.hpp"
#include "EquationSystem.hpp"

void seqAllocation(Node *node, SolverMode mode);
void seqElimination(Node *node);
void seqBackwardSubstitution(Node *node);

#endif
