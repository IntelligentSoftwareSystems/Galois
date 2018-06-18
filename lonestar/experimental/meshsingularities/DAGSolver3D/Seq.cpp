#include "Seq.hpp"

void seqAllocation(Node* node, SolverMode mode) {
  node->allocateSystem(mode);
  if (node->getLeft() != NULL && node->getRight() != NULL) {
    seqAllocation(node->getLeft(), mode);
    seqAllocation(node->getRight(), mode);
  }
}

void seqElimination(Node* node) {
  if (node->getLeft() != NULL && node->getRight() != NULL) {
    seqElimination(node->getLeft());
    seqElimination(node->getRight());
  }
  node->eliminate();
}

void seqBackwardSubstitution(Node* node) {
  node->bs();
  if (node->getLeft() != NULL && node->getRight() != NULL) {
    seqBackwardSubstitution(node->getLeft());
    seqBackwardSubstitution(node->getRight());
  }
}
