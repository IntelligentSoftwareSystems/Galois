/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "Analysis.hpp"
#include <algorithm>

using namespace std;

void Analysis::nodeAnaliser(Node* node, set<uint64_t>* parent) {
  auto getAllSupernodes = [](Node* n) {
    set<uint64_t>* supernodes = new set<uint64_t>();
    for (Element* e : n->getElements()) {
      for (uint64_t supernode : e->supernodes)
        supernodes->insert(supernode);
    }
    return supernodes;
  };

  set<uint64_t>* common;

  if (node->getLeft() != NULL && node->getRight() != NULL) {
    set<uint64_t>* lSupernodes = getAllSupernodes(node->getLeft());
    set<uint64_t>* rSupernodes = getAllSupernodes(node->getRight());

    common = new set<uint64_t>;
    std::set_intersection(lSupernodes->begin(), lSupernodes->end(),
                          rSupernodes->begin(), rSupernodes->end(),
                          std::inserter(*common, common->begin()));

    delete (lSupernodes);
    delete (rSupernodes);

    for (auto p = parent->cbegin(); p != parent->cend(); ++p) {
      common->insert(*p);
    }

    Analysis::nodeAnaliser(node->getLeft(), common);
    Analysis::nodeAnaliser(node->getRight(), common);

  } else {
    common = getAllSupernodes(node);
  }

  int i = 0;

  for (uint64_t supernode : *common) {
    if (!parent->count(supernode)) {
      node->addSupernode(supernode);
      ++i;
    }
  }

  node->setSupernodesToElim(i);

  for (uint64_t supernode : *common) {
    if (parent->count(supernode)) {
      node->addSupernode(supernode);
    }
  }

  delete common;
}

void Analysis::doAnalise(Mesh* mesh) {
  Node* root                 = mesh->getRootNode();
  std::set<uint64_t>* parent = new set<uint64_t>();
  Analysis::nodeAnaliser(root, parent);
  Analysis::mergeAnaliser(root);

  delete parent;
}

void Analysis::mergeAnaliser(Node* node) {
  Node* left;
  Node* right;
  int i;

  left  = node->getLeft();
  right = node->getRight();

  if (left != NULL && right != NULL) {
    node->leftPlaces.resize(left->getSupernodes().size() -
                            left->getSupernodesToElim());
    node->rightPlaces.resize(right->getSupernodes().size() -
                             right->getSupernodesToElim());

    map<uint64_t, uint64_t> reverseMap;

    for (i = 0; i < node->getSupernodes().size(); ++i) {
      reverseMap[node->getSupernodes()[i]] = i;
    }

    for (i = left->getSupernodesToElim(); i < left->getSupernodes().size();
         ++i) {
      node->leftPlaces[i - left->getSupernodesToElim()] =
          reverseMap[left->getSupernodes()[i]];
    }

    for (i = right->getSupernodesToElim(); i < right->getSupernodes().size();
         ++i) {
      node->rightPlaces[i - right->getSupernodesToElim()] =
          reverseMap[right->getSupernodes()[i]];
    }

    Analysis::mergeAnaliser(left);
    Analysis::mergeAnaliser(right);
  }
  node->computeOffsets();
}

void Analysis::debugNode(Node* n) {
  printf("Node: %d\n", n->getId());
  printf("  dofs: ");
  for (uint64_t dof : n->getSupernodes()) {
    printf("%lu ", dof);
  }
  printf("\n");
}

void Analysis::printTree(Node* n) {
  printf("Node id: %d ", n->getId());
  printf("[");
  for (uint64_t dof : n->getSupernodes()) {
    printf("%lu, ", dof);
  }
  printf("]");
  printf(" elim: %lu\n", n->getSupernodesToElim());

  if (n->getLeft() != NULL && n->getRight() != NULL) {
    printTree(n->getLeft());
    printTree(n->getRight());
  }
}

void Analysis::printElement(Element* e) {
  printf("E[%lu,%lu] = [", e->k, e->l);
  for (uint64_t supernode : e->supernodes) {
    printf("%lu, ", supernode);
  }
  printf("]\n");
}
