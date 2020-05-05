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

#include "Node.hpp"
#include <set>
#include <algorithm>
#include "Analysis.hpp"

void Node::setLeft(Node* left) { this->left = left; }

void Node::setRight(Node* right) { this->right = right; }

void Node::setParent(Node* parent) { this->parent = parent; }

void Node::addElement(Element* e) { this->mergedElements.push_back(e); }

void Node::clearElements() { this->mergedElements.clear(); }

void Node::setProduction(std::string& prodname) { this->production = prodname; }

std::string& Node::getProduction() { return this->production; }

Node* Node::getLeft() const { return this->left; }

Node* Node::getRight() const { return this->right; }

Node* Node::getParent() const { return this->parent; }

std::vector<Element*>& Node::getElements() { return this->mergedElements; }

int Node::getId() const { return this->node; }

void Node::addSupernode(uint64_t supernode) {
  this->supernodes.push_back(supernode);
}

std::vector<uint64_t>& Node::getSupernodes() { return this->supernodes; }

void Node::clearSupernodes() { this->supernodes.clear(); }

void Node::setSupernodesToElim(uint64_t supernodes) {
  this->supernodesToElim = supernodes;
}

uint64_t Node::getSupernodesToElim() const { return this->supernodesToElim; }

void Node::computeOffsets() {
  uint64_t j;
  uint64_t k;

  j = 0;
  k = 0;
  offsets.resize(getSupernodes().size() + 1);
  for (uint64_t i : getSupernodes()) {
    offsets[k] = j;
    j += (*meshSupernodes)[i];
    ++k;
  }
  offsets[k] = j;
}

void Node::allocateSystem(SolverMode mode) {
  uint64_t dofs;

  dofs = 0;
  for (uint64_t supernode : getSupernodes()) {
    dofs += (*meshSupernodes)[supernode];
  }
  this->system = new EquationSystem(dofs, mode);
}

void Node::deallocateSystem() {
  if (this->system)
    delete this->system;
  this->system = NULL;
}

void Node::fillin() const {
  int i, j;
  double** matrix = this->system->matrix;
  double* rhs     = this->system->rhs;
  int n           = this->system->n;

  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      matrix[j][i] = i == j ? 1.0 : 0.0;
    }
    rhs[j] = 1.0;
  }
}

void Node::merge() const {
  int i, j;
  uint64_t k, l;
  Node *left, *right;
  double** matrix;
  double** lmatrix;
  double** rmatrix;
  double* rhs;
  double* rrhs;
  double* lrhs;

  left  = this->left;
  right = this->right;

  matrix = system->matrix;
  rhs    = system->rhs;

  lmatrix = left->system->matrix;
  rmatrix = right->system->matrix;

  rrhs = right->system->rhs;
  lrhs = left->system->rhs;
  if (system->mode != CHOLESKY) {
    for (j = left->getSupernodesToElim(); j < left->getSupernodes().size();
         ++j) {
      for (i = left->getSupernodesToElim(); i < left->getSupernodes().size();
           ++i) {
        for (k = left->offsets[j]; k < left->offsets[j + 1]; ++k) {
          uint64_t x;
          x = offsets[leftPlaces[j - left->getSupernodesToElim()]] +
              (k - left->offsets[j]);
          for (l = left->offsets[i]; l < left->offsets[i + 1]; ++l) {
            uint64_t y;
            y = offsets[leftPlaces[i - left->getSupernodesToElim()]] +
                (l - left->offsets[i]);
            matrix[x][y] = lmatrix[k][l];
          }
        }
      }
      for (k = left->offsets[j]; k < left->offsets[j + 1]; ++k) {
        rhs[offsets[leftPlaces[j - left->getSupernodesToElim()]] +
            (k - left->offsets[j])] = lrhs[k];
      }
    }

    for (j = right->getSupernodesToElim(); j < right->getSupernodes().size();
         ++j) {
      for (i = right->getSupernodesToElim(); i < right->getSupernodes().size();
           ++i) {
        for (k = right->offsets[j]; k < right->offsets[j + 1]; ++k) {
          uint64_t x;
          x = offsets[rightPlaces[j - right->getSupernodesToElim()]] +
              (k - right->offsets[j]);
          for (l = right->offsets[i]; l < right->offsets[i + 1]; ++l) {
            uint64_t y;
            y = offsets[rightPlaces[i - right->getSupernodesToElim()]] +
                (l - right->offsets[i]);
            matrix[x][y] += rmatrix[k][l];
          }
        }
      }
      for (k = right->offsets[j]; k < right->offsets[j + 1]; ++k) {
        rhs[offsets[rightPlaces[j - right->getSupernodesToElim()]] +
            (k - right->offsets[j])] += rrhs[k];
      }
    }
  } else {
    for (j = left->getSupernodesToElim(); j < left->getSupernodes().size();
         ++j) {
      for (i = j; i < left->getSupernodes().size(); ++i) {
        for (k = left->offsets[j]; k < left->offsets[j + 1]; ++k) {
          uint64_t x;
          x = offsets[leftPlaces[j - left->getSupernodesToElim()]] +
              (k - left->offsets[j]);
          for (l = i == j ? k : left->offsets[i]; l < left->offsets[i + 1];
               ++l) {
            uint64_t y;
            y = offsets[leftPlaces[i - left->getSupernodesToElim()]] +
                (l - left->offsets[i]);
            matrix[x][y] = lmatrix[k][l];
          }
        }
      }
      for (k = left->offsets[j]; k < left->offsets[j + 1]; ++k) {
        rhs[offsets[leftPlaces[j - left->getSupernodesToElim()]] +
            (k - left->offsets[j])] = lrhs[k];
      }
    }

    for (j = right->getSupernodesToElim(); j < right->getSupernodes().size();
         ++j) {
      for (i = j; i < right->getSupernodes().size(); ++i) {
        for (k = right->offsets[j]; k < right->offsets[j + 1]; ++k) {
          uint64_t x;
          x = offsets[rightPlaces[j - right->getSupernodesToElim()]] +
              (k - right->offsets[j]);
          for (l = right->offsets[i]; l < right->offsets[i + 1]; ++l) {
            uint64_t y;
            y = offsets[rightPlaces[i - right->getSupernodesToElim()]] +
                (l - right->offsets[i]);
            matrix[x][y] += rmatrix[k][l];
          }
        }
      }
      for (k = right->offsets[j]; k < right->offsets[j + 1]; ++k) {
        rhs[offsets[rightPlaces[j - right->getSupernodesToElim()]] +
            (k - right->offsets[j])] += rrhs[k];
      }
    }
  }
}

void Node::eliminate() const {
  uint64_t dofs;
  uint64_t i;

  if ((left == NULL && right != NULL) || (left != NULL && right == NULL)) {
    printf("Error at node: %d\n", node);
    throw std::string("invalid tree!");
  }

  if (left != NULL && right != NULL) {
    // this->mergeProduction(left->system->matrix, left->system->rhs,
    //                      right->system->matrix, right->system->rhs,
    //                      this->system->matrix, this->system->rhs);
    this->merge();
  } else {
    // this->preprocessProduction();
    this->fillin();
  }
  dofs = 0;
  for (i = 0; i < getSupernodesToElim(); ++i) {
    dofs += (*meshSupernodes)[supernodes[i]];
  }
  system->eliminate(dofs);
}

void Node::bs() const {
  // system->backwardSubstitute(this->getSupernodesToElim());
  /*for (int i=0; i<system->n; ++i) {
      if (fabs(system->rhs[i]-1.0) > 1e-8) {
          printf("WRONG SOLUTION - [%lu] %d: %lf\n", this->getId(), i,
  system->rhs[i]);
      }
  }*/
}

/* DEBUG*/

int Node::treeSize() {
  int ret = 1;
  if (this->getLeft() != NULL) {
    ret += this->getLeft()->treeSize();
  }
  if (this->getRight() != NULL) {
    ret += this->getRight()->treeSize();
  }
  return ret;
}

unsigned long Node::getSizeInMemory(bool recursive) {
  unsigned long total =
      this->supernodes.size() * this->supernodes.size() * sizeof(double) +
      this->supernodes.size() * sizeof(double*);
  if (recursive && left != NULL && right != NULL) {
    total += left->getSizeInMemory() + right->getSizeInMemory();
  }
  return total;
}

unsigned long Node::getFLOPs(bool recursive) {
  auto flops = [](unsigned int a, unsigned int b) {
    return a * (6 * b * b - 6 * a * b + 6 * b + 2 * a * a - 3 * a + 1) / 6;
  };

  unsigned long total =
      flops(this->getSupernodesToElim(), this->getSupernodes().size());
  if (recursive && left != NULL && right != NULL) {
    total += left->getFLOPs() + right->getFLOPs();
  }

  return total;
}

unsigned long Node::getMemoryRearrangements() {
  unsigned long total = 0;

  if (left != NULL && right != NULL) {
    unsigned long memLeft =
        (left->getSupernodes().size() - left->getSupernodesToElim());
    memLeft *= memLeft;
    unsigned long memRight =
        (right->getSupernodes().size() - right->getSupernodesToElim());
    memRight *= memRight;

    total = memLeft + memRight + left->getMemoryRearrangements() +
            right->getMemoryRearrangements();
  }

  return total;
}

/*END OF DEBUG*/
