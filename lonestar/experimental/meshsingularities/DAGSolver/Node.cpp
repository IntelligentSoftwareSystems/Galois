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

void Node::addDof(uint64_t dof) { this->dofs.push_back(dof); }

std::vector<uint64_t>& Node::getDofs() { return this->dofs; }

void Node::clearDofs() { this->dofs.clear(); }

void Node::setDofsToElim(uint64_t dofs) { this->dofsToElim = dofs; }

uint64_t Node::getDofsToElim() const { return this->dofsToElim; }

void Node::allocateSystem(SolverMode mode) {
  this->system = new EquationSystem(this->getDofs().size(), mode);
  // printf("Size: %d x %d\n", system->n, system->n);
}

void Node::fillin() const {
  int i;
  for (int j = 0; j < this->system->n; ++j) {
    for (i = 0; i < this->system->n; ++i) {
      this->system->matrix[j][i] = i == j ? 1.0 : 0.0;
    }
    this->system->rhs[j] = 1.0;
  }
}

void Node::merge() const {
  double** M_right;
  double** M_left;
  double** M;

  double* RHS_right;
  double* RHS_left;
  double* RHS;

  int dofs_left;
  int dofs_right;

  int dofs_to_elim_left;
  int dofs_to_elim_right;

  M_right = right->system->matrix;
  M_left  = left->system->matrix;
  M       = system->matrix;

  RHS_right = right->system->rhs;
  RHS_left  = left->system->rhs;
  RHS       = system->rhs;

  dofs_left  = getLeft()->getDofs().size();
  dofs_right = getRight()->getDofs().size();

  dofs_to_elim_left  = getLeft()->getDofsToElim();
  dofs_to_elim_right = getRight()->getDofsToElim();

  for (int j = dofs_to_elim_left; j < dofs_left; ++j) {
    for (int i = dofs_to_elim_left; i < dofs_left; ++i) {
      M[leftPlaces[j - dofs_to_elim_left]][leftPlaces[i - dofs_to_elim_left]] =
          M_left[j][i];
    }
    RHS[leftPlaces[j - dofs_to_elim_left]] = RHS_left[j];
  }
  for (int j = dofs_to_elim_right; j < dofs_right; ++j) {
    for (int i = dofs_to_elim_right; i < dofs_right; ++i) {
      M[rightPlaces[j - dofs_to_elim_right]]
       [rightPlaces[i - dofs_to_elim_right]] += M_right[j][i];
    }
    RHS[rightPlaces[j - dofs_to_elim_right]] += RHS_right[j];
  }
}

void Node::eliminate() const {
  if (left != NULL && right != NULL) {
    // this->mergeProduction(left->system->matrix, left->system->rhs,
    //                      right->system->matrix, right->system->rhs,
    //                      this->system->matrix, this->system->rhs);
    this->merge();
  } else {
    // this->preprocessProduction();
    this->fillin();
  }
  system->eliminate(getDofsToElim());
}

void Node::bs() const {
  // system->backwardSubstitute(this->getDofsToElim());
  /*for (int i=0; i<system->n; ++i) {
      if (fabs(system->rhs[i]-1.0) > 1e-8) {
          printf("WRONG SOLUTION - [%lu] %d: %lf\n", this->getId(), i,
  system->rhs[i]);
      }
  }*/
}

bool Node::isNeighbour(Node* node, Node* parent) {
  if (parent == NULL) {
    return true;
  }
  if (node == NULL) {
    return true;
  }

  auto getAllDOFs = [](Node* n) {
    std::set<uint64_t>* dofs = new std::set<uint64_t>();
    for (Element* e : n->getElements()) {
      for (uint64_t dof : e->dofs)
        dofs->insert(dof);
    }
    return dofs;
  };

  std::set<uint64_t>* nodeDofs   = getAllDOFs(node);
  std::set<uint64_t>* parentDofs = getAllDOFs(parent);

  if (nodeDofs->empty() || parentDofs->empty()) {
    delete nodeDofs;
    delete parentDofs;
    return false;
  }

  auto nodeIt    = nodeDofs->begin();
  auto nodeItEnd = nodeDofs->end();

  auto parentIt    = parentDofs->begin();
  auto parentItEnd = parentDofs->end();

  // sets are internally sorted, so if the beginning of the one is greater
  // than end of other - we can finish
  if ((*parentIt > *nodeDofs->rbegin()) || (*nodeIt > *parentDofs->rbegin())) {
    delete nodeDofs;
    delete parentDofs;
    return false;
  }

  while (nodeIt != nodeItEnd && parentIt != parentItEnd) {
    if (*nodeIt == *parentIt) {
      delete nodeDofs;
      delete parentDofs;
      return true; // common element? yes, we are neighbours!
    } else if (*nodeIt > *parentIt) {
      ++parentIt;
    } else {
      ++nodeIt;
    }
  }

  delete nodeDofs;
  delete parentDofs;
  return false;
};

bool Node::isNeighbour(Element* element1, Element* element2) {
  if (element1 == NULL) {
    return true;
  }
  if (element2 == NULL) {
    return true;
  }

  std::set<uint64_t> element1Dofs(element1->dofs.cbegin(),
                                  element1->dofs.cend());
  std::set<uint64_t> element2Dofs(element2->dofs.cbegin(),
                                  element2->dofs.cend());

  if (element1Dofs.empty() || element2Dofs.empty()) {
    return false;
  }

  auto element1It    = element1Dofs.begin();
  auto element1ItEnd = element1Dofs.end();

  auto element2It    = element2Dofs.begin();
  auto element2ItEnd = element2Dofs.end();

  // sets are internally sorted, so if the beginning of the one is greater
  // than end of other - we can finish
  if ((*element2It > *element1Dofs.rbegin()) ||
      (*element1It > *element2Dofs.rbegin())) {
    return false;
  }

  while (element1It != element1ItEnd && element2It != element2ItEnd) {
    if (*element1It == *element2It) {
      return true; // common element? yes, we are neighbours!
    } else if (*element1It > *element2It) {
      ++element2It;
    } else {
      ++element1It;
    }
  }
  return false; // no common elements => no neighbourhood
};

int Node::getNumberOfNeighbours(std::vector<Element*>& allElements) {
  int common = 0;
  for (Element* e1 : allElements) {
    for (Element* e2 : this->mergedElements) {
      if (Node::isNeighbour(e1, e2)) {
        common++;
      }
    }
  }
  common -= this->mergedElements.size();
  return common;
}

void Node::rebuildElements() {
  this->clearElements();

  if (this->getLeft() != NULL) {
    for (Element* e : this->getLeft()->getElements()) {
      this->addElement(e);
    }
  }
  if (this->getRight() != NULL) {
    for (Element* e : this->getRight()->getElements()) {
      this->addElement(e);
    }
  }
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

int Node::isBalanced(bool* bal) {
  int h;

  if (this->getLeft()) {
    this->l = this->getLeft()->isBalanced(bal);
  } else {
    this->l = 0;
  }

  if (this->getRight()) {
    this->r = this->getRight()->isBalanced(bal);
  } else {
    this->r = 0;
  }

  h = this->r - this->l;
  if ((h >= 2) || (h <= -2)) {
    (*bal) = false;
  }

  if (this->r > this->l) {
    return this->r + 1;
  } else {
    return this->l + 1;
  }
}

bool Node::isLooped(std::set<Node*>* n) {
  if (n->count(this)) {
    return true;
  }
  n->insert(this);
  if ((this->getLeft()) && (this->getLeft()->isLooped(n))) {
    return true;
  }
  if ((this->getRight()) && (this->getRight()->isLooped(n))) {
    return true;
  }
  return false;
}

unsigned long Node::getSizeInMemory() {
  unsigned long total = this->dofs.size() * this->dofs.size() * sizeof(double) +
                        this->dofs.size() * sizeof(double*);
  if (left != NULL && right != NULL) {
    total += left->getSizeInMemory() + right->getSizeInMemory();
  }
  return total;
}

unsigned long Node::getFLOPs() {
  auto flops = [](unsigned int a, unsigned int b) {
    return a * (6 * b * b - 6 * a * b + 6 * b + 2 * a * a - 3 * a + 1) / 6;
  };

  unsigned long total = flops(this->getDofsToElim(), this->getDofs().size());
  if (left != NULL && right != NULL) {
    total += left->getFLOPs() + right->getFLOPs();
  }

  return total;
}

unsigned long Node::getMemoryRearrangements() {
  unsigned long total = 0;

  if (left != NULL && right != NULL) {
    unsigned long memLeft = (left->getDofs().size() - left->getDofsToElim());
    memLeft *= memLeft;
    unsigned long memRight = (right->getDofs().size() - right->getDofsToElim());
    memRight *= memRight;

    total = memLeft + memRight + left->getMemoryRearrangements() +
            right->getMemoryRearrangements();
  }

  return total;
}

/*END OF DEBUG*/
