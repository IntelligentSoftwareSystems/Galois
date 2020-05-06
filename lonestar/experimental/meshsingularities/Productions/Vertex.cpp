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

#include "Vertex.h"

Vertex::Vertex(Vertex* Left, Vertex* Right, Vertex* Parent, VertexType type,
               int systemSize) {
  this->left   = Left;
  this->right  = Right;
  this->parent = Parent;
  this->type   = type;
  this->system = new EquationSystem(systemSize);
}

Vertex::Vertex(Vertex* Left, Vertex* Right, Vertex* Parent, VertexType type,
               int systemSize, int node) {
  this->left   = Left;
  this->right  = Right;
  this->parent = Parent;
  this->type   = type;
  // TODO: restore NUMA support
  //    this->system = new NumaEquationSystem(systemSize, node);
}

Vertex::~Vertex() {

  if (this->left != NULL) {
    delete this->left;
  }
  if (this->right != NULL) {
    delete this->right;
  }

  if (this->system != NULL) {
    delete this->system;
  }
}

void Vertex::setLeft(Vertex* v) { this->left = v; }

void Vertex::setRight(Vertex* v) { this->right = v; }

void Vertex::setType(VertexType t) { this->type = t; }
