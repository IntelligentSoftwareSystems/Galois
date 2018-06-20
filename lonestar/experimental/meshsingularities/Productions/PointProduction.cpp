/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/*
 * PointProduction.cpp
 *
 *  Created on: Sep 12, 2013
 *      Author: dgoik
 */

#include "PointProduction.hxx"

GraphNode PointProduction::addNode(int incomingEdges, int outgoingEdges,
                                   int leafNumber, EProduction production,
                                   GraphNode src, GraphNode dst, Vertex* v,
                                   EquationSystem* system) {

  Node node(incomingEdges, production, this, v, system);
  GraphNode graph_node = graph->createNode(outgoingEdges, node);

  if (src == NULL)
    graph->addEdge(graph_node, dst, galois::MethodFlag::UNPROTECTED);
  else
    graph->addEdge(src, graph_node, galois::MethodFlag::UNPROTECTED);

  return graph_node;
}

void PointProduction::generateGraph() {
  int leafs = inputData->size();
  if (leafs < 2)
    throw std::runtime_error("At least 2 leafs required");

  graph = new Graph();

  this->S = new Vertex(NULL, NULL, NULL, ROOT, getInterfaceSize() * 3);

  Node eroot_node(2, EProduction::A2ROOT, this, S, NULL);

  GraphNode eroot_graph_node = graph->createNode(1, eroot_node);
  GraphNode bs_graph_node =
      addNode(1, 2, 0xff, EProduction::BS, eroot_graph_node, NULL, S, NULL);

  recursiveGraphGeneration(0, leafs - 1, bs_graph_node, eroot_graph_node, S);
}

void PointProduction::recursiveGraphGeneration(int low_range, int high_range,
                                               GraphNode bsSrcNode,
                                               GraphNode mergingDstNode,
                                               Vertex* parent) {
  GraphNode new_graph_node;
  GraphNode new_bs_graph_node;

  Vertex* left;
  Vertex* right;

  if ((high_range - low_range) > 2) {

    left  = new Vertex(NULL, NULL, parent, NODE, this->getInterfaceSize() * 3);
    right = new Vertex(NULL, NULL, parent, NODE, this->getInterfaceSize() * 3);

    parent->setLeft(left);
    parent->setRight(right);

    // left elimination
    new_graph_node = addNode(2, 1, 0xff, EProduction::A2NODE, NULL,
                             mergingDstNode, left, NULL);
    new_bs_graph_node =
        addNode(1, 2, 0xff, EProduction::BS, bsSrcNode, NULL, left, NULL);
    // left subtree generation

    recursiveGraphGeneration(low_range,
                             low_range + (high_range - low_range) / 2,
                             new_bs_graph_node, new_graph_node, left);

    // right elimination
    new_graph_node = addNode(2, 1, 0xff, EProduction::A2NODE, NULL,
                             mergingDstNode, right, NULL);
    new_bs_graph_node =
        addNode(1, 2, 0xff, EProduction::BS, bsSrcNode, NULL, right, NULL);
    // right subtree generation

    recursiveGraphGeneration(low_range + (high_range - low_range) / 2 + 1,
                             high_range, new_bs_graph_node, new_graph_node,
                             right);
  }
  // only 3 leafs remaining
  else if ((high_range - low_range) == 2) {
    int leafs = inputData->size();
    // first leaf
    // leaf creation
    if (low_range == 0) {
      left = new Vertex(NULL, NULL, parent, LEAF, this->getLeafSize());
      addNode(0, 1, 0, EProduction::A1, NULL, mergingDstNode, left,
              inputData->at(low_range));
    } else {
      left = new Vertex(NULL, NULL, parent, LEAF, this->getLeafSize());
      addNode(0, 1, low_range, EProduction::A, NULL, mergingDstNode, left,
              inputData->at(low_range));
    }
    // leaf bs
    addNode(1, 0, 0xff, EProduction::BS, bsSrcNode, NULL, left, NULL);

    // second and third leaf
    // elimination

    Vertex* node =
        new Vertex(NULL, NULL, parent, NODE, this->getInterfaceSize() * 3);

    parent->setLeft(left);
    parent->setRight(node);

    new_graph_node = addNode(2, 1, 0xff, EProduction::A2NODE, NULL,
                             mergingDstNode, node, NULL);
    // bs
    new_bs_graph_node =
        addNode(1, 2, 0xff, EProduction::BS, bsSrcNode, NULL, node, NULL);

    // left leaf creation
    left = new Vertex(NULL, NULL, node, LEAF, this->getLeafSize());
    addNode(0, 1, low_range + 1, EProduction::A, NULL, new_graph_node, left,
            inputData->at(low_range + 1));
    // right leaf creation
    if (high_range == leafs - 1) {
      right = new Vertex(NULL, NULL, node, LEAF, this->getLeafSize());
      addNode(0, 1, low_range + 2, EProduction::AN, NULL, new_graph_node, right,
              inputData->at(low_range + 2));
    } else {
      right = new Vertex(NULL, NULL, node, LEAF, this->getLeafSize());
      addNode(0, 1, low_range + 2, EProduction::A, NULL, new_graph_node, right,
              inputData->at(low_range + 2));
    }

    node->setLeft(left);
    node->setRight(right);

    // left leaf bs
    addNode(1, 0, 0xff, EProduction::BS, new_bs_graph_node, NULL, left, NULL);
    // right leaf bs
    addNode(1, 0, 0xff, EProduction::BS, new_bs_graph_node, NULL, right, NULL);

  }
  // two leafs remaining
  else if ((high_range - low_range) == 1) {
    int leafs = inputData->size();
    // left = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize(),
    // low_range/4);
    left = new Vertex(NULL, NULL, parent, LEAF, this->getLeafSize());

    // right = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize(),
    // high_range/4);
    right = new Vertex(NULL, NULL, parent, LEAF, this->getLeafSize());

    parent->setLeft(left);
    parent->setRight(right);

    // elimination and merging already finished at previous level
    new_graph_node = mergingDstNode;

    // leaf creation
    // left leaf
    if (low_range == 0)
      addNode(0, 1, low_range, EProduction::A1, NULL, new_graph_node, left,
              inputData->at(low_range));
    else
      addNode(0, 1, low_range, EProduction::A, NULL, new_graph_node, left,
              inputData->at(low_range));
    // right leaf
    if (high_range == leafs - 1)
      addNode(0, 1, high_range, EProduction::AN, NULL, new_graph_node, right,
              inputData->at(high_range));
    else
      addNode(0, 1, high_range, EProduction::A, NULL, new_graph_node, right,
              inputData->at(high_range));

    // left leaf bs
    addNode(1, 0, 0xff, EProduction::BS, bsSrcNode, NULL, left, NULL);
    // right leaf bs
    addNode(1, 0, 0xff, EProduction::BS, bsSrcNode, NULL, right, NULL);
  }
}

void PointProduction::Execute(EProduction productionToExecute, Vertex* v,
                              EquationSystem* input) {
  switch (productionToExecute) {
  case EProduction::A1:
    A1(v, input);
    break;
  case EProduction::A:
    A(v, input);
    break;
  case EProduction::AN:
    AN(v, input);
    break;
  case EProduction::A2NODE:
    A2Node(v);
    break;
  case EProduction::A2ROOT:
    A2Root(v);
    // v->system->print();
    break;
  case EProduction::BS:
    BS(v);
    // v->system->print();
    break;
  default:
    printf("Invalid production!\n");
    break;
  }
}

Vertex* PointProduction::getRootVertex() { return S; }

Graph* PointProduction::getGraph() { return graph; }

void PointProduction::A1(Vertex* v, EquationSystem* inData) const {
  // here we assumed, that inData is currently pre-processed
  double** const sys_matrix = v->system->matrix;
  double* const sys_rhs     = v->system->rhs;

  const double** const in_matrix = (const double**)inData->matrix;
  const double* const in_rhs     = (const double*)inData->rhs;

  if (offset > 0) {
    for (int i = 0; i < interfaceSize; ++i) {
      for (int j = 0; j < interfaceSize; ++j) {
        sys_matrix[i + offset][j + offset] =
            in_matrix[i + a1Offset][j + a1Offset]; // 1
        sys_matrix[i + offset][j + offset + interfaceSize] =
            in_matrix[i + a1Offset][j + offset + interfaceSize + a1Offset]; // 3
        sys_matrix[i + offset + interfaceSize][j + offset] =
            in_matrix[i + offset + interfaceSize + a1Offset][j + a1Offset]; // 7
        sys_matrix[i + offset + interfaceSize][j + offset + interfaceSize] =
            in_matrix[i + offset + interfaceSize + a1Offset]
                     [j + offset + interfaceSize + a1Offset]; // 9
      }
      sys_rhs[i + offset] = in_rhs[i + a1Offset];
      sys_rhs[i + offset + interfaceSize] =
          in_rhs[i + offset + interfaceSize + a1Offset];
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < offset; ++j) {
        sys_matrix[i][j] = in_matrix[i + interfaceSize + a1Offset]
                                    [j + interfaceSize + a1Offset]; // 5
      }
      sys_rhs[i] = in_rhs[i + interfaceSize + a1Offset];
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < interfaceSize; ++j) {
        sys_matrix[i][j + offset] =
            in_matrix[i + interfaceSize + a1Offset][j + a1Offset]; // 4
        sys_matrix[i][j + offset + interfaceSize] =
            in_matrix[i + interfaceSize + a1Offset]
                     [j + offset + interfaceSize + a1Offset]; // 6
        sys_matrix[j + offset][i] =
            in_matrix[j + a1Offset][i + interfaceSize + a1Offset]; // 2
        sys_matrix[j + offset + interfaceSize][i] =
            in_matrix[j + offset + interfaceSize + a1Offset]
                     [i + interfaceSize + a1Offset]; // 8
      }
    }

    v->system->eliminate(leafSize - 2 * interfaceSize);
  } else {
    // just copy data?

    for (int i = 0; i < leafSize; ++i) {
      for (int j = 0; j < leafSize; ++j) {
        sys_matrix[i][j] = in_matrix[i][j];
      }
      sys_rhs[i] = in_rhs[i];
    }
  }
}

void PointProduction::A(Vertex* v, EquationSystem* inData) const {
  // You will probably need to overwrite this method
  // to adjust it to your requirements.

  double** const sys_matrix = v->system->matrix;
  double* const sys_rhs     = v->system->rhs;

  const double** const in_matrix = (const double**)inData->matrix;
  const double* const in_rhs     = (const double*)inData->rhs;

  if (offset > 0) {
    for (int i = 0; i < interfaceSize; ++i) {
      for (int j = 0; j < interfaceSize; ++j) {
        sys_matrix[i + offset][j + offset] = in_matrix[i][j]; // 1
        sys_matrix[i + offset][j + offset + interfaceSize] =
            in_matrix[i][j + offset + interfaceSize]; // 3
        sys_matrix[i + offset + interfaceSize][j + offset] =
            in_matrix[i + offset + interfaceSize][j]; // 7
        sys_matrix[i + offset + interfaceSize][j + offset + interfaceSize] =
            in_matrix[i + offset + interfaceSize]
                     [j + offset + interfaceSize]; // 9
      }
      sys_rhs[i + offset]                 = inData->rhs[i];
      sys_rhs[i + offset + interfaceSize] = in_rhs[i + offset + interfaceSize];
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < offset; ++j) {
        sys_matrix[i][j] = in_matrix[i + interfaceSize][j + interfaceSize]; // 5
      }
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < interfaceSize; ++j) {
        sys_matrix[i][j + offset] = in_matrix[i + interfaceSize][j]; // 4
        sys_matrix[i][j + offset + interfaceSize] =
            in_matrix[i + interfaceSize][j + offset + interfaceSize]; // 6
        sys_matrix[j + offset][i] = in_matrix[j][i + interfaceSize];  // 2
        sys_matrix[j + offset + interfaceSize][i] =
            in_matrix[j + offset + interfaceSize][i + interfaceSize]; // 8
      }
      sys_rhs[i] = in_rhs[i + interfaceSize];
    }

    v->system->eliminate(offset);

  } else {
    for (int i = 0; i < interfaceSize * 2; ++i) {
      for (int j = 0; j < interfaceSize * 2; ++j) {
        sys_matrix[i][j] = in_matrix[i][j];
      }
      sys_rhs[i] = in_rhs[i];
    }
  }
}

void PointProduction::AN(Vertex* v, EquationSystem* inData) const {
  double** const sys_matrix = v->system->matrix;
  double* const sys_rhs     = v->system->rhs;

  const double** const in_matrix = (const double**)inData->matrix;
  const double* const in_rhs     = (const double*)inData->rhs;

  if (offset > 0) {
    for (int i = 0; i < interfaceSize; ++i) {
      for (int j = 0; j < interfaceSize; ++j) {
        sys_matrix[i + offset][j + offset] =
            in_matrix[i + anOffset][j + anOffset]; // 1
        sys_matrix[i + offset][j + offset + interfaceSize] =
            in_matrix[i + anOffset][j + offset + interfaceSize + anOffset]; // 3
        sys_matrix[i + offset + interfaceSize][j + offset] =
            in_matrix[i + offset + interfaceSize + anOffset][j + anOffset]; // 7
        sys_matrix[i + offset + interfaceSize][j + offset + interfaceSize] =
            in_matrix[i + offset + interfaceSize + anOffset]
                     [j + offset + interfaceSize + anOffset]; // 9
      }
      sys_rhs[i + offset] = in_rhs[i + anOffset];
      sys_rhs[i + offset + interfaceSize] =
          in_rhs[i + offset + interfaceSize + anOffset];
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < offset; ++j) {
        sys_matrix[i][j] = in_matrix[i + interfaceSize + anOffset]
                                    [j + interfaceSize + anOffset]; // 5
      }
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < interfaceSize; ++j) {
        sys_matrix[i][j + offset] =
            in_matrix[i + interfaceSize + anOffset][j + anOffset]; // 4
        sys_matrix[i][j + offset + interfaceSize] =
            in_matrix[i + interfaceSize + anOffset]
                     [j + offset + interfaceSize + anOffset]; // 6
        sys_matrix[j + offset][i] =
            in_matrix[j + anOffset][i + interfaceSize + anOffset]; // 2
        sys_matrix[j + offset + interfaceSize][i] =
            in_matrix[j + offset + interfaceSize + anOffset]
                     [i + interfaceSize + anOffset]; // 8
      }
      sys_rhs[i] = in_rhs[i + interfaceSize + anOffset];
    }

    v->system->eliminate(offset);
  } else {
    // just copy data?

    for (int i = 0; i < leafSize; ++i) {
      for (int j = 0; j < leafSize; ++j) {
        sys_matrix[i][j] = in_matrix[i][j];
      }
      sys_rhs[i] = in_rhs[i];
    }
  }
}

void PointProduction::A2Node(Vertex* v) const {
  const int offsetLeft =
      v->left->type == LEAF ? leafSize - 2 * interfaceSize : interfaceSize;
  const int offsetRight =
      v->right->type == LEAF ? leafSize - 2 * interfaceSize : interfaceSize;

  double** const sys_matrix = v->system->matrix;
  double* const sys_rhs     = v->system->rhs;

  const double** const left_matrix  = (const double**)v->left->system->matrix;
  const double** const right_matrix = (const double**)v->right->system->matrix;
  const double* const left_rhs      = (const double*)v->left->system->rhs;
  const double* const right_rhs     = (const double*)v->right->system->rhs;

  for (int i = 0; i < this->interfaceSize; ++i) {
    for (int j = 0; j < this->interfaceSize; ++j) {
      // x: left y: top
      sys_matrix[i][j] = left_matrix[i + offsetLeft + interfaceSize]
                                    [j + offsetLeft + interfaceSize] +
                         right_matrix[i + offsetRight][j + offsetRight];

      // x: center y: top
      sys_matrix[i][j + interfaceSize] =
          left_matrix[i + offsetLeft + interfaceSize][j + offsetLeft];

      // x: left y:center
      sys_matrix[i + interfaceSize][j] =
          left_matrix[i + offsetLeft][j + offsetLeft + interfaceSize];

      // x: center y:center
      sys_matrix[i + interfaceSize][j + interfaceSize] =
          left_matrix[i + offsetLeft][j + offsetLeft];

      // x: bottom y: bottom
      sys_matrix[i + 2 * interfaceSize][j + 2 * interfaceSize] =
          right_matrix[i + offsetRight + interfaceSize]
                      [j + offsetRight + interfaceSize];

      // x: left y:bottom
      sys_matrix[i + 2 * interfaceSize][j] =
          right_matrix[i + offsetRight + interfaceSize][j + offsetRight];

      // x: right y: top
      sys_matrix[i][j + 2 * interfaceSize] =
          right_matrix[i + offsetRight][j + offsetRight + interfaceSize];
    }
    sys_rhs[i] =
        left_rhs[i + offsetLeft + interfaceSize] + right_rhs[i + offsetRight];
    sys_rhs[i + interfaceSize]     = left_rhs[i + offsetLeft];
    sys_rhs[i + 2 * interfaceSize] = right_rhs[i + offsetRight + interfaceSize];
  }

  v->system->eliminate(this->interfaceSize);
}

void PointProduction::A2Root(Vertex* v) const {
  const int offsetLeft =
      v->left->type == LEAF ? leafSize - 2 * interfaceSize : interfaceSize;
  const int offsetRight =
      v->right->type == LEAF ? leafSize - 2 * interfaceSize : interfaceSize;

  double** const sys_matrix = v->system->matrix;
  double* const sys_rhs     = v->system->rhs;

  const double** const left_matrix  = (const double**)v->left->system->matrix;
  const double** const right_matrix = (const double**)v->right->system->matrix;
  const double* const left_rhs      = (const double*)v->left->system->rhs;
  const double* const right_rhs     = (const double*)v->right->system->rhs;

  for (int i = 0; i < this->interfaceSize; ++i) {
    for (int j = 0; j < this->interfaceSize; ++j) {
      // x: left y: top
      sys_matrix[i][j] = left_matrix[i + offsetLeft + interfaceSize]
                                    [j + offsetLeft + interfaceSize] +
                         right_matrix[i + offsetRight][j + offsetRight];

      // x: center y: top
      sys_matrix[i][j + interfaceSize] =
          left_matrix[i + offsetLeft + interfaceSize][j + offsetLeft];

      // x: left y:center
      sys_matrix[i + interfaceSize][j] =
          left_matrix[i + offsetLeft][j + offsetLeft + interfaceSize];

      // x: center y:center
      sys_matrix[i + interfaceSize][j + interfaceSize] =
          left_matrix[i + offsetLeft][j + offsetLeft];

      // x: bottom y: bottom
      sys_matrix[i + 2 * interfaceSize][j + 2 * interfaceSize] =
          right_matrix[i + offsetRight + interfaceSize]
                      [j + offsetRight + interfaceSize];

      // x: left y:bottom
      sys_matrix[i + 2 * interfaceSize][j] =
          right_matrix[i + offsetRight + interfaceSize][j + offsetRight];

      // x: right y: top
      sys_matrix[i][j + 2 * interfaceSize] =
          right_matrix[i + offsetRight][j + offsetRight + interfaceSize];
    }
    sys_rhs[i] =
        left_rhs[i + offsetLeft + interfaceSize] + right_rhs[i + offsetRight];
    sys_rhs[i + interfaceSize]     = left_rhs[i + offsetLeft];
    sys_rhs[i + 2 * interfaceSize] = right_rhs[i + offsetRight + interfaceSize];
  }

  v->system->eliminate(this->interfaceSize * 3);
}

void PointProduction::BS(Vertex* v) const {
  if (v->type == ROOT) {
    v->system->backwardSubstitute(this->interfaceSize * 3 - 1);
  } else if (v->type == NODE) {
    const int offsetA =
        v->parent->left == v ? interfaceSize * 2 : interfaceSize;
    const int offsetB =
        v->parent->left == v ? interfaceSize : interfaceSize * 2;

    double** const sys_matrix = v->system->matrix;
    double* const sys_rhs     = v->system->rhs;

    const double* const par_rhs = (const double*)v->parent->system->rhs;

    for (int i = interfaceSize; i < 3 * interfaceSize; ++i) {
      for (int j = i; j < 3 * interfaceSize; ++j) {
        sys_matrix[i][j] = i == j ? 1.0 : 0.0;
      }
    }

    for (int i = 0; i < this->interfaceSize; ++i) {
      sys_rhs[i + offsetA] = par_rhs[i];
      sys_rhs[i + offsetB] = par_rhs[i + offsetB];
    }

    v->system->backwardSubstitute(interfaceSize - 1);
  } else if (v->type == LEAF) {
    const int offsetA = v->parent->left == v ? interfaceSize : 0;
    const int offsetB = v->parent->left == v ? 0 : 2 * interfaceSize;

    double** const sys_matrix = v->system->matrix;
    double* const sys_rhs     = v->system->rhs;

    const double* const par_rhs = (const double*)v->parent->system->rhs;

    for (int i = leafSize - 2 * interfaceSize; i < leafSize; ++i) {
      for (int j = i; j < leafSize; ++j) {
        sys_matrix[i][j] = (i == j) ? 1.0 : 0.0;
      }
    }

    for (int i = 0; i < this->interfaceSize; ++i) {
      sys_rhs[i + offset]                 = par_rhs[i + offsetA];
      sys_rhs[i + offset + interfaceSize] = par_rhs[i + offsetB];
    }

    v->system->backwardSubstitute(leafSize - 2 * interfaceSize);
  }
}

int PointProduction::getInterfaceSize() const { return this->interfaceSize; }

int PointProduction::getLeafSize() const { return this->leafSize; }

int PointProduction::getA1Size() const { return this->a1Size; }

int PointProduction::getANSize() const { return this->anSize; }

EquationSystem* PointProduction::preprocessA1(EquationSystem* input) const {
  EquationSystem* system =
      new EquationSystem(input->matrix, input->rhs, input->n);

  system->eliminate(this->getA1Size() - this->getLeafSize());

  return (system);
}

EquationSystem* PointProduction::preprocessA(EquationSystem* input) const {
  return new EquationSystem(input->matrix, input->rhs, input->n);
}

EquationSystem* PointProduction::preprocessAN(EquationSystem* input) const {
  EquationSystem* system    = new EquationSystem(this->getANSize());
  double** const tierMatrix = input->matrix;
  double* const tierRhs     = input->rhs;

  double** const sys_matrix = system->matrix;
  double* const sys_rhs     = system->rhs;

  const int leafSize = this->getLeafSize();
  const int offset   = this->getANSize() - leafSize;
  if (offset > 0) {
    for (int i = 0; i < leafSize; ++i) {
      for (int j = 0; j < leafSize; ++j) {
        sys_matrix[i + offset][j + offset] = tierMatrix[i][j];
      }
      sys_rhs[i + offset] = tierRhs[i];
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < leafSize; ++j) {
        sys_matrix[i][j + offset] = tierMatrix[i + leafSize][j];
        sys_matrix[j + offset][i] = tierMatrix[j][i + leafSize];
      }
    }

    for (int i = 0; i < offset; ++i) {
      for (int j = 0; j < offset; ++j) {
        sys_matrix[i][j] = tierMatrix[i + leafSize][j + leafSize];
      }
      sys_rhs[i] = tierRhs[i + leafSize];
    }

    system->eliminate(offset);
  }
  return (system);
}

void PointProduction::postprocessA1(Vertex* leaf, EquationSystem* outputData,
                                    std::vector<double>* result,
                                    int num) const {
  EquationSystem* a1 = outputData;

  int offset     = this->getA1Size() - this->getLeafSize();
  int leafOffset = this->getLeafSize() - 2 * this->getInterfaceSize();

  if (offset > 0) {
    for (int j = 0; j < this->getLeafSize(); ++j) {
      for (int k = 0; k < this->getLeafSize(); ++k) {
        a1->matrix[j + offset][k + offset] = j == k ? 1.0 : 0.0;
      }
    }

    for (int j = 0; j < this->getInterfaceSize(); ++j) {
      a1->rhs[j + offset] = leaf->system->rhs[j + leafOffset];
      a1->rhs[j + offset + leafOffset + this->getInterfaceSize()] =
          leaf->system->rhs[j + leafOffset + this->getInterfaceSize()];
    }

    for (int j = 0; j < leafOffset; ++j) {
      a1->rhs[j + offset + this->getInterfaceSize()] = leaf->system->rhs[j];
    }

    a1->backwardSubstitute(offset);
    for (int j = 0; j < this->getA1Size(); ++j) {
      (*result)[j] = a1->rhs[j];
    }
  }
}

void PointProduction::postprocessA(Vertex* leaf, EquationSystem* outputData,
                                   std::vector<double>* result, int num) const {
  const int leafOffset = this->getLeafSize() - 2 * this->getInterfaceSize();
  const int totalOffset =
      (this->getA1Size() - this->getInterfaceSize()) +
      (num - 1) * (this->getLeafSize() - this->getInterfaceSize());

  for (int j = 0; j < this->getInterfaceSize(); ++j) {
    (*result)[totalOffset + j + leafOffset + this->getInterfaceSize()] =
        leaf->system->rhs[j + leafOffset + this->getInterfaceSize()];
  }

  for (int j = 0; j < leafOffset; ++j) {
    (*result)[totalOffset + j + this->getInterfaceSize()] =
        leaf->system->rhs[j];
  }
}

void PointProduction::postprocessAN(Vertex* leaf, EquationSystem* outputData,
                                    std::vector<double>* result,
                                    int num) const {

  const int leafOffset = this->getLeafSize() - 2 * this->getInterfaceSize();
  const int totalOffset =
      (this->getA1Size() - this->getInterfaceSize()) +
      (num - 1) * (this->getLeafSize() - this->getInterfaceSize());
  const int offset = this->getANSize() - this->getLeafSize();

  EquationSystem* an = outputData;

  for (int j = 0; j < this->getLeafSize(); ++j) {
    for (int k = 0; k < this->getLeafSize(); ++k) {
      an->matrix[j + offset][k + offset] = j == k ? 1.0 : 0.0;
    }
  }

  for (int j = 0; j < this->getInterfaceSize(); ++j) {
    an->rhs[j + offset] = leaf->system->rhs[j + leafOffset];
    an->rhs[j + offset + leafOffset + this->getInterfaceSize()] =
        leaf->system->rhs[j + leafOffset + this->getInterfaceSize()];
  }

  for (int j = 0; j < leafOffset; ++j) {
    an->rhs[j + offset + this->getInterfaceSize()] = leaf->system->rhs[j];
  }

  an->backwardSubstitute(offset);

  for (int j = 0; j < this->getLeafSize(); ++j) {
    (*result)[j + totalOffset] = an->rhs[j + offset];
  }

  for (int j = 0; j < offset; ++j) {
    (*result)[j + totalOffset + this->getLeafSize()] = an->rhs[j];
  }
}

std::vector<Vertex*>* PointProduction::collectLeafs(Vertex* p) {
  std::vector<Vertex*>* left   = NULL;
  std::vector<Vertex*>* right  = NULL;
  std::vector<Vertex*>* result = NULL;

  if (p == NULL) {
    return NULL;
  }

  result = new std::vector<Vertex*>();

  if (p != NULL && p->right == NULL && p->left == NULL) {
    result->push_back(p);
    return result;
  }

  if (p != NULL && p->left != NULL) {
    left = collectLeafs(p->left);
  }

  if (p != NULL && p->right != NULL) {
    right = collectLeafs(p->right);
  }

  if (left != NULL) {
    for (std::vector<Vertex*>::iterator it = left->begin(); it != left->end();
         ++it) {
      if (*it != NULL) {
        result->push_back(*it);
      }
    }
    delete left;
  }

  if (right != NULL) {
    for (std::vector<Vertex*>::iterator it = right->begin(); it != right->end();
         ++it) {
      if (*it != NULL) {
        result->push_back(*it);
      }
    }
    delete right;
  }

  return result;
}

std::vector<EquationSystem*>*
PointProduction::preprocess(std::vector<EquationSystem*>* input) const {
  std::vector<EquationSystem*>* outputVector =
      new std::vector<EquationSystem*>();
  std::vector<EquationSystem*>::iterator it = input->begin();

  for (int i = 0; it != input->end(); ++it, ++i) {
    if (i == 0) {
      outputVector->push_back(this->preprocessA1(*it));
    } else if (i == input->size() - 1) {
      outputVector->push_back(this->preprocessAN(*it));
    } else {
      outputVector->push_back(this->preprocessA(*it));
    }
  }
  return outputVector;
}

std::vector<double>* PointProduction::getResult() {
  std::vector<Vertex*>* leafs = collectLeafs(S);
  std::vector<double>* result = new std::vector<double>(
      (leafs->size() - 2) * getLeafSize() + getA1Size() + getANSize() -
      (leafs->size() - 1) * getInterfaceSize());

  for (int i = 0; i < leafs->size(); ++i) {
    if (i == 0) {
      this->postprocessA1(leafs->at(i), inputData->at(i), result, i);
    } else if (i == leafs->size() - 1) {
      this->postprocessAN(leafs->at(i), inputData->at(i), result, i);
    } else {
      this->postprocessA(leafs->at(i), inputData->at(i), result, i);
    }
  }

  delete leafs;

  return result;
}
