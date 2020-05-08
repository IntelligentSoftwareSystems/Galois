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

/*
 * Node.h
 *
 *  Created on: Aug 30, 2013
 *      Author: kjopek
 */

#ifndef NODE_H_
#define NODE_H_

#include "EProduction.hxx"
#include "EquationSystem.h"
#include "Vertex.h"

#include "galois/Graph/LC_Morph_Graph.h"

class AbstractProduction;

class Node {
private:
  int number;
  EProduction productionToExecute;
  AbstractProduction* productions;
  Vertex* v;
  EquationSystem* input;

public:
  int incomingEdges;
  Node(int incomingEdges, EProduction production, AbstractProduction* prod,
       Vertex* v, EquationSystem* input)
      : incomingEdges(incomingEdges), productionToExecute(production),
        productions(prod), v(v), input(input)

  {}

  void setVertex(Vertex* v1) { this->v = v1; }
  void execute();
};

typedef int EdgeData;

typedef galois::graphs::LC_Morph_Graph<Node, EdgeData> Graph;
typedef galois::graphs::LC_Morph_Graph<Node, EdgeData>::GraphNode GraphNode;
typedef galois::graphs::LC_Morph_Graph<Node, EdgeData>::iterator LCM_iterator;
typedef galois::graphs::LC_Morph_Graph<Node, EdgeData>::edge_iterator
    LCM_edge_iterator;

#endif /* NODE_H_ */
