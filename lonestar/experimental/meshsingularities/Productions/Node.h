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

#include "Galois/Graph/LC_Morph_Graph.h"

class AbstractProduction;

class Node
{
private:
	int number;
	EProduction productionToExecute;
	AbstractProduction *productions;
	Vertex *v;
	EquationSystem *input;

public:
	int incomingEdges;
	Node(int incomingEdges,
		 EProduction production,
		 AbstractProduction *prod,
		 Vertex *v,
		 EquationSystem *input):
			 incomingEdges(incomingEdges),
			 productionToExecute(production),
			 productions(prod), v(v),
			 input(input)

	{
    }

    void setVertex(Vertex *v1) { this->v = v1; }
    void execute();
};

typedef int EdgeData;

typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData> Graph;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::GraphNode GraphNode;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::iterator LCM_iterator;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::edge_iterator LCM_edge_iterator;


#endif /* NODE_H_ */
