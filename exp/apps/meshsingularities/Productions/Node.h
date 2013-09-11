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
#include "Production.h"

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
	};

	void setVertex(Vertex *v1) { this->v = v1; }
	void execute();
};


#endif /* NODE_H_ */
