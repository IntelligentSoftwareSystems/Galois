#ifndef PRODUCTION_H
#define PRODUCTION_H

#include "Vertex.h"
#include "EProduction.hxx"
#include "EquationSystem.h"

#include <vector>

#include "Node.h"

class AbstractProduction {
  protected:
	Vertex *S;
	Graph *graph;

	std::vector<EquationSystem*> *inputData;

  public:
	AbstractProduction(std::vector<int>* productionParameters,
					   std::vector<EquationSystem*> *inputData)  {
  	};

	virtual ~AbstractProduction() {
		delete graph;
		delete S;
	}

	virtual void Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input) = 0;

	virtual Vertex *getRootVertex() = 0;
	virtual Graph *getGraph() = 0;

};

#endif
