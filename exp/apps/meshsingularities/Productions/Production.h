#ifndef PRODUCTION_H
#define PRODUCTION_H

#include "Node.h"
#include "Vertex.h"
#include "EProduction.hxx"
#include "EquationSystem.h"

#include <vector>

#include "Galois/Graph/LC_Morph_Graph.h"

typedef int EdgeData;

typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData> Graph;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::GraphNode GraphNode;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::iterator LCM_iterator;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::edge_iterator LCM_edge_iterator;

class AbstractProduction {
  private:
	virtual void generateGraph() = 0;

	Vertex *S;
	Graph *graph;

	std::vector<EquationSystem*> *inputData;
	int leafs;

  public:
	AbstractProduction(std::vector<int>* productionParameters,
					   int leafs,
					   std::vector<EquationSystem*> *inputData) : leafs(leafs), inputData(inputData) {
		generateGraph();
  	};

	virtual ~AbstractProduction() {
		delete graph;
		delete S;
	}

	virtual void Execute(EProduction productionToExecute, Vertex* v, EquationSystem* input) = 0;

	Vertex *getRootVertex();
	Graph *getGraph();

};

#endif
