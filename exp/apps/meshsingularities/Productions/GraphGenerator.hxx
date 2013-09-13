/*
 * GraphGenerator.hxx
 *
 *  Created on: Aug 6, 2013
 *      Author: dgoik
 */

#ifndef GRAPHGENERATOR_HXX_
#define GRAPHGENERATOR_HXX_

#include "EProduction.hxx"
#include "Galois/Graph/LC_Morph_Graph.h"
#include "PointProduction.hxx"
#include "Vertex.h"
#include "EquationSystem.h"
#include <vector>
#include "Node.h"

typedef int EdgeData;

typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData> Graph;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::GraphNode GraphNode;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::iterator LCM_iterator;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::edge_iterator LCM_edge_iterator;

class GraphGenerator {
public:
	GraphGenerator() : S(NULL), graph(NULL), edge_data(0), productions(NULL),
					   inputData(NULL), leafs(0) {
	}

	virtual ~GraphGenerator()
	{
		delete S;
	}

	Vertex* generateGraph(int leafs, PointProduction *prod, std::vector<EquationSystem*> *inputData);
	Graph *getGraph();

private:

	EdgeData edge_data = 0;
	Graph* graph;
	Vertex *S;
	PointProduction *productions;
	std::vector<EquationSystem*> *inputData;
	int leafs;

	void recursiveGraphGeneration(int low_range, int high_range,
			GraphNode bsSrcNode,
			GraphNode mergingDstNode,
			Vertex *parent);

	GraphNode addNode(int incomingEdges,
			int outgoingEdges,
			int leafNumber,
			EProduction production,
			GraphNode src,
			GraphNode dst,
			Vertex *v,
			EquationSystem *system);
};

class GraphGeneratorQuad {
private:
	EdgeData edge_data = 0;
	Graph *graph;
	Vertex *S;
	AbstractProduction *productions;
	int leafs;
	std::vector<EquationSystem *> *inputData;

	Vertex *recursiveGraphGeneration(int low_range, int high_range,
			GraphNode mergingDstNode);

	GraphNode addNode(int incomingEdges,
			int outgoingEdges,
			VertexType type,
			EProduction production,
			GraphNode src,
			GraphNode dst,
			int eqSystemSize);

public:
	GraphGeneratorQuad() : S(NULL), graph(NULL), productions(NULL), leafs(0) {

	}

	Vertex *generateGraph(int leafs, AbstractProduction *prod, std::vector<EquationSystem*> *inputData);
	Graph *getGraph();

};
#endif /* GRAPHGENERATOR_HXX_ */
