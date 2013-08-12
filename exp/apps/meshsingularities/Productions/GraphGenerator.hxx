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
#include "Production.h"
#include "Vertex.h"
#include "EquationSystem.h"
#include <vector>
#include "Preprocessor.h"

struct Node;

typedef int EdgeData;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData> Graph;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::GraphNode GraphNode;

class GraphGenerator {
public:
	static int id;

	GraphGenerator()
	{
		id=0;
		S=NULL;
		graph=NULL;
		edge_data=0;
	}

	virtual ~GraphGenerator()
	{
		delete S;
	}

	void generateGraph(int nr_of_leafs, AbstractProduction &prod, std::vector<EquationSystem*> *inputData);
	Graph *getGraph();

private:

	EdgeData edge_data = 0;
	Graph* graph;
	Vertex *S;

	void recursiveGraphGeneration(int nr_of_leafs,
			int low_range, int high_range,
			GraphNode backward_substitution_src_node,
			GraphNode merging_dst_node,
			AbstractProduction &productions,
			std::vector<EquationSystem*> *inputData,
			Vertex *parent);

	GraphNode addNode(int nr_of_incoming_edges,
			EProduction production,
			GraphNode src,
			GraphNode dst,
			int nr_of_outgoing_edges,
			AbstractProduction &productions,
			Vertex *v,
			EquationSystem *input);
};

struct Node {
	int x;
	int nr_of_incoming_edges;
	EProduction productionToExecute;
	AbstractProduction &productions;
	Vertex *v;
	EquationSystem *input;
	Node(int nr_of_incoming_edges, EProduction production,
			AbstractProduction &prod, Vertex *v, EquationSystem *input):
				nr_of_incoming_edges(nr_of_incoming_edges),
				productionToExecute(production),
				productions(prod), v(v),
				input(input)

	{
		x = GraphGenerator::id++;

	};
};




#endif /* GRAPHGENERATOR_HXX_ */
