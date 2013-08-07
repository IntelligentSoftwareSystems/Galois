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


struct Node;

typedef int EdgeData;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData> Graph;
typedef Galois::Graph::LC_Morph_Graph<Node,EdgeData>::GraphNode GraphNode;



class GraphGenerator {
public:
	static int id;
	GraphGenerator()
	{

	}
	virtual ~GraphGenerator()
	{

	}

	void GenerateGraph(int nr_of_leafs);

private:

	EdgeData edge_data = 0;
	Graph* graph;
	void RecursiveGraphGeneration(int nr_of_leafs, int low_range, int high_range,
			GraphNode backward_substitution_src_node, GraphNode merging_dst_node);
	GraphNode AddNode(int nr_of_incoming_edges, EProduction production, GraphNode src, GraphNode dst, int nr_of_outgoing_edges);
};


struct Node {
  int x;
  int nr_of_incoming_edges;
  EProduction production;
  Node(int nr_of_incoming_edges, EProduction production): nr_of_incoming_edges(nr_of_incoming_edges), production(production)
  { x = GraphGenerator::id++; };
};



#endif /* GRAPHGENERATOR_HXX_ */
