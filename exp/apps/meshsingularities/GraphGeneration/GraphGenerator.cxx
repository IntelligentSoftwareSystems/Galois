/*
 * GraphGenerator.cxx
 *
 *  Created on: Aug 6, 2013
 *      Author: dgoik
 */

#include "GraphGenerator.hxx"


int GraphGenerator::id = 0;

GraphNode GraphGenerator::AddNode(int nr_of_incoming_edges, EProduction production,
		GraphNode src_graph_node, GraphNode dst_graph_node, int nr_of_outgoing_edges)
{

	Node node(nr_of_incoming_edges,production);
	GraphNode graph_node = graph->createNode(nr_of_outgoing_edges,node);

	if(src_graph_node == NULL)
		graph->addEdge(graph_node,dst_graph_node,Galois::MethodFlag::NONE);
	else
		graph->addEdge(src_graph_node,graph_node,Galois::MethodFlag::NONE);

	return graph_node;
}

void GraphGenerator::GenerateGraph(int nr_of_leafs)
{
	if(nr_of_leafs < 2)
		throw std::runtime_error("At least 2 leafs required");

	graph = new Graph();

	Node eroot_node(2,EProduction::EROOT);
	GraphNode eroot_graph_node = graph->createNode(2,eroot_node);
	GraphNode new_graph_node = AddNode(2,EProduction::A2,NULL,eroot_graph_node,1);

	RecursiveGraphGeneration(nr_of_leafs,0,nr_of_leafs-1,eroot_graph_node, new_graph_node);


}

void GraphGenerator::RecursiveGraphGeneration(int nr_of_leafs, int low_range, int high_range,
		GraphNode backward_substitution_src_node, GraphNode merging_dst_node)
{
	GraphNode new_graph_node;
	GraphNode new_bs_graph_node;

	if((high_range - low_range) > 2)
	{

		//left elimination
		new_graph_node = AddNode(1,EProduction::E,NULL,merging_dst_node,1);
		//left merging
		new_graph_node = AddNode(2,EProduction::A2,NULL,new_graph_node,1);
		//left bs
		new_bs_graph_node = AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,2);
		//left subtree generation

		RecursiveGraphGeneration(nr_of_leafs,low_range,low_range + (high_range-low_range)/2,new_bs_graph_node,new_graph_node);

		//right elimination
		new_graph_node = AddNode(1,EProduction::E,NULL,merging_dst_node,1);
		//right merging
		new_graph_node = AddNode(2,EProduction::A2,NULL,new_graph_node,1);
		//right bs
		new_bs_graph_node = AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,2);
		//right subtree generation

		RecursiveGraphGeneration(nr_of_leafs,low_range + (high_range-low_range)/2 + 1,high_range,new_bs_graph_node,new_graph_node);
	}
	//only 3 leafs remaining
	else if((high_range - low_range) == 2)
	{
		//first leaf
		//leaf creation
		if(low_range == 0)
			AddNode(0,EProduction::A1,NULL,merging_dst_node,1);
		else
			AddNode(0,EProduction::A,NULL,merging_dst_node,1);

		//leaf bs
		AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,0);

		//second and third leaf
		//elimination
		new_graph_node = AddNode(1,EProduction::E,NULL,merging_dst_node,1);
		//merging
		new_graph_node = AddNode(2,EProduction::A2,NULL,new_graph_node,1);
		//bs
		new_bs_graph_node = AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,2);

		//left leaf creation
		AddNode(0,EProduction::A,NULL,new_graph_node,1);
		//right leaf creation
		if(high_range == nr_of_leafs - 1)
			AddNode(0,EProduction::AN,NULL,new_graph_node,1);
		else
			AddNode(0,EProduction::A,NULL,new_graph_node,1);

		//left leaf bs
		AddNode(1,EProduction::BS,new_bs_graph_node,NULL,0);
		//right leaf bs
		AddNode(1,EProduction::BS,new_bs_graph_node,NULL,0);


	}
	//two leafs remaining
	else if((high_range - low_range) == 1)
	{

		//elimination and merging already finished at previous level
		new_graph_node = merging_dst_node;

		//bs
		//new_bs_graph_node = AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,2);
		//leaf creation
		//left leaf
		if(low_range == 0)
			AddNode(0,EProduction::A1,NULL,new_graph_node,1);
		else
			AddNode(0,EProduction::A,NULL,new_graph_node,1);
		//right leaf
		if(high_range == nr_of_leafs - 1)
			AddNode(0,EProduction::AN,NULL,new_graph_node,1);
		else
			AddNode(0,EProduction::A,NULL,new_graph_node,1);

		//left leaf bs
		AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,0);
		//right leaf bs
		AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,0);
	}

}



