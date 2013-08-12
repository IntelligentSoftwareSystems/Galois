/*
 * GraphGenerator.cxx
 *
 *  Created on: Aug 6, 2013
 *      Author: dgoik
 */

#include "GraphGenerator.hxx"

GraphNode GraphGenerator::addNode(int nr_of_incoming_edges, EProduction production,
		GraphNode src_graph_node, GraphNode dst_graph_node, int nr_of_outgoing_edges,
		AbstractProduction &productions, Vertex *v, EquationSystem *input)
{

	Node node(nr_of_incoming_edges, production, productions, v, input);
	GraphNode graph_node = graph->createNode(nr_of_outgoing_edges,node);

	if(src_graph_node == NULL)
		graph->addEdge(graph_node,dst_graph_node,Galois::MethodFlag::NONE);
	else
		graph->addEdge(src_graph_node,graph_node,Galois::MethodFlag::NONE);

	return graph_node;
}

void GraphGenerator::generateGraph(int nr_of_leafs, AbstractProduction &productions, std::vector<EquationSystem*> *inputData)
{
	if(nr_of_leafs < 2)
		throw std::runtime_error("At least 2 leafs required");

	graph = new Graph();

	Node eroot_node(2,EProduction::EROOT);
	S = new Vertex(NULL, NULL, NULL, ROOT, productions.getInterfaceSize()*3);
	GraphNode eroot_graph_node = graph->createNode(2,eroot_node);
	GraphNode new_graph_node = addNode(2,EProduction::A2,NULL,eroot_graph_node,1,productions, S, NULL);
	recursiveGraphGeneration(nr_of_leafs,0,nr_of_leafs-1,eroot_graph_node, new_graph_node, productions, inputData, S);

}

void GraphGenerator::recursiveGraphGeneration(int nr_of_leafs, int low_range, int high_range,
		GraphNode backward_substitution_src_node, GraphNode merging_dst_node,
		AbstractProduction &productions, std::vector<EquationSystem*> *inputData,
		Vertex *parent)
{
	GraphNode new_graph_node;
	GraphNode new_bs_graph_node;

	Vertex *left;
	Vertex *right;


	if((high_range - low_range) > 2)
	{

		left = new Vertex(NULL, NULL, parent, NODE, productions.getInterfaceSize()*3);
		right = new Vertex(NULL, NULL, parent, NODE, productions.getInterfaceSize()*3);

		parent->setLeft(left);
		parent->setRight(right);

		//left elimination
		new_graph_node = addNode(1,EProduction::E,NULL,merging_dst_node,1,productions, left, NULL);
		//left merging
		new_graph_node = addNode(2,EProduction::A2,NULL,new_graph_node,1,productions, left, NULL);
		//left bs
		new_bs_graph_node = addNode(1,EProduction::BS,backward_substitution_src_node,NULL,2,productions, left, NULL);
		//left subtree generation

		recursiveGraphGeneration(nr_of_leafs,low_range,low_range + (high_range-low_range)/2,new_bs_graph_node,new_graph_node,productions,inputData, left);

		//right elimination
		new_graph_node = addNode(1,EProduction::E,NULL,merging_dst_node,1, productions, right, NULL);
		//right merging
		new_graph_node = addNode(2,EProduction::A2,NULL,new_graph_node,1, productions, right, NULL);
		//right bs
		new_bs_graph_node = addNode(1,EProduction::BS,backward_substitution_src_node,NULL,2,productions, right, NULL);
		//right subtree generation

		recursiveGraphGeneration(nr_of_leafs,low_range + (high_range-low_range)/2 + 1,high_range,new_bs_graph_node,new_graph_node,productions,inputData,right);
	}
	//only 3 leafs remaining
	else if((high_range - low_range) == 2)
	{
		//first leaf
		//leaf creation
		if(low_range == 0) {
			left = new Vertex(NULL, NULL, parent, LEAF, productions.getA1Size());
			addNode(0,EProduction::A1,NULL,merging_dst_node,1,productions, left, inputData->at(low_range));
		}
		else {
			left = new Vertex(NULL, NULL, parent, LEAF, productions.getLeafSize());
			addNode(0,EProduction::A,NULL,merging_dst_node,1,productions, left, inputData->at(low_range));
		}
		//leaf bs
		addNode(1,EProduction::BS,backward_substitution_src_node,NULL,0,productions, left, NULL);

		//second and third leaf
		//elimination

		Vertex *node = new Vertex(NULL, NULL, parent, NODE, productions.getInterfaceSize()*3);

		parent->setLeft(left);
		parent->setRight(node);

		new_graph_node = addNode(1,EProduction::E,NULL,merging_dst_node,1,productions, node, NULL);
		//merging
		new_graph_node = addNode(2,EProduction::A2,NULL,new_graph_node,1,productions, node, NULL);
		//bs
		new_bs_graph_node = addNode(1,EProduction::BS,backward_substitution_src_node,NULL,2,productions, node, NULL);

		//left leaf creation
		left = new Vertex(NULL, NULL, node, LEAF, productions.getLeafSize());
		addNode(0,EProduction::A,NULL,new_graph_node,1,productions,left, inputData->at(low_range+1));
		//right leaf creation
		if(high_range == nr_of_leafs - 1) {
			right = new Vertex(NULL, NULL, node, LEAF, productions.getANSize());
			addNode(0,EProduction::AN,NULL,new_graph_node,1,productions, right, inputData->at(low_range+2));
		}
		else{
			right = new Vertex(NULL, NULL, node, LEAF, productions.getLeafSize());
			addNode(0,EProduction::A,NULL,new_graph_node,1,productions, right, inputData->at(low_range+2));
		}

		node->setLeft(left);
		node->setRight(right);

		//left leaf bs
		addNode(1,EProduction::BS,new_bs_graph_node,NULL,0,productions, left, NULL);
		//right leaf bs
		addNode(1,EProduction::BS,new_bs_graph_node,NULL,0,productions, right, NULL);


	}
	//two leafs remaining
	else if((high_range - low_range) == 1)
	{
		if (low_range == 0) {
			left = new Vertex(NULL, NULL, parent, NODE, productions.getA1Size());
		} else {
			left = new Vertex(NULL, NULL, parent, NODE, productions.getLeafSize());
		}

		if (high_range == nr_of_leafs - 1) {
			right = new Vertex(NULL, NULL, parent, NODE, productions.getANSize());
		} else {
			right = new Vertex(NULL, NULL, parent, NODE, productions.getLeafSize());
		}

		parent->setLeft(left);
		parent->setRight(right);

		//elimination and merging already finished at previous level
		new_graph_node = merging_dst_node;

		//bs
		//new_bs_graph_node = AddNode(1,EProduction::BS,backward_substitution_src_node,NULL,2);
		//leaf creation
		//left leaf
		if(low_range == 0)
			addNode(0,EProduction::A1,NULL,new_graph_node,1,productions, left,  inputData->at(low_range));
		else
			addNode(0,EProduction::A,NULL,new_graph_node,1,productions, left,  inputData->at(low_range));
		//right leaf
		if(high_range == nr_of_leafs - 1)
			addNode(0,EProduction::AN,NULL,new_graph_node,1,productions, right, inputData->at(high_range));
		else
			addNode(0,EProduction::A,NULL,new_graph_node,1,productions, right, inputData->at(high_range));

		//left leaf bs
		addNode(1,EProduction::BS,backward_substitution_src_node,NULL,0,productions, left, NULL);
		//right leaf bs
		addNode(1,EProduction::BS,backward_substitution_src_node,NULL,0,productions, right, NULL);
	}

}

Graph *GraphGenerator::getGraph()
{
	return this->graph;
}

