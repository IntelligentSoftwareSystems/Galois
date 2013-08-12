/*
 * GraphGenerator.cxx
 *
 *  Created on: Aug 6, 2013
 *      Author: dgoik
 */

#include "GraphGenerator.hxx"

GraphNode GraphGenerator::addNode(int nr_of_incoming_edges, EProduction production,
		GraphNode src_graph_node, GraphNode dst_graph_node, int nr_of_outgoing_edges,
		Vertex *v, EquationSystem *system)
{

	Node node(nr_of_incoming_edges, production, productions, v, system);
	GraphNode graph_node = graph->createNode(nr_of_outgoing_edges,node);

	if(src_graph_node == NULL)
		graph->addEdge(graph_node,dst_graph_node,Galois::MethodFlag::NONE);
	else
		graph->addEdge(src_graph_node,graph_node,Galois::MethodFlag::NONE);

	return graph_node;
}

void GraphGenerator::generateGraph(int nr_of_leafs, AbstractProduction *productions, std::vector<EquationSystem*> *inputData)
{
	if(nr_of_leafs < 2)
		throw std::runtime_error("At least 2 leafs required");

	this->productions = productions;
	this->inputData = inputData;

	graph = new Graph();

	S = new Vertex(NULL, NULL, NULL, ROOT, productions->getInterfaceSize()*3);
	Node eroot_node(2,EProduction::EROOT, productions, S, NULL);
	GraphNode eroot_graph_node = graph->createNode(2,eroot_node);
	GraphNode new_graph_node = addNode(2,EProduction::A2,NULL,eroot_graph_node,1, S, NULL);
	recursiveGraphGeneration(nr_of_leafs,0,nr_of_leafs-1,eroot_graph_node, new_graph_node, S);

}

void GraphGenerator::recursiveGraphGeneration(int nr_of_leafs, int low_range, int high_range,
		GraphNode backward_substitution_src_node, GraphNode merging_dst_node,
		Vertex *parent)
{
	GraphNode new_graph_node;
	GraphNode new_bs_graph_node;

	Vertex *left;
	Vertex *right;


	if((high_range - low_range) > 2)
	{

		left = new Vertex(NULL, NULL, parent, NODE, productions->getInterfaceSize()*3);
		right = new Vertex(NULL, NULL, parent, NODE, productions->getInterfaceSize()*3);

		parent->setLeft(left);
		parent->setRight(right);

		//left elimination
		new_graph_node = addNode(1,EProduction::E,NULL,merging_dst_node,1, left, NULL);
		//left merging
		new_graph_node = addNode(2,EProduction::A2,NULL,new_graph_node,1, left, NULL);
		//left bs
		new_bs_graph_node = addNode(1,EProduction::BS,backward_substitution_src_node,NULL,2, left, NULL);
		//left subtree generation

		recursiveGraphGeneration(nr_of_leafs,low_range,low_range + (high_range-low_range)/2,new_bs_graph_node,new_graph_node, left);

		//right elimination
		new_graph_node = addNode(1,EProduction::E,NULL,merging_dst_node,1, right, NULL);
		//right merging
		new_graph_node = addNode(2,EProduction::A2,NULL,new_graph_node,1, right, NULL);
		//right bs
		new_bs_graph_node = addNode(1,EProduction::BS,backward_substitution_src_node,NULL,2, right, NULL);
		//right subtree generation

		recursiveGraphGeneration(nr_of_leafs,low_range + (high_range-low_range)/2 + 1,high_range,new_bs_graph_node,new_graph_node,right);
	}
	//only 3 leafs remaining
	else if((high_range - low_range) == 2)
	{
		//first leaf
		//leaf creation
		if(low_range == 0) {
			left = new Vertex(NULL, NULL, parent, LEAF, productions->getA1Size());
			addNode(0,EProduction::A1,NULL,merging_dst_node,1, left, inputData->at(low_range));
		}
		else {
			left = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize());
			addNode(0,EProduction::A,NULL,merging_dst_node,1, left, inputData->at(low_range));
		}
		//leaf bs
		addNode(1,EProduction::BS,backward_substitution_src_node,NULL,0, left, NULL);

		//second and third leaf
		//elimination

		Vertex *node = new Vertex(NULL, NULL, parent, NODE, productions->getInterfaceSize()*3);

		parent->setLeft(left);
		parent->setRight(node);

		new_graph_node = addNode(1,EProduction::E,NULL,merging_dst_node,1, node, NULL);
		//merging
		new_graph_node = addNode(2,EProduction::A2,NULL,new_graph_node,1, node, NULL);
		//bs
		new_bs_graph_node = addNode(1,EProduction::BS,backward_substitution_src_node,NULL,2, node, NULL);

		//left leaf creation
		left = new Vertex(NULL, NULL, node, LEAF, productions->getLeafSize());
		addNode(0,EProduction::A,NULL,new_graph_node,1,left, inputData->at(low_range+1));
		//right leaf creation
		if(high_range == nr_of_leafs - 1) {
			right = new Vertex(NULL, NULL, node, LEAF, productions->getANSize());
			addNode(0,EProduction::AN,NULL,new_graph_node,1, right, inputData->at(low_range+2));
		}
		else{
			right = new Vertex(NULL, NULL, node, LEAF, productions->getLeafSize());
			addNode(0,EProduction::A,NULL,new_graph_node,1, right, inputData->at(low_range+2));
		}

		node->setLeft(left);
		node->setRight(right);

		//left leaf bs
		addNode(1,EProduction::BS,new_bs_graph_node,NULL,0, left, NULL);
		//right leaf bs
		addNode(1,EProduction::BS,new_bs_graph_node,NULL,0, right, NULL);


	}
	//two leafs remaining
	else if((high_range - low_range) == 1)
	{
		if (low_range == 0) {
			left = new Vertex(NULL, NULL, parent, NODE, productions->getA1Size());
		} else {
			left = new Vertex(NULL, NULL, parent, NODE, productions->getLeafSize());
		}

		if (high_range == nr_of_leafs - 1) {
			right = new Vertex(NULL, NULL, parent, NODE, productions->getANSize());
		} else {
			right = new Vertex(NULL, NULL, parent, NODE, productions->getLeafSize());
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
			addNode(0,EProduction::A1,NULL,new_graph_node,1, left,  inputData->at(low_range));
		else
			addNode(0,EProduction::A,NULL,new_graph_node,1, left,  inputData->at(low_range));
		//right leaf
		if(high_range == nr_of_leafs - 1)
			addNode(0,EProduction::AN,NULL,new_graph_node,1, right, inputData->at(high_range));
		else
			addNode(0,EProduction::A,NULL,new_graph_node,1, right, inputData->at(high_range));

		//left leaf bs
		addNode(1,EProduction::BS,backward_substitution_src_node,NULL,0, left, NULL);
		//right leaf bs
		addNode(1,EProduction::BS,backward_substitution_src_node,NULL,0, right, NULL);
	}

}

Graph *GraphGenerator::getGraph()
{
	return this->graph;
}

