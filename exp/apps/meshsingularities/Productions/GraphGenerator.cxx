/*
 * GraphGenerator.cxx
 *
 *  Created on: Aug 6, 2013
 *      Author: dgoik
 */

#include "GraphGenerator.hxx"
#include "Node.h"

GraphNode GraphGenerator::addNode(int incomingEdges,
		int outgoingEdges,
		int leafNumber,
		EProduction production,
		GraphNode src,
		GraphNode dst,
		Vertex *v,
		EquationSystem *system)
{

	Node node(incomingEdges, production, productions, v, system);
	GraphNode graph_node = graph->createNode(outgoingEdges,node);

	if(src == NULL)
		graph->addEdge(graph_node,dst,Galois::MethodFlag::NONE);
	else
		graph->addEdge(src,graph_node,Galois::MethodFlag::NONE);

	return graph_node;
}

Vertex *GraphGenerator::generateGraph(int leafs, PointProduction *productions, std::vector<EquationSystem*> *inputData)
{
	this->leafs = leafs;

	if(leafs < 2)
		throw std::runtime_error("At least 2 leafs required");

	this->productions = productions;
	this->inputData = inputData;

	graph = new Graph();

	S = new Vertex(NULL, NULL, NULL, ROOT, productions->getInterfaceSize()*3);
	Node eroot_node(2,EProduction::A2ROOT, productions, S, NULL);
	GraphNode eroot_graph_node = graph->createNode(1,eroot_node);
	GraphNode bs_graph_node = addNode(1, 2, 0xff, EProduction::BS, eroot_graph_node, NULL, S, NULL);
	recursiveGraphGeneration(0,leafs-1,bs_graph_node, eroot_graph_node, S);
	return S;
}

void GraphGenerator::recursiveGraphGeneration(int low_range,
		int high_range,
		GraphNode bsSrcNode,
		GraphNode mergingDstNode,
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
		new_graph_node = addNode(2, 1, 0xff, EProduction::A2NODE, NULL, mergingDstNode, left, NULL);
		new_bs_graph_node = addNode(1, 2, 0xff, EProduction::BS, bsSrcNode, NULL, left, NULL);
		//left subtree generation

		recursiveGraphGeneration(low_range,
				low_range + (high_range-low_range)/2,
				new_bs_graph_node,
				new_graph_node,
				left);

		//right elimination
		new_graph_node = addNode(2, 1, 0xff, EProduction::A2NODE, NULL,mergingDstNode, right, NULL);
		new_bs_graph_node = addNode(1, 2, 0xff, EProduction::BS, bsSrcNode, NULL, right, NULL);
		//right subtree generation

		recursiveGraphGeneration(low_range + (high_range-low_range)/2 + 1,high_range,new_bs_graph_node,new_graph_node,right);
	}
	//only 3 leafs remaining
	else if((high_range - low_range) == 2)
	{
		//first leaf
		//leaf creation
		if(low_range == 0) {
			left = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize());
			addNode(0, 1, 0, EProduction::A1, NULL, mergingDstNode, left, inputData->at(low_range));
		}
		else {
			left = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize());
			addNode(0, 1, low_range, EProduction::A, NULL, mergingDstNode, left, inputData->at(low_range));
		}
		//leaf bs
		addNode(1, 0, 0xff, EProduction::BS, bsSrcNode, NULL, left, NULL);

		//second and third leaf
		//elimination

		Vertex *node = new Vertex(NULL, NULL, parent, NODE, productions->getInterfaceSize()*3);

		parent->setLeft(left);
		parent->setRight(node);

		new_graph_node = addNode(2, 1, 0xff, EProduction::A2NODE, NULL, mergingDstNode, node, NULL);
		//bs
		new_bs_graph_node = addNode(1, 2, 0xff, EProduction::BS, bsSrcNode, NULL, node, NULL);

		//left leaf creation
		left = new Vertex(NULL, NULL, node, LEAF, productions->getLeafSize());
		addNode(0, 1, low_range+1, EProduction::A, NULL, new_graph_node, left, inputData->at(low_range+1));
		//right leaf creation
		if(high_range == leafs - 1) {
			right = new Vertex(NULL, NULL, node, LEAF, productions->getLeafSize());
			addNode(0, 1, low_range+2, EProduction::AN, NULL, new_graph_node, right, inputData->at(low_range+2));
		}
		else{
			right = new Vertex(NULL, NULL, node, LEAF, productions->getLeafSize());
			addNode(0, 1, low_range+2, EProduction::A, NULL, new_graph_node, right, inputData->at(low_range+2));
		}

		node->setLeft(left);
		node->setRight(right);

		//left leaf bs
		addNode(1, 0, 0xff, EProduction::BS, new_bs_graph_node, NULL, left, NULL);
		//right leaf bs
		addNode(1, 0, 0xff, EProduction::BS, new_bs_graph_node, NULL, right, NULL);


	}
	//two leafs remaining
	else if((high_range - low_range) == 1)
	{
		//left = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize(), low_range/4);
		left = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize());

		//right = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize(), high_range/4);
		right = new Vertex(NULL, NULL, parent, LEAF, productions->getLeafSize());

		parent->setLeft(left);
		parent->setRight(right);

		//elimination and merging already finished at previous level
		new_graph_node = mergingDstNode;

		//leaf creation
		//left leaf
		if(low_range == 0)
			addNode(0, 1, low_range, EProduction::A1, NULL, new_graph_node, left,  inputData->at(low_range));
		else
			addNode(0, 1, low_range, EProduction::A, NULL, new_graph_node, left,  inputData->at(low_range));
		//right leaf
		if(high_range == leafs - 1)
			addNode(0, 1, high_range, EProduction::AN, NULL, new_graph_node, right, inputData->at(high_range));
		else
			addNode(0, 1, high_range, EProduction::A, NULL, new_graph_node, right, inputData->at(high_range));

		//left leaf bs
		addNode(1, 0, 0xff, EProduction::BS, bsSrcNode, NULL, left, NULL);
		//right leaf bs
		addNode(1, 0, 0xff, EProduction::BS, bsSrcNode, NULL, right, NULL);
	}

}

Graph *GraphGenerator::getGraph()
{
	return this->graph;
}


// GraphGeneratorQuad

Vertex *GraphGeneratorQuad::generateGraph(int leafs, AbstractProduction *productions, std::vector<EquationSystem*> *inputData)
{
	this->leafs = leafs;

	if(leafs < 2)
		throw std::runtime_error("At least 2 leafs required");

	this->productions = productions;
	this->inputData = inputData;
	graph = new Graph();

	Node rootNode (2, EProduction::MBRoot, productions, NULL, NULL);

	GraphNode mbRoot = graph->createNode(1, rootNode);

	Vertex *vd1 = new Vertex(NULL, NULL, NULL, LEAF, 9);
	Vertex *vd2 = new Vertex(NULL, NULL, NULL, LEAF, 9);

	Vertex *vmd = new Vertex(NULL, NULL, NULL, NODE, 11);

	vmd->setLeft(vd1);
	vmd->setRight(vd2);

	printf("PODMIENIONE D\n");
	Node d1Node(0, EProduction::D, productions, vd1, inputData->at(leafs-2));
	Node d2Node(0, EProduction::D, productions, vd2, inputData->at(leafs-1));

	GraphNode d1GraphNode = graph->createNode(1, d1Node);
	GraphNode d2GraphNode = graph->createNode(1, d2Node);

	Node mdNode(2, EProduction::MD, productions, vmd, NULL);
	GraphNode mdGraphNode = graph->createNode(1, mdNode);

	graph->addEdge(mdGraphNode, mbRoot, Galois::MethodFlag::NONE);

	graph->addEdge(d1GraphNode, mdGraphNode, Galois::MethodFlag::NONE);
	graph->addEdge(d2GraphNode, mdGraphNode, Galois::MethodFlag::NONE);


	Node mbNode (2, EProduction::MB, productions, NULL, NULL);
	GraphNode mbGraphNode = graph->createNode(1, mbNode);
	graph->addEdge(mbGraphNode, mbRoot, Galois::MethodFlag::NONE);

	Node mbc1Node(2, EProduction::MBC, productions, NULL, NULL);
	Node mbc2Node(2, EProduction::MBC, productions, NULL, NULL);

	GraphNode mbc1GraphNode = graph->createNode(1, mbc1Node);
	GraphNode mbc2GraphNode = graph->createNode(1, mbc2Node);

	graph->addEdge(mbc1GraphNode, mbGraphNode, Galois::MethodFlag::NONE);
	graph->addEdge(mbc2GraphNode, mbGraphNode, Galois::MethodFlag::NONE);

	Vertex *mbc1Vertex = recursiveGraphGeneration(0, (leafs-3)/2, mbc1GraphNode);
	Vertex *mbc2Vertex = recursiveGraphGeneration((leafs-3)/2+1, leafs-3, mbc2GraphNode);

	Vertex *vmb = new Vertex(NULL, NULL, NULL, NODE, mbc1Vertex->left->system->n+6);
	vmb->setLeft(mbc1Vertex);
	vmb->setRight(mbc2Vertex);

	mbGraphNode->data.setVertex(vmb);

	Vertex *vmbRoot = new Vertex(NULL, NULL, NULL, NODE, vmb->left->system->n+4);
	vmbRoot->setLeft(vmb);
	vmbRoot->setRight(vmd);

	mbRoot->data.setVertex(vmbRoot);

	mbc1GraphNode->data.setVertex(mbc1Vertex);
	mbc2GraphNode->data.setVertex(mbc2Vertex);

	return vmbRoot;
}

Vertex *GraphGeneratorQuad::recursiveGraphGeneration(int low_range,
		int high_range,
		GraphNode mergingDstNode)
{
	if (high_range - low_range == 3) {
		// bottom part of tree
        Vertex *vc1 = new Vertex(NULL, NULL, NULL, LEAF, 9);
        Vertex *vc2 = new Vertex(NULL, NULL, NULL, LEAF, 9);

        Vertex *vb1 = new Vertex(NULL, NULL, NULL, LEAF, 9);
        Vertex *vb2 = new Vertex(NULL, NULL, NULL, LEAF, 9);

        Vertex *vmc = new Vertex(NULL, NULL, NULL, NODE, 11);
        Vertex *vmb = new Vertex(NULL, NULL, NULL, NODE, 11);

        Vertex *vmbc = new Vertex(NULL, NULL, NULL, NODE, 14);

        vmc->setLeft(vc1);
        vmc->setRight(vc2);

        vmb->setLeft(vb1);
        vmb->setRight(vb2);

        vmbc->setLeft(vmb);
        vmbc->setRight(vmc);

        //Node vmbcNode (2, EProduction::MBC, productions, vmbc, NULL);
        //GraphNode vmbcGraphNode = graph->createNode(1, vmbcNode);

        Node vmcNode(2, EProduction::MC, productions, vmc, NULL);
        Node vmbNode(2, EProduction::MBLeaf, productions, vmb, NULL);


        GraphNode vmbGraphNode = graph->createNode(1, vmbNode);
        GraphNode vmcGraphNode = graph->createNode(1, vmcNode);

        graph->addEdge(vmbGraphNode, mergingDstNode, Galois::MethodFlag::NONE);
        graph->addEdge(vmcGraphNode, mergingDstNode, Galois::MethodFlag::NONE);

        Node vc1Node(0, EProduction::C, productions, vc1, inputData->at(low_range+2));
        Node vc2Node(0, EProduction::C, productions, vc2, inputData->at(low_range+3));


        GraphNode vc1GraphNode = graph->createNode(1, vc1Node);
        GraphNode vc2GraphNode = graph->createNode(1, vc2Node);

        graph->addEdge(vc1GraphNode, vmcGraphNode, Galois::MethodFlag::NONE);
        graph->addEdge(vc2GraphNode, vmcGraphNode, Galois::MethodFlag::NONE);

        Node vb1Node(0, EProduction::B, productions, vb1, inputData->at(low_range));
        Node vb2Node(0, EProduction::B, productions, vb2, inputData->at(low_range+1));

        GraphNode vb1GraphNode = graph->createNode(1, vb1Node);
        GraphNode vb2GraphNode = graph->createNode(1, vb2Node);

        graph->addEdge(vb1GraphNode, vmbGraphNode, Galois::MethodFlag::NONE);
        graph->addEdge(vb2GraphNode, vmbGraphNode, Galois::MethodFlag::NONE);

        return vmbc;

	} else if (high_range - low_range > 3) {
		Vertex *vc1 = new Vertex(NULL, NULL, NULL, LEAF, 9);
		Vertex *vc2 = new Vertex(NULL, NULL, NULL, LEAF, 9);

		Vertex *mc = new Vertex(NULL, NULL, NULL, NODE, 11);

		mc->setLeft(vc1);
		mc->setRight(vc2);

		Node c1Node(0, EProduction::C, productions, vc1, inputData->at(high_range-1));
		Node c2Node(0, EProduction::C, productions, vc2, inputData->at(high_range));

		GraphNode c1GraphNode = graph->createNode(1, c1Node);
		GraphNode c2GraphNode = graph->createNode(1, c2Node);

		Node mcNode(2, EProduction::MC, productions, mc, NULL);
		GraphNode mcGraphNode = graph->createNode(1, mcNode);

		graph->addEdge(mcGraphNode, mergingDstNode, Galois::MethodFlag::NONE);

		graph->addEdge(c1GraphNode, mcGraphNode, Galois::MethodFlag::NONE);
		graph->addEdge(c2GraphNode, mcGraphNode, Galois::MethodFlag::NONE);

		Node mbNode (2, EProduction::MB, productions, NULL, NULL);
		GraphNode mbGraphNode = graph->createNode(1, mbNode);
		graph->addEdge(mbGraphNode, mergingDstNode, Galois::MethodFlag::NONE);

		Node mbc1Node(2, EProduction::MBC, productions, NULL, NULL);
		Node mbc2Node(2, EProduction::MBC, productions, NULL, NULL);

		GraphNode mbc1GraphNode = graph->createNode(1, mbc1Node);
		GraphNode mbc2GraphNode = graph->createNode(1, mbc2Node);

		graph->addEdge(mbc1GraphNode, mbGraphNode, Galois::MethodFlag::NONE);
		graph->addEdge(mbc2GraphNode, mbGraphNode, Galois::MethodFlag::NONE);

		Vertex *mbc1Vertex = recursiveGraphGeneration(low_range, low_range+(high_range-low_range-2)/2, mbc1GraphNode);
		Vertex *mbc2Vertex = recursiveGraphGeneration(low_range+(high_range-low_range-2)/2+1, high_range-2, mbc2GraphNode);

		Vertex *vmb = new Vertex(NULL, NULL, NULL, NODE, mbc1Vertex->left->system->n+6);
		vmb->setLeft(mbc1Vertex);
		vmb->setRight(mbc2Vertex);

		mbGraphNode->data.setVertex(vmb);

		Vertex *vmbc = new Vertex(NULL, NULL, NULL, NODE, vmb->left->system->n+4);
		vmbc->setLeft(vmb);
		vmbc->setRight(mc);

		mbc1GraphNode->data.setVertex(mbc1Vertex);
		mbc2GraphNode->data.setVertex(mbc2Vertex);

		return vmbc;

	} else {
		throw std::string("Invalid range!");
	}
}


GraphNode GraphGeneratorQuad::addNode(int incomingEdges,
		int outgoingEdges,
		VertexType type,
		EProduction production,
		GraphNode src,
		GraphNode dst,
		int eqSystemSize)
{

	Vertex *v = new Vertex(NULL, NULL, NULL, type, eqSystemSize);
	Node node(incomingEdges, production, productions, v, NULL);
	GraphNode graph_node = graph->createNode(outgoingEdges,node);

	if(src == NULL)
		graph->addEdge(graph_node,dst,Galois::MethodFlag::NONE);
	else
		graph->addEdge(src,graph_node,Galois::MethodFlag::NONE);

	return graph_node;
}

Graph *GraphGeneratorQuad::getGraph()
{
	return this->graph;
}

