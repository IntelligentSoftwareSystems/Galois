#include "Aig.h"

#include <fstream>
#include <unordered_map>

namespace andInverterGraph {

Aig::Aig() {

}

Aig::~Aig() {

}

Graph & Aig::getGraph() {
	return this->graph;
}

std::vector< GNode > & Aig::getInputNodes() {
	return this->inputNodes;
}

std::vector< GNode > & Aig::getLatchNodes() {
	return this->latchNodes;
}

std::vector< GNode > & Aig::getOutputNodes() {
	return this->outputNodes;
}

int Aig::getNumInputs() {
	return this->inputNodes.size();
}

int Aig::getNumLatches() {
	return this->latchNodes.size();
}

int Aig::getNumOutputs() {
	return this->outputNodes.size();
}

int Aig::getNumAnds() {
	int nNodes = std::distance( this->graph.begin(), this->graph.end() );
	return ( nNodes - ( getNumInputs() + getNumLatches() + getNumOutputs() + 1 ) ); // +1 is to disconsider the constant node.
}

std::vector< std::string > & Aig::getInputNames() {
	return this->inputNames;
}

void Aig::setInputNames( std::vector< std::string > inputNames ) {
	this->inputNames = inputNames;
}

std::vector< std::string > & Aig::getLatchNames() {
	return this->latchNames;
}

void Aig::setLatchNames( std::vector< std::string > latchNames ) {
	this->latchNames = latchNames;
}

std::vector< std::string > & Aig::getOutputNames() {
	return this->outputNames;
}

void Aig::setOutputNames( std::vector< std::string > outputNames ) {
	this->outputNames = outputNames;
}

std::string Aig::getDesignName() {
	return this->designName;
}

void Aig::setDesignName( std::string designName ) {
	this->designName = designName;
}

void Aig::resetAndIDs() {

	std::stack< GNode > stack;

	computeTopologicalSort( stack );

	int currentID = this->getNumInputs() + this->getNumLatches() + 1;

	while ( !stack.empty() ) {
		GNode node = stack.top();
		stack.pop();
		NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		//std::cout << nodeData.id << " -> ";		
		nodeData.id = currentID++;
		//std::cout << nodeData.id << std::endl;
	}

	std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::computeTopologicalSort( std::stack< GNode > & stack ) {

	int size = this->getNumInputs() + this->getNumLatches() + this->getNumAnds();
	std::vector< bool > visited( size );

	for ( auto node : this->graph ) {
		
		NodeData & nodeData = this->graph.getData( node, galois::MethodFlag::READ );
       		
		if ( nodeData.type == NodeType::AND ) { 
			visited[ nodeData.id ] = false;
		}
	}
	
	for ( auto pi : this->inputNodes ) {
		
		for ( auto outEdge : this->graph.out_edges( pi ) ) {

			GNode node = this->graph.getEdgeDst( outEdge );
			NodeData & nodeData = this->graph.getData( node, galois::MethodFlag::READ );

			if ( (!visited[ nodeData.id ]) && (nodeData.type == NodeType::AND) ) {
	        		topologicalSort( node, visited, stack );
			}
		}
	} 
}

void Aig::topologicalSort( GNode node, std::vector< bool > & visited, std::stack< GNode > & stack ) {

	NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
	visited[ nodeData.id ] = true;
 	
	for ( auto outEdge : this->graph.out_edges( node ) ) {
		
		GNode nextNode = this->graph.getEdgeDst( outEdge );
		NodeData & nextNodeData = this->graph.getData( nextNode, galois::MethodFlag::READ );

		if ( (!visited[ nextNodeData.id ]) && (nextNodeData.type == NodeType::AND) ) {
			 topologicalSort( nextNode, visited, stack );
		}
 	}
    
	stack.push( node );
}

std::string Aig::toDot() {

	// Preprocess PI and PO names
	std::unordered_map< int, std::string > piNames;	
	for ( int i = 0; i < this->inputNodes.size(); i++ ) {
		aig::NodeData & nodeData = graph.getData( this->inputNodes[i], galois::MethodFlag::READ );
		piNames.insert( std::make_pair( nodeData.id, this->inputNames[i]) );
	}

	std::unordered_map< int, std::string > poNames;	
	for ( int i = 0; i < this->outputNodes.size(); i++ ) {
		aig::NodeData & nodeData = graph.getData( this->outputNodes[i], galois::MethodFlag::READ );
		poNames.insert( std::make_pair( nodeData.id, this->outputNames[i]) );
	}

	std::stringstream dot, inputs, outputs, ands, edges;

	for ( auto node : this->graph ) {

		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );

		// Write Edges
		for ( auto edge : graph.in_edges( node ) ) {
			aig::GNode dstNode = graph.getEdgeDst( edge );
			aig::NodeData & dstData = graph.getData( dstNode, galois::MethodFlag::READ );
			bool polarity = graph.getEdgeData( edge, galois::MethodFlag::READ );			

			std::string nodeName, dstName;

			if ( nodeData.type == NodeType::PI ) {
				nodeName = piNames[ nodeData.id ];
			}
			else {
				if ( nodeData.type == NodeType::PO ) {
					nodeName = poNames[ nodeData.id ];
				}
				else {
					nodeName = std::to_string( nodeData.id );
				}
			}

			if ( dstData.type == NodeType::PI ) {
				dstName = piNames[ dstData.id ];
			}
			else {
				if ( dstData.type == NodeType::PO ) {
					dstName = poNames[ dstData.id ];
				}
				else {
					dstName = std::to_string( dstData.id );
				}
			}

			edges << "\"" << dstName << "\" -> \"" << nodeName << "\"";

			if ( polarity ) {
				edges << " [penwidth = 3, color=blue]" << std::endl;
			}
			else {
				edges << " [penwidth = 3, color=red, style=dashed]" << std::endl;
			}
		}

		if ( nodeData.type == NodeType::PI ) {
			inputs << "\"" << piNames[ nodeData.id ] << "\"";
			inputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#ff8080\", fontsize=20]" << std::endl;
			continue;
		}
		
		if ( nodeData.type == NodeType::PO ) {
			outputs << "\"" << poNames[ nodeData.id ] << "\"";
			outputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#008080\", fontsize=20]" << std::endl;	
			continue;
		}
	
		if ( nodeData.type == NodeType::AND ) {
			ands << "\"" << nodeData.id << "\"";
			ands << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#ffffff\", fontsize=20]" << std::endl;		
		}
	}
	
	dot << "digraph aig {" << std::endl;
	dot << "ranksep=1.5;" << std::endl;
	dot << "nodesep=1.5;" << std::endl;
	dot << inputs.str();
	dot << ands.str();
	dot << outputs.str();
	dot << edges.str();
	dot << "{ rank=source;";
	for ( GNode node: this->inputNodes ) {
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		dot << " \"" << piNames[ nodeData.id ] << "\"";
	}
	dot << " }" << std::endl;
	
	dot << "{ rank=sink;";
	for ( GNode node: this->outputNodes ) {
	 	aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
	 	dot << " \"" << poNames[ nodeData.id ] << "\"";
	}
	dot << " }" << std::endl;
	
	dot << "rankdir=\"BT\"" << std::endl;
	dot << "}" << std::endl;

	return dot.str();
}

void Aig::writeDot( std::string path, std::string dotText ) {

	std::ofstream dotFile;
	dotFile.open( path );
	dotFile << dotText;
	dotFile.close();
}

} /* namespace aig */
