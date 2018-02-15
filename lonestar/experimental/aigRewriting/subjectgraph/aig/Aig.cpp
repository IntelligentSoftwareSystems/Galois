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

std::vector< GNode > & Aig::getNodes() {
	return this->nodes;
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

int Aig::getDepth() {

	resetAllIds();

	int max = -1;

	for ( auto po : this->outputNodes ) {
		NodeData & poData = this->graph.getData( po, galois::MethodFlag::READ );
		if ( max < poData.level ) {
			max = poData.level;
		}
	}

	assert( max > -1 );

	return max;
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

std::vector< std::pair< int, int > > & Aig::getNodesTravId() {
	return this->nodesTravId;
}

void Aig::registerTravId( int nodeId, int threadId, int travId ) {
	this->nodesTravId[ nodeId ].first = threadId;
	this->nodesTravId[ nodeId ].second = travId;
}

bool Aig::lookupTravId( int nodeId, int threadId, int travId ) {
	if ( (this->nodesTravId[ nodeId ].first == threadId) && (this->nodesTravId[ nodeId ].second == travId) ) {
		return true; 
	}
	else {
		return false;
	}
}

std::unordered_multimap< unsigned, GNode > & Aig::getFanoutMap( int nodeId ) {
	return this->nodesFanoutMap[ nodeId ];
}

std::vector< std::unordered_multimap< unsigned, GNode > > & Aig::getNodesFanoutMap() {
	return this->nodesFanoutMap;
}

GNode Aig::getConstZero() {
	return this->nodes[0];
}

void Aig::insertNodeInFanoutMap( GNode andNode, GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol ) {

	NodeData & lhsNodeData = this->graph.getData( lhsNode, galois::MethodFlag::READ );
	NodeData & rhsNodeData = this->graph.getData( rhsNode, galois::MethodFlag::READ );

	unsigned key = makeAndHashKey( lhsNode, rhsNode, lhsNodeData.id, rhsNodeData.id, lhsPol, rhsPol );

	if ( lhsNodeData.id < rhsNodeData.id ) {
		this->nodesFanoutMap[ lhsNodeData.id ].emplace( key, andNode );
	}
	else {
		this->nodesFanoutMap[ rhsNodeData.id ].emplace( key, andNode );
	}
}

void Aig::removeNodeInFanoutMap( GNode removedNode, GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol ) {

	GNode lhsInNode;
	GNode rhsInNode;
	bool lhsInNodePol;
	bool rhsInNodePol;
	int smallestId;

	NodeData & lhsNodeData = this->graph.getData( lhsNode, galois::MethodFlag::READ );
	NodeData & rhsNodeData = this->graph.getData( rhsNode, galois::MethodFlag::READ );

	unsigned key = makeAndHashKey( lhsNode, rhsNode, lhsNodeData.id, rhsNodeData.id, lhsPol, rhsPol );
	
	if ( lhsNodeData.id < rhsNodeData.id ) {
		smallestId = lhsNodeData.id;
	}
	else {
		smallestId = rhsNodeData.id;
	}

	std::unordered_multimap< unsigned, GNode > & fanoutMap = this->nodesFanoutMap[ smallestId ];
	auto range = fanoutMap.equal_range( key );

	for ( auto it = range.first; it != range.second;  ) {

		GNode fanoutNode = it->second;
		NodeData & fanoutNodeData = this->graph.getData( fanoutNode, galois::MethodFlag::READ );

		if ( fanoutNodeData.type != NodeType::AND ) {
			it++;
			continue;
		}

		auto inEdge = this->graph.in_edge_begin( fanoutNode );
		lhsInNode = this->graph.getEdgeDst( inEdge );
		lhsInNodePol = this->graph.getEdgeData( inEdge );

		if ( lhsInNode == lhsNode ) {
			inEdge++;
			rhsInNode = this->graph.getEdgeDst( inEdge );
			rhsInNodePol = this->graph.getEdgeData( inEdge );
		}
		else {
			rhsInNode = lhsInNode;
			rhsInNodePol = lhsInNodePol;
			inEdge++;
			lhsInNode = this->graph.getEdgeDst( inEdge );
			lhsInNodePol = this->graph.getEdgeData( inEdge );
		}

		if ( (lhsInNode == lhsNode) && (lhsInNodePol == lhsPol) && (rhsInNode == rhsNode) && (rhsInNodePol == rhsPol) && (fanoutNode == removedNode ) ) {
			it = fanoutMap.erase( it );
		}
		else {
			it++;
		}
	}
}

GNode Aig::lookupNodeInFanoutMap( GNode lhsNode, GNode rhsNode, bool lhsPol, bool rhsPol ) {

	GNode lhsInNode;
	GNode rhsInNode;
	bool lhsInNodePol;
	bool rhsInNodePol;
	int smallestId;

	NodeData & lhsNodeData = this->graph.getData( lhsNode, galois::MethodFlag::READ );
	NodeData & rhsNodeData = this->graph.getData( rhsNode, galois::MethodFlag::READ );

	unsigned key = makeAndHashKey( lhsNode, rhsNode, lhsNodeData.id, rhsNodeData.id, lhsPol, rhsPol );

	if ( lhsNodeData.id < rhsNodeData.id ) {
		smallestId = lhsNodeData.id;
	}
	else {
		smallestId = rhsNodeData.id;
	}

	std::unordered_multimap< unsigned, GNode > & fanoutMap = this->nodesFanoutMap[ smallestId ];
	auto range = fanoutMap.equal_range( key );
	
	for ( auto it = range.first; it != range.second; it++ ) {

		GNode fanoutNode = it->second;
		NodeData & fanoutNodeData = this->graph.getData( fanoutNode, galois::MethodFlag::READ );

		if ( fanoutNodeData.type != NodeType::AND ) {	
			continue;
		}

		auto inEdge = this->graph.in_edge_begin( fanoutNode );
		lhsInNode = this->graph.getEdgeDst( inEdge );
		lhsInNodePol = this->graph.getEdgeData( inEdge );

		if ( lhsInNode == lhsNode ) {
			inEdge++;
			rhsInNode = this->graph.getEdgeDst( inEdge );
			rhsInNodePol = this->graph.getEdgeData( inEdge );
		}
		else {
			rhsInNode = lhsInNode;
			rhsInNodePol = lhsInNodePol;
			inEdge++;
			lhsInNode = this->graph.getEdgeDst( inEdge );
			lhsInNodePol = this->graph.getEdgeData( inEdge );
			assert( lhsInNode == lhsNode );
		}
		if ( (lhsInNode == lhsNode) && (lhsInNodePol == lhsPol) && (rhsInNode == rhsNode) && (rhsInNodePol == rhsPol) ) {
			return fanoutNode;
		}
	}

	return nullptr;
}

unsigned Aig::makeAndHashKey( GNode lhsNode, GNode rhsNode, int lhsId, int rhsId, bool lhsPol, bool rhsPol ) {

	unsigned key = 0;

	if ( lhsId < rhsId ) {
		key ^= lhsId * 7937;
		key ^= rhsId * 2971;
		key ^= lhsPol ? 911 : 0;
		key ^= rhsPol ? 353 : 0;
	}
	else {
		key ^= rhsId * 7937;
		key ^= lhsId * 2971;
		key ^= rhsPol ? 911 : 0;
		key ^= lhsPol ? 353 : 0;
	}

	return key;
}

void Aig::resetAndIds() {

	std::stack< GNode > stack;

	computeTopologicalSortForAnds( stack );

	int currentId = this->getNumInputs() + this->getNumLatches() + 1;

	while ( !stack.empty() ) {
		GNode node = stack.top();
		stack.pop();
		NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		nodeData.id = currentId++;
		this->nodes[ nodeData.id ] = node;
	}

	//std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::resetAndPIsIds() {

	std::stack< GNode > stack;

	computeTopologicalSortForAnds( stack );

	int currentId = 1;

	for ( GNode pi : this->inputNodes ) {
		NodeData & piData = this->graph.getData( pi, galois::MethodFlag::WRITE );
		piData.id = currentId++;
	}

	while ( !stack.empty() ) {
		GNode node = stack.top();
		stack.pop();
		NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		nodeData.id = currentId++;
		this->nodes[ nodeData.id ] = node;
	}

	//std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::resetAndPOsIds() {

	std::stack< GNode > stack;

	computeTopologicalSortForAnds( stack );

	int currentId = this->getNumInputs() + this->getNumLatches() + 1;

	while ( !stack.empty() ) {
		GNode node = stack.top();
		stack.pop();
		NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		nodeData.id = currentId++;
		this->nodes[ nodeData.id ] = node;
	}

	for ( GNode po : this->outputNodes ) {
		NodeData & poData = this->graph.getData( po, galois::MethodFlag::WRITE );
		poData.id = currentId++;
	}

	//std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::resetAllIds() {

	std::stack< GNode > stack;

	computeTopologicalSortForAnds( stack );

	int currentId = 1;

	for ( GNode pi : this->inputNodes ) {
		NodeData & piData = this->graph.getData( pi, galois::MethodFlag::WRITE );
		piData.id = currentId++;
		piData.level = 0;
	}

	while ( !stack.empty() ) {
		GNode node = stack.top();
		stack.pop();
		NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		nodeData.id = currentId++;

		auto inEdge = this->graph.in_edge_begin( node );
		GNode lhsNode = this->graph.getEdgeDst( inEdge );
		NodeData lhsData = this->graph.getData( lhsNode, galois::MethodFlag::READ );
		inEdge++;
		GNode rhsNode = this->graph.getEdgeDst( inEdge );
		NodeData rhsData = this->graph.getData( rhsNode, galois::MethodFlag::READ );
	
		nodeData.level = 1 + std::max( lhsData.level, rhsData.level );

		this->nodes[ nodeData.id ] = node;
	}

	for ( GNode po : this->outputNodes ) {
		NodeData & poData = this->graph.getData( po, galois::MethodFlag::WRITE );
		poData.id = currentId++;

		auto inEdge = this->graph.in_edge_begin( po );
		GNode inNode = this->graph.getEdgeDst( inEdge );
		NodeData inData = this->graph.getData( inNode, galois::MethodFlag::READ );

		poData.level = inData.level;
	}

	//std::cout << std::endl << "All AND node IDs were reseted!" << std::endl;
}

void Aig::computeTopologicalSortForAll( std::stack< GNode > & stack ) {

	int size = this->nodes.size();
	std::vector< bool > visited( size, false );
	
	for ( GNode pi : this->inputNodes ) {
		for ( auto outEdge : this->graph.out_edges( pi ) ) {
	
			GNode node = this->graph.getEdgeDst( outEdge );
			NodeData & nodeData = this->graph.getData( node, galois::MethodFlag::READ );
	
			if ( !visited[ nodeData.id ] ) {
	        	topologicalSortAll( node, visited, stack );
			}
		}

		stack.push( pi );
	}
}

void Aig::topologicalSortAll( GNode node, std::vector< bool > & visited, std::stack< GNode > & stack ) {

	NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
	visited[ nodeData.id ] = true;
 	
	for ( auto outEdge : this->graph.out_edges( node ) ) {
		
		GNode nextNode = this->graph.getEdgeDst( outEdge );
		NodeData & nextNodeData = this->graph.getData( nextNode, galois::MethodFlag::READ );

		if ( !visited[ nextNodeData.id ] ) {
			 topologicalSortAll( nextNode, visited, stack );
		}
 	}
    
	stack.push( node );
}

void Aig::computeTopologicalSortForAnds( std::stack< GNode > & stack ) {

	int size = this->nodes.size();
	std::vector< bool > visited( size, false );
	
	for ( GNode pi : this->inputNodes ) {
		for ( auto outEdge : this->graph.out_edges( pi ) ) {
	
			GNode node = this->graph.getEdgeDst( outEdge );
			NodeData & nodeData = this->graph.getData( node, galois::MethodFlag::READ );
	
			if ( (!visited[ nodeData.id ]) && (nodeData.type == NodeType::AND) ) {
	        	topologicalSortAnds( node, visited, stack );
			}
		}
	}
}

void Aig::topologicalSortAnds( GNode node, std::vector< bool > & visited, std::stack< GNode > & stack ) {

	NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
	visited[ nodeData.id ] = true;
 	
	for ( auto outEdge : this->graph.out_edges( node ) ) {
		
		GNode nextNode = this->graph.getEdgeDst( outEdge );
		NodeData & nextNodeData = this->graph.getData( nextNode, galois::MethodFlag::READ );

		if ( (!visited[ nextNodeData.id ]) && (nextNodeData.type == NodeType::AND) ) {
			 topologicalSortAnds( nextNode, visited, stack );
		}
 	}
    
	stack.push( node );
}

void Aig::resetAllNodeCounters() {

	for( GNode node : this->graph ) {
		NodeData & nodeData = this->graph.getData( node, galois::MethodFlag::WRITE );
		nodeData.counter = 0;
	}
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
