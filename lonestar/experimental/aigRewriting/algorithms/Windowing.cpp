#include "Windowing.h"

#include <iostream>
#include <set>
#include <unordered_set>
#include <deque>
#include <limits>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <utility>

#define VERBOSE false

namespace algorithm {

Windowing::Windowing( aig::Graph & graph, galois::PerIterAllocTy & allocator, int nInputs, int nOutputs, int nLevels, int cutSizeLimit ) 
					: graph( graph ), allocator( allocator ), window( allocator ), leaves( allocator ),roots( allocator ),
					  dagNodes( allocator ), nInputs( nInputs ), nOutputs( nOutputs ), nLevels( nLevels ), cutSizeLimit( cutSizeLimit ) { }

Windowing::~Windowing() {

}
					
bool Windowing::computeReconvDrivenWindow( aig::GNode & node ) {

	GNodeSet sources( this->allocator );
	sources.insert( node );

	//collectNodesTFO( sources ); // REC
	collectNodesTFO_iter( sources ); // ITER
	collectRoots();

	if ( roots.size() > nOutputs ) {
		return false;
	}

	bool removeAfter = false;
	if ( roots.count( node ) == 0 ) {
		roots.insert( node );
		removeAfter = true;
	}

	//reconvDrivenCut( roots ); // REC
	reconvDrivenCut_iter( roots ); // ITER
	
	if ( removeAfter ) {
		roots.erase( node );
	}

	checkLeaves();
	checkRoots();
	checkWindow();

	if ( (leaves.size() > nInputs) || (leaves.size() < 2) ) {
		return false;
	}

	//if ( VERBOSE ) {
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		std::cout << "Reconvergence Driven Windowing for node: " << nodeData.id << std::endl;
		printSet( graph, window, "W" );
		printSet( graph, dagNodes, "D" );
		printSet( graph, leaves, "L" );
		printSet( graph, roots, "R" );
	//}

	return true;
}

bool Windowing::computeWindowTEST( aig::GNode & node ) {

	GNodeSet sources( this->allocator );
	sources.insert( node );

	//collectNodesTFO( sources ); // REC
	collectNodesTFO_iter( sources ); // ITER
	collectRoots();

	if ( roots.size() > nOutputs ) {
		return false;
	}

	bool removeAfter = false;
	if ( roots.count( node ) == 0 ) {
		roots.insert( node );
		removeAfter = true;
	}

	//reconvDrivenCut( roots ); // REC
	reconvDrivenCut_iter( roots ); // ITER
	
	if ( removeAfter ) {
		roots.erase( node );
	}

	checkLeaves();

	this->nLevels = 6;
	collectNodesTFO_iter( leaves );
	collectRoots();

	checkLeaves();
	checkRoots();
	checkWindow();

	if ( (leaves.size() > nInputs) || (leaves.size() < 2) ) {
		return false;
	}

	//if ( VERBOSE ) {
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		std::cout << "Reconvergence Driven Windowing for node: " << nodeData.id << std::endl;
		printSet( graph, window, "W" );
		printSet( graph, dagNodes, "D" );
		printSet( graph, leaves, "L" );
		printSet( graph, roots, "R" );
	//}

	return true;
}


/*
RDWindowingB {
	reconvDrivenCut( graph, start, leaves, window, dagNodes, nInputs );
	reconvDrivenBackCut( graph, leaves, roots, window, dagNodes, nOutputs );
	checkLeaves( graph, window, leaves, dagNodes );
	checkRoots( graph, window, roots, dagNodes );
	checkWindow( graph, window, leaves );
}
*/

// ################################################################ TFO ########################################################################################
// REC
void Windowing::collectNodesTFO( GNodeSet & sources ) {

	int levelCounter = 0;
	for ( aig::GNode node : sources ) {
		window.insert( node );
		collectNodesTFOrec( node, levelCounter );
	}
}

void Windowing::collectNodesTFOrec( aig::GNode & node, int levelCounter ) {

	if ( levelCounter == nLevels ) {
		return;
	}

	levelCounter++;
	for ( auto edge : graph.out_edges( node ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		window.insert( currentNode );
	
		int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
		aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );

		if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
			dagNodes.insert( currentNode );
		}

		collectNodesTFOrec( currentNode, levelCounter );
	}
}

// ITER
void Windowing::collectNodesTFO_iter( GNodeSet & sources ) {

	GNodeDeque deque( allocator );
	int level, lastInLevel, sizeBefore, counter;
	bool updateLastInLevel;

	for ( aig::GNode node : sources ) {
		
		lastInLevel = 1, level = 0, counter = 0;
		updateLastInLevel = true;
		deque.push_back( node );
		window.insert( node );
	
		while( !deque.empty() ) {

			aig::GNode currentNode = deque.front();
			deque.pop_front();
			lastInLevel--;

			int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
			aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );

			if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
				dagNodes.insert( currentNode );
			}

			if ( lastInLevel == 0 ) {
				updateLastInLevel = true;
			}

			if ( level < nLevels ) {
				for ( auto edge : graph.out_edges( currentNode ) ) {
					aig::GNode nextNode = graph.getEdgeDst( edge );
					aig::NodeData & nextNodeData = graph.getData( nextNode, galois::MethodFlag::READ );
					if ( nextNodeData.type != aig::NodeType::PO ) {
						auto status = window.insert( nextNode );
						if ( status.second == true ) {
							deque.push_back( nextNode );
							counter++;
						}
					}
				}

				if ( updateLastInLevel ) {
					lastInLevel = counter;
					level++;
					counter = 0;
					updateLastInLevel = false;
				}
			}
		}
	}
}
// ########################################################################################################################################################

void Windowing::collectRoots() {

	for ( aig::GNode node : window ) {
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		if ( nodeData.type == aig::NodeType::PO ) {
			roots.insert( node );
		}
		else {
			for ( auto edge : graph.out_edges( node ) ) {
				aig::GNode currentNode = graph.getEdgeDst( edge );
				auto it = window.find( currentNode );
				if ( it == window.end() ) {
					roots.insert( node );
				}
			}
		}
	}
}

// ##################################################################### RDCUT ###################################################################################
// REC
void Windowing::reconvDrivenCut( GNodeSet & sources ) {

	GNodeSet currentWindow( allocator );
	GNodeSet currentLeaves( allocator );

	for ( aig::GNode node : sources ) {
		//currentWindow.clear();
		currentLeaves.clear();
		currentWindow.insert( node );
		currentLeaves.insert( node );
		constructCut( currentLeaves, currentWindow );
		unification( window, currentWindow );
		unification( leaves, currentLeaves );
	}
}

void Windowing::constructCut( GNodeSet & currentLeaves, GNodeSet & currentWindow ) {

	aig::GNode minCostNode;
	int minCost = std::numeric_limits<int>::max();
	bool onlyPIs = true;
	for ( aig::GNode node : currentLeaves ) {
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		if ( nodeData.type != aig::NodeType::PI ) {
			int cost = leafCost( node, currentWindow );
			if ( minCost > cost ) {
				minCost = cost;
				minCostNode = node;
				onlyPIs = false;
			}
		}
	}
	if ( onlyPIs || (currentLeaves.size() + minCost) > cutSizeLimit ) {
		return;
	}

	currentLeaves.erase( minCostNode );
	for ( auto edge : graph.in_edges( minCostNode ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		currentLeaves.insert( currentNode );
		currentWindow.insert( currentNode );

		int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
		aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );

		if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
			dagNodes.insert( currentNode );
		}
	}

	constructCut( currentLeaves, currentWindow );
}

// ITET
void Windowing::reconvDrivenCut_iter( GNodeSet & sources ) {

	GNodeSet currentWindow( allocator );
	GNodeSet currentLeaves( allocator );

	for ( aig::GNode node : sources ) {
		// currentWindow.clear();
		currentLeaves.clear();
		currentWindow.insert( node );
		currentLeaves.insert( node );

		while( true ) {
			aig::GNode minCostNode;
			int minCost = std::numeric_limits<int>::max();
			bool onlyPIs = true;
			for ( aig::GNode node : currentLeaves ) {
				aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
				if ( nodeData.type != aig::NodeType::PI ) {
					int cost = leafCost( node, currentWindow );
					if ( minCost > cost ) {
						minCost = cost;
						minCostNode = node;
						onlyPIs = false;
					}
				}
			}

			if ( onlyPIs || (currentLeaves.size() + minCost) > cutSizeLimit ) {
				break;
			}

			currentLeaves.erase( minCostNode );
			for ( auto edge : graph.in_edges( minCostNode ) ) {
				aig::GNode currentNode = graph.getEdgeDst( edge );
				currentLeaves.insert( currentNode );
				currentWindow.insert( currentNode );

				int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
				aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );

				if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
					dagNodes.insert( currentNode );
				}
			}
		}

		unification( window, currentWindow );
		unification( leaves, currentLeaves );
	}
}

int Windowing::leafCost( aig::GNode & node, GNodeSet & currentWindow ) {

	int cost = -1;
	for ( auto edge : graph.in_edges( node ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		auto it = currentWindow.find( currentNode );
		if ( it == currentWindow.end() ) {
			cost++;
		}
	}
	return cost;
}
// ########################################################################################################################################################


void Windowing::checkRoots() {

	GNodeSet notRoots( allocator );
	GNodeSet newRoots( allocator );

	for ( aig::GNode root : roots ) {
		if ( dagNodes.count( root ) != 0 ) {
			dagNodes.erase( root );	
		}

		aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::READ );

		if ( rootData.type == aig::NodeType::PO ) {
			notRoots.insert( root );
			auto inEdge = graph.in_edges( root ).begin();
			aig::GNode inNode = graph.getEdgeDst( *inEdge );
			newRoots.insert( inNode );
		}
	}

	for ( aig::GNode node : notRoots ) {
		roots.erase( node );
		window.erase( node );
	}

	for ( aig::GNode node : newRoots ) {
		if ( leaves.count( node ) == 0 ) {
			roots.insert( node );
		}
	}

	for ( aig::GNode node : dagNodes ) {
		for ( auto edge : graph.out_edges( node ) ) {
			aig::GNode currentNode = graph.getEdgeDst( edge );
			auto itW = window.find( currentNode );
			auto itR = roots.find( currentNode );
			if ( (itW == window.end()) && (itR == roots.end()) ) {
				roots.insert( node ); // In this case, the node is a side output of the window
				aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
				nodeData.counter = 1;
			}
		}
	}
}

void Windowing::checkLeaves() {

	for ( aig::GNode node : leaves ) {
		if ( window.count( node ) != 0 ) {
			window.erase( node );	
		}
		if ( dagNodes.count( node ) != 0 ) {
			dagNodes.erase( node );
		}
		if ( roots.count ( node ) != 0 ) {
			roots.erase( node );
		}
	}

	GNodeSet notLeaves( allocator );
	GNodeSet newLeaves( allocator );

	for ( aig::GNode node : leaves ) {
	
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );

		if ( nodeData.type != aig::NodeType::PI ) {

			int inWL = 0;
			aig::GNode newLeave; 

			for ( auto edge : graph.in_edges( node ) ) {
			
				aig::GNode inNode = graph.getEdgeDst( edge );
				auto itW = window.find( inNode );
				auto itL = leaves.find( inNode );

				if ( (itW != window.end()) || (itL != leaves.end()) ) {
					inWL++;
				}
				else {
					newLeave = inNode;
				}
			}
		
			if ( inWL == 2 ) {
				notLeaves.insert( node );
			}
			else {
				if ( inWL == 1 ) {
					notLeaves.insert( node );
					newLeaves.insert( newLeave );
				}
			}
		}
	}

	for ( aig::GNode node : notLeaves ) {
		leaves.erase( node );
		window.insert( node );
		int fanout = std::distance( graph.out_edges( node ).begin(), graph.out_edges( node ).end() );
		if ( fanout > 1 ) {
			dagNodes.insert( node );
		}
	}

	for ( aig::GNode node : newLeaves ) {
		leaves.insert( node );
	}
}

void Windowing::checkWindow() {

	for ( aig::GNode node : window ) {

		for ( auto edge : graph.in_edges( node ) ) {
		
			aig::GNode inNode = graph.getEdgeDst( edge );
			auto itW = window.find( inNode );
			auto itL = leaves.find( inNode );

			if ( (itW == window.end()) && (itL == leaves.end()) ) {
				leaves.insert( inNode );
			}
		}
	}
}

void Windowing::unification( GNodeSet & S1, GNodeSet & S2 ) {
	for ( aig::GNode node : S2 ) {
		S1.insert( node );
	}
}

GNodeSet & Windowing::getWindow() {
	return this->window;
}

GNodeSet & Windowing::getRoots() {
	return this->roots;
}

GNodeSet & Windowing::getLeaves() {
	return this->leaves;
}

GNodeSet & Windowing::getDagNodes() {
	return this->dagNodes;
}


// ################################################## UTIL ################################################## //

void Windowing::printSet( aig::Graph & graph, GNodeSet & S, std::string label ) {

	std::cout << label << " = { ";
	for ( aig::GNode s : S ) {
		aig::NodeData & sData = graph.getData( s, galois::MethodFlag::READ );
		std::cout << sData.id << " ";
	}
	std::cout << "}" << std::endl;
}

std::string Windowing::toDot() {

	std::stringstream dot, inputs, outputs, ands, edges;
	
	for ( aig::GNode node: leaves ) {
		// Write Nodes
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		inputs << "\"" << nodeData.id << "\"";
		inputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#ff8080\", fontsize=20]" << std::endl;
		// Write Edges
		for ( auto edge : graph.out_edges( node ) ) {
			aig::GNode dstNode = graph.getEdgeDst( edge );
			auto itW = window.find( dstNode );
			if ( itW != window.end() ) {
				aig::NodeData & dstData = graph.getData( dstNode, galois::MethodFlag::READ );
				bool polarity = graph.getEdgeData( edge, galois::MethodFlag::READ );
				edges << "\"" << nodeData.id << "\" -> \"" << dstData.id << "\"";
				if ( polarity ) {
					edges << " [penwidth = 3, color=blue]" << std::endl;
				}
				else {
					edges << " [penwidth = 3, color=red, style=dashed]" << std::endl;
				}
			}
		}
	}
	
	for ( aig::GNode node: window ) {
		// Write Nodes
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		auto itR = roots.find( node );
		if ( itR != roots.end() ) {
			outputs << "\"" << nodeData.id << "\"";
			if ( nodeData.counter == 1 ) { // Used to diferenciate the DagNodes taht become root
				outputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#139042\", fontsize=20]" << std::endl;
			}
			else {
				outputs << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#008080\", fontsize=20]" << std::endl;	
			}
		}
		else {
			ands << "\"" << nodeData.id << "\"";
			ands << " [shape=circle, height=1, width=1, penwidth=5 style=filled, fillcolor=\"#ffffff\", fontsize=20]" << std::endl;		
		}
		// Write Edges
		for ( auto edge : graph.out_edges( node ) ) {
			aig::GNode dstNode = graph.getEdgeDst( edge );
			auto itW = window.find( dstNode );
			if ( itW != window.end() ) {
				aig::NodeData & dstData = graph.getData( dstNode, galois::MethodFlag::READ );
				bool polarity = graph.getEdgeData( edge, galois::MethodFlag::READ );
				edges << "\"" << nodeData.id << "\" -> \"" << dstData.id << "\"";
				if ( polarity ) {
					edges << " [penwidth = 3, color=blue]" << std::endl;
				}
				else {
					edges << " [penwidth = 3, color=red, style=dashed]" << std::endl;
				}
			}
		}
	}

	dot << "digraph aig {" << std::endl;
	dot << inputs.str();
	dot << ands.str();
	dot << outputs.str();
	dot << edges.str();
	dot << "{ rank=source;";
	for ( aig::GNode node: leaves ) {
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		dot << " \"" << nodeData.id << "\"";
	}
	dot << " }" << std::endl;
	
	/* It is commented for better visualization of the graph */
	// dot << "{ rank=sink;";
	// for ( aig::GNode node: roots ) {
	// 	aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
	// 	dot << " \"" << nodeData.id << "\"";
	// }
	// dot << " }" << std::endl;
	
	dot << "rankdir=\"BT\"" << std::endl;
	dot << "}" << std::endl;

	return dot.str();
}

void Windowing::writeWindowDot( std::string dotText, std::string label ) {

	std::stringstream path;
	path << "/workspace/vnpossani/GALOIS_DUMPS/dots/window_n_" << label << ".dot";
	std::ofstream dotFile;
	dotFile.open( path.str() );
	dotFile << dotText;
	dotFile.close();
}


} /* namespace algorithm */



