#include "Windowing.h"
#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include <iostream>
#include <set>
#include <limits>

namespace algorithm {

Windowing::Windowing( aig::Aig & aig ) : aig( aig ) { }

Windowing::~Windowing() { }

void collectNodesTFI( aig::Graph & graph, std::set< aig::GNode > & nodes, std::set< aig::GNode > & result, int maxLevel );
void collectNodesTFIrec( aig::Graph & graph, aig::GNode & node, std::set< aig::GNode > & result, int maxLevel, int levelCounter );
void collectNodesTFO( aig::Graph & graph, std::set< aig::GNode > & nodes, std::set< aig::GNode > & result, int maxLevel );
void collectNodesTFOrec( aig::Graph & graph, aig::GNode & node, std::set< aig::GNode > & result, int maxLevel, int levelCounter );
void collectLeaves( aig::Graph & graph, std::set< aig::GNode > & S, std::set< aig::GNode > & L );
void collectRoots( aig::Graph & graph, std::set< aig::GNode > & S, std::set< aig::GNode > & R );
void checkRoots( aig::Graph & graph, std::set< aig::GNode > & windowNodes, std::set< aig::GNode > & roots, std::set< aig::GNode > & dagNodes );
void reconvDrivenCut( aig::Graph & graph, std::set< aig::GNode > & nodes, std::set< aig::GNode > & leaves, std::set< aig::GNode > & windowNodes, std::set< aig::GNode > & dagNodes, int cutSizeLimit );
void constructCut( aig::Graph & graph, std::set< aig::GNode > & leaves, std::set< aig::GNode > & visited, std::set< aig::GNode > & dagNodes, int cutSizeLimit );
int leafCost( aig::Graph & graph, aig::GNode & node, std::set< aig::GNode > & visited );
void intersection( aig::Graph & graph, std::set< aig::GNode > & S1, std::set< aig::GNode > & S2, std::set< aig::GNode > & S );
void unification( std::set< aig::GNode > & S1, std::set< aig::GNode > & S2 );
void printSet( aig::Graph & graph, std::set< aig::GNode > & S, std::string label );

// ******************************************************************************************************************************************** //

void collectNodesTFI( aig::Graph & graph, std::set< aig::GNode > & nodes, std::set< aig::GNode > & result, int maxLevel ) {
	
	int levelCounter = 0;
	for ( aig::GNode node : nodes ) {
		result.insert( node );
		collectNodesTFIrec( graph, node, result, maxLevel, levelCounter );
	}
}

void collectNodesTFIrec( aig::Graph & graph, aig::GNode & node, std::set< aig::GNode > & result, int maxLevel, int levelCounter ) {
	
	if ( levelCounter == maxLevel ) {
		return;
	}

	levelCounter++;
	for ( auto edge : graph.in_edges( node ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		result.insert( currentNode );
		collectNodesTFIrec( graph, currentNode, result, maxLevel, levelCounter );
	}
}

void collectNodesTFO( aig::Graph & graph, std::set< aig::GNode > & nodes, std::set< aig::GNode > & result, int maxLevel ) {
	
	int levelCounter = 0;
	for ( aig::GNode node : nodes ) {
		result.insert( node );
		collectNodesTFOrec( graph, node, result, maxLevel, levelCounter );
	}
}

void collectNodesTFOrec( aig::Graph & graph, aig::GNode & node, std::set< aig::GNode > & result, int maxLevel, int levelCounter ) {
	
	if ( levelCounter == maxLevel ) {
		return;
	}
	
	levelCounter++;
	for ( auto edge : graph.out_edges( node ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		result.insert( currentNode );
		collectNodesTFOrec( graph, currentNode, result, maxLevel, levelCounter );
	}
}

void reconvDrivenCut( aig::Graph & graph, std::set< aig::GNode > & nodes, std::set< aig::GNode > & leaves, std::set< aig::GNode > & windowNodes, std::set< aig::GNode > & dagNodes, int cutSizeLimit ) {
	
	std::set< aig::GNode > currentLeaves;

	for ( aig::GNode node : nodes ) {
		// windowNodes.clear();
		currentLeaves.clear();
		windowNodes.insert( node );
		currentLeaves.insert( node );
		constructCut( graph, currentLeaves, windowNodes, dagNodes, cutSizeLimit );
		unification( leaves, currentLeaves );
	}
}

void constructCut( aig::Graph & graph, std::set< aig::GNode > & leaves, std::set< aig::GNode > & visited, std::set< aig::GNode > & dagNodes, int cutSizeLimit ) {
	
	aig::GNode minCostNode;
	int minCost = std::numeric_limits<int>::max();
	bool onlyPIs = true;
	for ( aig::GNode node : leaves ) {
		aig::NodeData & nodeData = graph.getData( node );
		if ( nodeData.type != aig::NodeType::PI ) {
			int currentCost = leafCost( graph, node, visited );
			if ( minCost > currentCost ) {
				minCost = currentCost;
				minCostNode = node;
				onlyPIs = false;
			}
		}
	}
	if ( onlyPIs || (leaves.size() + minCost) > cutSizeLimit ) {
		return;
	}

	leaves.erase( minCostNode );
	for ( auto edge : graph.in_edges( minCostNode ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		leaves.insert( currentNode );
		visited.insert( currentNode );

		int fanout = std::distance( graph.out_edges( currentNode ).begin(),graph.out_edges( currentNode ).end() );
		aig::NodeData & currentNodeData = graph.getData( currentNode );

		if ( fanout > 1 && (currentNodeData.type != aig::NodeType::PI) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
			dagNodes.insert( currentNode );
		}
	}

	constructCut( graph, leaves, visited, dagNodes, cutSizeLimit );
}

void collectLeaves( aig::Graph & graph, std::set< aig::GNode > & S, std::set< aig::GNode > & L ) {

	for ( aig::GNode node : S ) {
		for ( auto edge : graph.in_edges( node ) ) {
			aig::GNode currentNode = graph.getEdgeDst( edge );
			auto it = S.find( currentNode );
			if ( it == S.end() ) {
				L.insert( currentNode );
			}
		}
	}
}

void collectRoots( aig::Graph & graph, std::set< aig::GNode > & S, std::set< aig::GNode > & R ) {

	for ( aig::GNode node : S ) {
		aig::NodeData & nodeData = graph.getData( node );
		if ( nodeData.type == aig::NodeType::PO ) {
			R.insert( node );
		}
		else {
			for ( auto edge : graph.out_edges( node ) ) {
				aig::GNode currentNode = graph.getEdgeDst( edge );
				auto it = S.find( currentNode );
				if ( it == S.end() ) {
					R.insert( node );
				}
			}
		}
	}
}

void checkRoots( aig::Graph & graph, std::set< aig::GNode > & windowNodes, std::set< aig::GNode > & roots, std::set< aig::GNode > & dagNodes ) {

	for ( aig::GNode node : dagNodes ) {
		for ( auto edge : graph.out_edges( node ) ) {	
			aig::GNode currentNode = graph.getEdgeDst( edge );
			auto it = windowNodes.find( currentNode );
			if ( it == windowNodes.end() ) {
				roots.insert( node ); // In this case, node is a side output of the window
			}
		}
	}
}

int leafCost( aig::Graph & graph, aig::GNode & node, std::set< aig::GNode > & visited ) {

	int cost = -1;
	for ( auto edge : graph.in_edges( node ) ) {
		aig::GNode currentNode = graph.getEdgeDst( edge );
		auto it = visited.find( currentNode );
		if ( it == visited.end() ) {
			cost++;
		}
	}
	return cost;
}

void intersection( aig::Graph & graph, std::set< aig::GNode > & S1, std::set< aig::GNode > & S2, std::set< aig::GNode > & S ) {

	for ( aig::GNode iNode : S1 ) {
		aig::NodeData & iNodeData = graph.getData( iNode );
		for ( aig::GNode oNode : S2 ) {		
			aig::NodeData & oNodeData = graph.getData( oNode );
			if ( iNodeData.id == oNodeData.id ) {
				S.insert( iNode );
			}
		}
	}
}

void unification( std::set< aig::GNode > & S1, std::set< aig::GNode > & S2 ) {
	
	for ( aig::GNode node : S2 ) {
		S1.insert( node );
	}
}

void printSet( aig::Graph & graph, std::set< aig::GNode > & S, std::string label ) {

	std::cout << label << " = { ";
	for ( aig::GNode s : S ) {
		aig::NodeData & sData = graph.getData( s );
		std::cout << sData.id << " ";
	}
	std::cout << "}" << std::endl;
}

struct StandardWindowing {

	aig::Graph & graph;
	int nFanins;
	int nFanouts;

	StandardWindowing( aig::Graph & graph, int nFanins, int nFanouts ) : graph( graph ), nFanins( nFanins ), nFanouts( nFanouts ) { }

	void operator()( aig::GNode node, galois::UserContext< aig::GNode > & ctx ) const {
		
		aig::NodeData & nodeData = graph.getData( node );
		
		if ( nodeData.type == aig::NodeType::AND ) {
			std::set< aig::GNode > I1, I2, O1, O2, S, L, R, initialSet;
			initialSet.insert( node );

			collectNodesTFI( graph, initialSet, I1, nFanins );
			collectNodesTFO( graph, initialSet, O1, nFanouts );
			collectNodesTFI( graph, O1, I2, (nFanins+nFanouts) );
			collectNodesTFO( graph, I1, O2, (nFanins+nFanouts) );
			intersection( graph, I2, O2, S );
			collectLeaves( graph, S, L );
			collectRoots( graph, S, R );

			std::cout << "Standard Windowing for node: " << nodeData.id << std::endl;
			printSet( graph, I1, "I1" );
			printSet( graph, O1, "O1" );
			printSet( graph, I2, "I2" );
			printSet( graph, O2, "O2" );
			printSet( graph, S, "S" );
			printSet( graph, L, "L" );
			printSet( graph, R, "R" );
		}
	}

};

void Windowing::runStandardWindowing( int nFanins, int nFanouts ) {

	galois::InsertBag< aig::GNode > workList;

	for ( aig::GNode node : this->aig.getGraph() ) {
		aig::NodeData & nodeData = this->aig.getGraph().getData( node );
		if ( nodeData.id == 8 ) {
			workList.push( node );
		}
	}

	galois::for_each( workList.begin(), workList.end(), StandardWindowing( this->aig.getGraph(), nFanins, nFanouts ) );
}

struct ReconvDrivenWindowing {

	aig::Graph & graph;
	int nInputs;
	int nOutputs;
	int nLevels;

	ReconvDrivenWindowing( aig::Graph & graph, int nInputs, int nOutputs, int nLevels ) : graph( graph ), nInputs( nInputs ), nOutputs( nOutputs ), nLevels( nLevels ) { }

	void operator()( aig::GNode node, galois::UserContext< aig::GNode > & ctx ) const {
		
		aig::NodeData & nodeData = graph.getData( node );
		
		if ( nodeData.type == aig::NodeType::AND ) {
			
			std::set< aig::GNode > Start, TFO, DagNodes, Window, Leaves, Roots;
			Start.insert( node );

			collectNodesTFO( graph, Start, TFO, nLevels );
			collectRoots( graph, TFO, Roots );
			reconvDrivenCut( graph, Roots, Leaves, Window, DagNodes, nInputs );
			checkRoots( graph, Window, Roots, DagNodes );

			std::cout << "Reconvergence Driven Windowing for node: " << nodeData.id << std::endl;
			printSet( graph, TFO, "TFO" );
			printSet( graph, Window, "W" );
			printSet( graph, DagNodes, "D" );
			printSet( graph, Leaves, "L" );
			printSet( graph, Roots, "R" );
		}
	}
};

void Windowing::runReconvDrivenWindowing( int nInputs, int nOutputs, int nLevels ) {

	galois::InsertBag< aig::GNode > workList;

	for ( aig::GNode node : this->aig.getGraph() ) {
		aig::NodeData & nodeData = this->aig.getGraph().getData( node );
		if ( nodeData.id == 12 ) {
			workList.push( node );
		}
	}
	
	galois::for_each( workList.begin(), workList.end(), ReconvDrivenWindowing( this->aig.getGraph(), nInputs, nOutputs, nLevels ) );
}

} /* namespace algorithm */

