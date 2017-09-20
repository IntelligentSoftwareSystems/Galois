#include "ReconvDrivenWindowing.h"
#include "FunctionHandler.h"
#include "FunctionPool.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Accumulator.h"

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
#define DUMPDOT false

#define WIN_BUCKET_SIZE 5
#define WIN_BUCKET_NUM 20

#define FANOUT_BUCKET_SIZE 1
#define FANOUT_BUCKET_NUM 50

namespace algorithm {

// ################################################## AUXILIARY STRUCTS AND CLASSES ################################################## //
galois::GAccumulator<size_t> windowHistogram[ WIN_BUCKET_NUM ];
galois::GAccumulator<size_t> fanoutHistogram[ FANOUT_BUCKET_NUM ];


typedef std::unordered_set< aig::GNode, std::hash<aig::GNode>, std::equal_to<aig::GNode>, galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeSet;
typedef std::deque< aig::GNode, galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeDeque;
typedef Functional::FunctionPool FunctionPool;
typedef Functional::word word;


// ################################################## FUNCTION HEADEARS ################################################## //

void addInfoWindowHistogram( GNodeSet & window );
void addInfoFanoutHistogram( int fanout );

void optimize( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots );
void findWindowExpSup( aig::Graph & graph, GNodeSet & leaves, GNodeSet & roots, std::unordered_map< int, std::pair< std::string, std::unordered_set< int > > > & expSupMap );
void findExpSup( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::unordered_map< int, std::pair< std::string, std::unordered_set< int > > > & expSupMap );

void buildWindowFunctions( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots );
word * buildFunction( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::unordered_map< int, Functional::word* > & computedFunctions, FunctionPool & functionPool, int nWords );



void printSet( aig::Graph & graph, GNodeSet & S, std::string label );
std::string buildDot( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots );
void circuitToDot( aig::Aig & aig, std::string label );
void dumpLogicFuntions( std::string hexaText, std::string label );
void dumpWindowDot( std::string dotText, std::string label );


/*
struct FromPI {
	
	aig::Graph & graph;
	galois::InsertBag< aig::GNode > & workList;
  
	FromPI( aig::Graph & graph, galois::InsertBag< aig::GNode > & workList ) : graph( graph ), workList( workList ) { }
  
	void operator()( aig::GNode pi ) const {
			
		for ( auto outEdge : graph.out_edges( pi ) ) {
			aig::GNode outNode = graph.getEdgeDst( outEdge );
			aig::NodeData & outNodeData = graph.getData( outNode, galois::MethodFlag::WRITE );
			if ( outNodeData.scheduled == false ) {
				workList.push( outNode );
				outNodeData.scheduled = true;
			}
		}
	}
};
*/

struct FromDagNodes {
	
	aig::Graph & graph;
	galois::InsertBag< aig::GNode > & workList;
	int maxFanout;
  
	FromDagNodes( aig::Graph & graph, galois::InsertBag< aig::GNode > & workList, int nFanout ) : graph( graph ), workList( workList ) { 
		maxFanout = nFanout;
	}
  
	void operator()( aig::GNode node ) const {
			
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		int nodeFanout = std::distance( graph.out_edges( node ).begin(), graph.out_edges( node ).end() );

		//if ( (nodeData.type == aig::NodeType::AND) && (nodeFanout > 1) && (nodeFanout <= maxFanout) ) {
		if ( (nodeFanout > 1) && (nodeFanout <= maxFanout) ) {
			workList.push( node );
		}
		
		//addInfoFanoutHistogram( nodeFanout );
	}
};



// ################################################## WINDOWING ################################################## //

struct RDWindowing {
	typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;
	typedef int tt_needs_per_iter_alloc;
	typedef galois::PerIterAllocTy Alloc;

	aig::Graph & graph;
	int nInputs;
	int nOutputs;
	int nLevels;
	int cutSizeLimit;

	RDWindowing( aig::Graph & graph, int nInputs, int nOutputs, int nLevels, int cutSizeLimit ) 
				: 	graph( graph ), 
					nInputs( nInputs ), 
					nOutputs( nOutputs ), 
					nLevels( nLevels ), 
					cutSizeLimit( cutSizeLimit ) { }
					
	void operator()( aig::GNode node, galois::UserContext< aig::GNode > & ctx ) const {

		if ( !graph.containsNode( node, galois::MethodFlag::READ ) ) {
			return;
		}
				
		galois::PerIterAllocTy & allocator = ctx.getPerIterAlloc();

		GNodeSet start( allocator );
		GNodeSet dagNodes( allocator );
		GNodeSet window( allocator );
		GNodeSet leaves( allocator );
		GNodeSet roots( allocator );

		start.insert( node );

		//collectNodesTFO( graph, start, window, dagNodes, nLevels ); // REC
		collectNodesTFO_iter( graph, start, window, dagNodes, nLevels, allocator ); // ITER
		collectRoots( graph, window, roots );

		if ( roots.size() > nOutputs ) {
			return;
		}

		roots.insert( node );
		//reconvDrivenCut( graph, roots, leaves, window, dagNodes, nInputs, allocator ); // REC
		reconvDrivenCut_iter( graph, roots, leaves, window, dagNodes, cutSizeLimit, allocator ); // ITER
		roots.erase( node );

		checkLeaves( graph, window, leaves, dagNodes, allocator );
		checkRoots( graph, window, roots, dagNodes );
		checkWindow( graph, window, leaves );

		if ( leaves.size() > nInputs ) {
			return;
		}

		addInfoWindowHistogram( window );

		if ( VERBOSE ) {
			aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
			std::cout << "Reconvergence Driven Windowing for node: " << nodeData.id << std::endl;
			printSet( graph, window, "W" );
			printSet( graph, dagNodes, "D" );
			printSet( graph, leaves, "L" );
			printSet( graph, roots, "R" );
		}

		if ( DUMPDOT ) {
			aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
			std::string dotText = buildDot( graph, window, leaves, roots );
			dumpWindowDot( dotText, std::to_string( nodeData.id ) );
		}

		optimize( graph, window, leaves, roots );
	}

	// ################################################################ TFO ########################################################################################
	// REC
	void collectNodesTFO( aig::Graph & graph, GNodeSet & sources, GNodeSet & window, GNodeSet & dagNodes, int maxLevel ) const {
	
		int levelCounter = 0;
		for ( aig::GNode node : sources ) {
			window.insert( node );
			collectNodesTFOrec( graph, node, window, dagNodes, maxLevel, levelCounter );
		}
	}

	void collectNodesTFOrec( aig::Graph & graph, aig::GNode & node, GNodeSet & window, GNodeSet & dagNodes, int maxLevel, int levelCounter ) const {
	
		if ( levelCounter == maxLevel ) {
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

			collectNodesTFOrec( graph, currentNode, window, dagNodes, maxLevel, levelCounter );
		}
	}

	// ITER
	void collectNodesTFO_iter( aig::Graph & graph, GNodeSet & sources, GNodeSet & window, GNodeSet & dagNodes, int maxLevel, galois::PerIterAllocTy & allocator ) const {

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

				if ( lastInLevel == 0 ) {
					updateLastInLevel = true;
				}

				if ( level < maxLevel ) {
					for ( auto edge : graph.out_edges( currentNode ) ) {
						aig::GNode nextNode = graph.getEdgeDst( edge );
						auto status = window.insert( nextNode );
						if ( status.second == true ) {
							deque.push_back( nextNode );
							counter++;
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

	void collectRoots( aig::Graph & graph, GNodeSet & window, GNodeSet & roots ) const {

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
	void reconvDrivenCut( aig::Graph & graph, GNodeSet & sources, GNodeSet & leaves, GNodeSet & window, GNodeSet & dagNodes, int cutSizeLimit, galois::PerIterAllocTy & allocator ) const {

		GNodeSet currentWindow( allocator );
		GNodeSet currentLeaves( allocator );

		for ( aig::GNode node : sources ) {
			// currentWindow.clear();
			currentLeaves.clear();
			currentWindow.insert( node );
			currentLeaves.insert( node );
			constructCut( graph, currentLeaves, currentWindow, dagNodes, cutSizeLimit );
			unification( window, currentWindow );
			unification( leaves, currentLeaves );
		}
	}

	void constructCut( aig::Graph & graph, GNodeSet & leaves, GNodeSet & currentWindow, GNodeSet & dagNodes, int cutSizeLimit ) const {
	
		aig::GNode minCostNode;
		int minCost = std::numeric_limits<int>::max();
		bool onlyPIs = true;
		for ( aig::GNode node : leaves ) {
			aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
			if ( nodeData.type != aig::NodeType::PI ) {
				int cost = leafCost( graph, node, currentWindow );
				if ( minCost > cost ) {
					minCost = cost;
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
			currentWindow.insert( currentNode );

			int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
			aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );

			if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
				dagNodes.insert( currentNode );
			}
		}

		constructCut( graph, leaves, currentWindow, dagNodes, cutSizeLimit );
	}

	// ITET
	void reconvDrivenCut_iter( aig::Graph & graph, GNodeSet & sources, GNodeSet & leaves, GNodeSet & window, GNodeSet & dagNodes, int cutSizeLimit, galois::PerIterAllocTy & allocator ) const {

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
						int cost = leafCost( graph, node, currentWindow );
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


	int leafCost( aig::Graph & graph, aig::GNode & node, GNodeSet & currentWindow ) const {

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
	

	void checkRoots( aig::Graph & graph, GNodeSet & window, GNodeSet & roots, GNodeSet & dagNodes ) const {

		for ( aig::GNode node : roots ) {
			if ( dagNodes.count( node ) != 0 ) {
				dagNodes.erase( node );	
			}
		}

		for ( aig::GNode node : dagNodes ) {
			for ( auto edge : graph.out_edges( node ) ) {
				aig::GNode currentNode = graph.getEdgeDst( edge );
				auto it = window.find( currentNode );
				if ( it == window.end() ) {
					roots.insert( node ); // In this case, the node is a side output of the window
					aig::NodeData & nodeData = graph.getData( node );
					nodeData.counter = 1;
				}
			}
		}
	}

	void checkLeaves( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & dagNodes, galois::PerIterAllocTy & allocator ) const {

		for ( aig::GNode node : leaves ) {
			if ( window.count( node ) != 0 ) {
				window.erase( node );	
			}
			if ( dagNodes.count( node ) != 0 ) {
				dagNodes.erase( node );
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

	void checkWindow( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves ) const {

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

	void unification( GNodeSet & S1, GNodeSet & S2 ) const {
		for ( aig::GNode node : S2 ) {
			S1.insert( node );
		}
	}

};

/*
RDWindowingB {
	reconvDrivenCut( graph, start, leaves, window, dagNodes, nInputs );
	reconvDrivenBackCut( graph, leaves, roots, window, dagNodes, nOutputs );
	checkLeaves( graph, window, leaves, dagNodes );
	checkRoots( graph, window, roots, dagNodes );
	checkWindow( graph, window, leaves );
}

RDWindowingC {
	reconvDrivenCut( graph, start, leaves, window, dagNodes, nInputs );
	collectNodesTFO( graph, leaves, window, dagNodes, nLevels );
	collectRoots( graph, window, roots );
	checkLeaves( graph, window, leaves, dagNodes );
	checkRoots( graph, window, roots, dagNodes );
	checkWindow( graph, window, leaves );
}
*/




// ################################################## GRAPH PROFILE ################################################## //
void addInfoWindowHistogram( GNodeSet & window ) {

	int index = window.size() / WIN_BUCKET_SIZE;

	if ( index < WIN_BUCKET_NUM ) {
		windowHistogram[ index ] += 1;
	}
	else {
		std::cout << "Window histogram index out of bound: " << index << std::endl;
	}
}

void addInfoFanoutHistogram( int fanout ) {

	int index = fanout / FANOUT_BUCKET_SIZE;

	if ( index < FANOUT_BUCKET_NUM ) {
		fanoutHistogram[ index ] += 1;
	}
	else {
		std::cout << "Fanout histogram index out of bound: " << index << std::endl;
	}
}




// ################################################## REWRITING ################################################## //

void optimize( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots ) {

	std::unordered_map< int, std::pair< std::string, std::unordered_set< int > > > expSupMap;

	findWindowExpSup( graph, leaves, roots, expSupMap );

	// Mixer

	// for ( auto root : roots ) {
		// expression = expSupMap[ r.id ].first
		// support = expSupMap[ r.id ].second
		// FC ( expression, support )
		// Mixer.mix( FC.solutions )
	// }

	// Covering Finder

	// Covering Chooser

	// if ( hasInprovement ) {
		// Replace
	//}
	
}

void findWindowExpSup( aig::Graph & graph, GNodeSet & leaves, GNodeSet & roots, std::unordered_map< int, std::pair< std::string, std::unordered_set< int > > > & expSupMap ) {

	for ( aig::GNode leave : leaves ) {	
		aig::NodeData & leaveData = graph.getData( leave, galois::MethodFlag::READ );
		std::unordered_set< int > support;
		support.insert( leaveData.id );
		std::string expression = ("i"+std::to_string( leaveData.id ));
		auto pair = std::make_pair( expression, support );
		expSupMap.insert( std::make_pair( leaveData.id, pair ) );
	}

	for ( aig::GNode root : roots ) {
		
		findExpSup( graph, leaves, root, expSupMap );

		//if ( VERBOSE ) {
			aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::READ );
			std::cout << "ID: " << rootData.id << " -> Value: " << expSupMap[ rootData.id ].first << std::endl;
		//}
	}
}

void findExpSup( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::unordered_map< int, std::pair< std::string, std::unordered_set< int > > > & expSupMap ) {
	
	std::string expression = "";
	std::unordered_set< int > support;

	for ( auto inEdge : graph.in_edges( currentNode ) ) {

		aig::GNode inNode = graph.getEdgeDst( inEdge );
		aig::NodeData & inNodeData = graph.getData( inNode, galois::MethodFlag::READ );
		bool inEdgeData = graph.getEdgeData( inEdge );

		auto it = expSupMap.find( inNodeData.id );
		if ( it == expSupMap.end() ) {
			findExpSup( graph, leaves, inNode, expSupMap );
		}

		auto element = expSupMap[ inNodeData.id ];

		if ( inEdgeData == true ) {
			if ( expression.empty() ) {
				expression = element.first;
			}
			else {
				expression = "(" + expression + "*" + element.first + ")";
			}
		}
		else {
			if ( expression.empty() ) {
				expression = "!" + element.first;
			}
			else {
				expression = "(" + expression + "*!" + element.first + ")";
			}	
		}

		support.insert( element.second.begin(), element.second.end() );
	}

	aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );
	auto pair = std::make_pair( expression, support );
	expSupMap.insert( std::make_pair( currentNodeData.id, pair ) );
}


void buildWindowFunctions( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots ) {

	std::vector< std::string > varSet;
	std::unordered_map< std::string, word* > varMasks;
	std::unordered_map< int, word* > computedFunctions;

	for ( aig::GNode leave : leaves ) {	
		aig::NodeData & leaveData = graph.getData( leave, galois::MethodFlag::READ );
		varSet.push_back( "i"+std::to_string( leaveData.id ) );
	}

	int nWords = Functional::wordNum( varSet.size() );
	int nFunctions = 1000; // FIXME
	FunctionPool functionPool( nWords, nFunctions );
	Functional::startTruthTable( varSet, varMasks, functionPool );

	for ( auto leave : leaves ) {
		aig::NodeData & leaveData = graph.getData( leave, galois::MethodFlag::READ );
		auto it = varMasks.find( ("i"+std::to_string( leaveData.id )) );
		computedFunctions.insert( std::make_pair( leaveData.id, it->second ) );
	}

	for ( aig::GNode root : roots ) {
		buildFunction( graph, leaves, root, computedFunctions, functionPool, nWords );
	}

	if ( VERBOSE ) {
		for ( auto root: roots ) {
			aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::READ );
			std::cout << "ID: " << rootData.id << " -> Value: " << Functional::toHex( computedFunctions[ rootData.id ], nWords ) << std::endl;
		}
	}
}

word * buildFunction( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::unordered_map< int, Functional::word* > & computedFunctions, FunctionPool & functionPool, int nWords ) {

	aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );
	auto it = computedFunctions.find( currentNodeData.id );
	
	if ( it != computedFunctions.end() ) {
		return it->second;
	}

	if ( currentNodeData.type == aig::NodeType::AND ) {
	
		auto edgeIt = graph.in_edges( currentNode ).begin();
		aig::GNode lhsNode = graph.getEdgeDst( *edgeIt );
		bool lhsEdge = graph.getEdgeData( *edgeIt );
		edgeIt++;
		aig::GNode rhsNode = graph.getEdgeDst( *edgeIt );
		bool rhsEdge = graph.getEdgeData( *edgeIt );

		
		word * lhsFunction = buildFunction( graph, leaves, lhsNode, computedFunctions, functionPool, nWords );
		word * rhsFunction = buildFunction( graph, leaves, rhsNode, computedFunctions, functionPool, nWords );

		if ( lhsEdge == false ) {
			word * aux = functionPool.getNextFunction();
			Functional::NOT( aux, lhsFunction, nWords );
			lhsFunction = aux;
		}

		if ( rhsEdge == false ) {
			word * aux = functionPool.getNextFunction();
			Functional::NOT( aux, rhsFunction, nWords );
			rhsFunction = aux;
		}

		word * result = functionPool.getNextFunction();
		Functional::AND( result, lhsFunction, rhsFunction, nWords );
		computedFunctions.insert( std::make_pair( currentNodeData.id, result ) );

		return computedFunctions[ currentNodeData.id ];
	}
	else {
		if ( currentNodeData.type == aig::NodeType::PO ) {

			auto edgeIt = graph.in_edges( currentNode ).begin();
			aig::GNode inNode = graph.getEdgeDst( *edgeIt );
			bool inEdge = graph.getEdgeData( *edgeIt );

			word * inFunction = buildFunction( graph, leaves, inNode, computedFunctions, functionPool, nWords );
	
			if ( inEdge == false ) {
				word * aux = functionPool.getNextFunction();
				Functional::NOT( aux, inFunction, nWords );
				computedFunctions.insert( std::make_pair( currentNodeData.id, aux ) );
			}
			else {
				computedFunctions.insert( std::make_pair( currentNodeData.id, inFunction ) );
			}

			return computedFunctions[ currentNodeData.id ];
		}
		else {
			std::cout << "ERROR WHILE BULDING WINDOW FUNCTIONS!" << std::endl;
			exit(1);
		}
	}
}




// ################################################## RUNNER ################################################## //

ReconvDrivenWindowing::ReconvDrivenWindowing( aig::Aig & aig ) : aig( aig ) { }

ReconvDrivenWindowing::~ReconvDrivenWindowing() { }

void ReconvDrivenWindowing::run( int nOutputs, int nInputs, int nLevels, int nFanout, int cutSizeLimit, int verbose ) {
	
	galois::InsertBag< aig::GNode > workList;

	// PREPROCESS //
	// galois::do_all( this->aig.getInputNodes().begin(), this->aig.getInputNodes().end(), FromPI( this->aig.getGraph(), workList ) ); // Starting from PIs
	galois::do_all( this->aig.getGraph().begin(), this->aig.getGraph().end(), FromDagNodes( this->aig.getGraph(), workList, nFanout ) ); // Starting from DAG nodes

	if ( verbose == 1 ) {
		std::cout << std::distance(workList.begin(), workList.end()) << ";";
	}
	else {
		if ( verbose == 2 ) {
			std::cout << "Number of Active Nodes: " << std::distance(workList.begin(), workList.end()) << std::endl;
		}
	}

	// PROCESS //
	galois::for_each( workList.begin(), workList.end(), RDWindowing( this->aig.getGraph(), nInputs, nOutputs, nLevels, cutSizeLimit ) ); // Active nodes from Worklist
	//galois::for_each( this->aig.getGraph().begin(), this->aig.getGraph().end(), RDWindowing( this->aig.getGraph(), nInputs, nOutputs, nLevels ) ); // Graph Topology-Driven

	if ( verbose == 3 ) {
		for ( int i = 0; i < WIN_BUCKET_NUM; i++ ) {
			std::cout << windowHistogram[i].reduce() << ";";
		}
		std::cout << std::endl;
		for ( int i = 0; i < WIN_BUCKET_NUM; i++ ) {
			std::cout << ((i+1)*WIN_BUCKET_SIZE) << ";";
		}
		std::cout << std::endl << std::endl;
	}
	else {
		if ( verbose == 2 ) {
			std::cout << "Histogram for the window size (buckets in units of " << WIN_BUCKET_SIZE << "s):\n";
			for ( int i = 0; i < WIN_BUCKET_NUM; i++ ) {
				std::cout << ((i+1)*WIN_BUCKET_SIZE) << ": " << windowHistogram[i].reduce() << std::endl;
			}
		}
	}

/*
	std::cout << "Histogram of node fanout (buckets in units of " << FANOUT_BUCKET_SIZE << "s):\n";
	for ( int i = 0; i < FANOUT_BUCKET_NUM; i++ ) {
		std::cout << ((i+1)*FANOUT_BUCKET_SIZE) << ": " << fanoutHistogram[i].reduce() << std::endl;
	}
*/

}




// ################################################## UTIL ################################################## //

void printSet( aig::Graph & graph, GNodeSet & S, std::string label ) {

	std::cout << label << " = { ";
	for ( aig::GNode s : S ) {
		aig::NodeData & sData = graph.getData( s, galois::MethodFlag::READ );
		std::cout << sData.id << " ";
	}
	std::cout << "}" << std::endl;
}

std::string buildDot( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots ) {

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

void circuitToDot( aig::Aig & aig, std::string label, galois::PerIterAllocTy & allocator ) {

	GNodeSet window( allocator );
	GNodeSet leaves( allocator );
	GNodeSet roots( allocator );

	for ( auto it = aig.getGraph().begin(); it != aig.getGraph().end(); it++ ) {
		aig::GNode currentNode = (*it);
		aig::NodeData & currentNodeData = aig.getGraph().getData( currentNode, galois::MethodFlag::READ );
		if ( (currentNodeData.type != aig::NodeType::PI) && (currentNodeData.type != aig::NodeType::CONST) ) {
			window.insert( currentNode );
		}
	}

	for ( auto node : aig.getInputNodes() ) {
		leaves.insert( node );
	}

	for ( auto node : aig.getOutputNodes() ) {
		roots.insert( node );
	}

	std::string dotText = buildDot( aig.getGraph(), window, leaves, roots );
	dumpWindowDot( dotText, label );
}

void dumpLogicFuntions( std::string hexaText, std::string label ) {

	std::stringstream path;
	path << "/workspace/vnpossani/GALOIS_DUMPS/functions/" << label << "-inputs.txt";
	std::ofstream functionFile;
	functionFile.open( path.str(), std::ios_base::app );
	functionFile << hexaText;
	functionFile.close();
}

void dumpWindowDot( std::string dotText, std::string label ) {

	std::stringstream path;
	path << "/workspace/vnpossani/GALOIS_DUMPS/dots/window_n_" << label << ".dot";
	std::ofstream dotFile;
	dotFile.open( path.str() );
	dotFile << dotText;
	dotFile.close();
}


} /* namespace algorithm */



