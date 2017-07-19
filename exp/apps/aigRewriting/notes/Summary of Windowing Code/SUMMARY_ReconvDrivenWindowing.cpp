#include "ReconvDrivenWindowing.h"
#include "FunctionHandler.h"
#include "FunctionPool.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"

#include <iostream>
#include <set>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <utility>

#define MAXLEAVES 16

namespace algorithm {

// ################################################## AUXILIARY STRUCTS AND CLASSES ################################################## //

/* Custom hasher for storing GNodes in STL constainers such as unordered_set and unordered_map */
class Hasher {
	aig::Graph & graph;

	public:
    	Hasher( aig::Graph & graph ) : graph( graph ) { }

		size_t operator()( const aig::GNode node ) const {
			aig::NodeData & nodeData = graph.getData( node, Galois::MethodFlag::READ );
			return nodeData.id;
		}
};

/* Custom comparator for storing GNodes in STL constainers such as unordered_set and unordered_map */
class Comparator {
    aig::Graph & graph;

	public:
    	Comparator( aig::Graph & graph ) : graph( graph ) { }

		bool operator()( aig::GNode lhsNode, aig::GNode rhsNode ) const {
			aig::NodeData & lhsNodeData = graph.getData( lhsNode, Galois::MethodFlag::READ );
			aig::NodeData & rhsNodeData = graph.getData( rhsNode, Galois::MethodFlag::READ );
			return (lhsNodeData.id == rhsNodeData.id);
		}
};

typedef std::unordered_set< aig::GNode, Hasher, Comparator, Galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeSet;
typedef Functional::FunctionPool FunctionPool;
typedef Functional::word word;




// ################################################## WINDOWING ################################################## //

/* Functor structure for computing windows on AIGs */
struct RDWindowing {
	typedef int tt_does_not_need_aborts;
    typedef int tt_does_not_need_push;
	typedef int tt_needs_per_iter_alloc;
	typedef Galois::PerIterAllocTy Alloc;

	aig::Graph & graph;
	Hasher hasher;
	Comparator comparator;
	size_t nBuckets = 50; // FIXME
	int nInputs; 	// It is the cut-size limit of reconvergence-driven cuts
	int nOutputs;	// It is not being used yet. In this version, the number ob outputs of the window are uncontrained.
	int nLevels;	// It is the number of level that the algorithm looks ahead when computing the transitive fanout (TFO) of a griven node.

	RDWindowing( aig::Graph & graph, int nInputs, int nOutputs, int nLevels ) 
				: 	graph( graph ), 
					hasher( graph ), 
					comparator( graph ), 
					nInputs( nInputs ), 
					nOutputs( nOutputs ), 
					nLevels( nLevels ) { }

	void operator()( aig::GNode refNode, Galois::UserContext< aig::GNode > & ctx ) const {

		if ( !graph.containsNode( refNode, Galois::MethodFlag::READ ) ) {
			return;
		}

		aig::NodeData & nodeData = graph.getData( refNode, Galois::MethodFlag::READ );
		int fanout = std::distance( graph.out_edges( refNode ).begin(), graph.out_edges( refNode ).end() ); // Compute the fanout of the refNode
		
		if ( (nodeData.type == aig::NodeType::AND) && (fanout > 1) ) {
			
			Galois::PerIterAllocTy & allocator = ctx.getPerIterAlloc();

			/* 
			Create all necessary sets of nodes to compute the window
			*/
			GNodeSet sources( nBuckets, hasher, comparator, allocator );
			GNodeSet dagNodes( nBuckets, hasher, comparator, allocator );
			GNodeSet window( nBuckets, hasher, comparator, allocator );
			GNodeSet leaves( nBuckets, hasher, comparator, allocator );
			GNodeSet roots( nBuckets, hasher, comparator, allocator );

			/* 
			The collecNodesTFO routine receives a set of source nodes to start 
			the computation. In this case, we have just one source node.
			*/
			sources.insert( refNode );
			collectNodesTFO( graph, sources, window, dagNodes, nLevels );

			/*
			After compute the transitive fanout for the refNode, the set of roots must be defined.
			*/
			collectRoots( graph, window, roots );

			/* 
			Compute one reconvergence-driven cut for each root refNode. 
			In order to enforce the algorithm to compute a reconvergence-driven cut for the refNode, 
			we temporarily insert the refNode in the root set.
			*/
			roots.insert( refNode );
			reconvDrivenCut( graph, roots, leaves, window, dagNodes, nInputs, allocator );
			roots.erase( refNode );

			/*
			After computing the all reconvergence-driven cuts, it may be necessary to do 
			updates in the leaves, roots and window sets to keep the window consistent.
			*/
			checkLeaves( graph, window, leaves, dagNodes, allocator );
			checkRoots( graph, window, roots, dagNodes );
			checkWindow( graph, window, leaves );

			/* 
			We will synthesise Boolean function with at most 16-inputs. 
			Then, our upper bound for MAXLEAVES is 16.
			*/
			if ( leaves.size() > MAXLEAVES ) {
				return;
			}

			/*
			REWRITING CALL (UNDER DEVELOPMENT)
			rewrite( graph, window, leaves, roots );
			*/
		}
	}

	/* Computes the transitive fanout for each source node */
	void collectNodesTFO( aig::Graph & graph, GNodeSet & sources, GNodeSet & window, GNodeSet & dagNodes, int maxLevel ) const {
	
		int levelCounter = 0;
		for ( aig::GNode node : sources ) {
			window.insert( node );
			collectNodesTFOrec( graph, node, window, dagNodes, maxLevel, levelCounter );
		}
	}

	/* 
	Computes the transitive fanout for a given source node. 
	In other words, this routine looks for at most maxLevels ahead (outgoing edges) storing all visited nodes in the window set 
	*/
	void collectNodesTFOrec( aig::Graph & graph, aig::GNode & node, GNodeSet & window, GNodeSet & dagNodes, int maxLevel, int levelCounter ) const {
	
		if ( levelCounter == maxLevel ) {
			return;
		}
	
		levelCounter++;
		for ( auto edge : graph.out_edges( node ) ) {
			aig::GNode currentNode = graph.getEdgeDst( edge );
			window.insert( currentNode );
		
			int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
			aig::NodeData & currentNodeData = graph.getData( currentNode, Galois::MethodFlag::READ );

			if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
				dagNodes.insert( currentNode );
			}

			collectNodesTFOrec( graph, currentNode, window, dagNodes, maxLevel, levelCounter );
		}
	}

	/*
	Creates a subset (roots) of all nodes that are in the window and are feeding other nodes outside the window.
	*/
	void collectRoots( aig::Graph & graph, GNodeSet & window, GNodeSet & roots ) const {

		for ( aig::GNode node : window ) {
			aig::NodeData & nodeData = graph.getData( node, Galois::MethodFlag::READ );
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

	/*
	Extends the routine presented in the paper of Alan Mishchenko (IWLS 2006) to compute a 
	reconvergence-driven cut for each source node and merge them in a single set of leaves and window.
	*/
	void reconvDrivenCut( aig::Graph & graph, GNodeSet & sources, GNodeSet & leaves, GNodeSet & window, GNodeSet & dagNodes, int cutSizeLimit, Galois::PerIterAllocTy & allocator ) const {
	
		GNodeSet currentWindow( nBuckets, hasher, comparator, allocator );
		GNodeSet currentLeaves( nBuckets, hasher, comparator, allocator );

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

	/*
	Compute the reconvergence-driven cut itself by expanding the nodes with smallest leafCost. 
	This routine is strongly base on the one presented in the paper of Alan Mishchenko (IWLS 2006)
	*/
	void constructCut( aig::Graph & graph, GNodeSet & leaves, GNodeSet & currentWindow, GNodeSet & dagNodes, int cutSizeLimit ) const {
	
		aig::GNode minCostNode;
		int minCost = std::numeric_limits<int>::max();
		bool onlyPIs = true;
		for ( aig::GNode node : leaves ) {
			aig::NodeData & nodeData = graph.getData( node, Galois::MethodFlag::READ );
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
			aig::NodeData & currentNodeData = graph.getData( currentNode, Galois::MethodFlag::READ );

			if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
				dagNodes.insert( currentNode );
			}
		}

		constructCut( graph, leaves, currentWindow, dagNodes, cutSizeLimit );
	}

	/*
	Compute the leafCost according to the paper of Alan Mishchenko (IWLS 2006).
	Basically, the cost is defined by the impact that a node expansion has in the size of the set of leaves.
	*/
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

	/*
	Checks if there are nodes inside the window that are feeding other nodes outside the window.
	When this situation is found, the nodes inside the window are also inserte in the root set.
	*/
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

	/*
	Checks the consistence of the leave set. As the algorithm computes many reconvergence-driven cuts from different sources, 
	it may lead to updates in the leave set by removing some nodes and/or inserting new nodes in this set.
	*/
	void checkLeaves( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & dagNodes, Galois::PerIterAllocTy & allocator ) const {

		for ( aig::GNode node : leaves ) {
			if ( window.count( node ) != 0 ) {
				window.erase( node );	
			}
			if ( dagNodes.count( node ) != 0 ) {
				dagNodes.erase( node );
			}
		}

		GNodeSet notLeaves( nBuckets, hasher, comparator, allocator );
		GNodeSet newLeaves( nBuckets, hasher, comparator, allocator );

		for ( aig::GNode node : leaves ) {
		
			aig::NodeData & nodeData = graph.getData( node, Galois::MethodFlag::READ );

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

	/*
	Checks if there are nodes inside the window that are fed by nodes outside the window and also outside the leave set.
	When this situation is found, these nodes outside the window are inserte in the leave set to keep the window consistent.
	*/
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

	/* Performs the union of two sets of nodes */
	void unification( GNodeSet & S1, GNodeSet & S2 ) const {
		for ( aig::GNode node : S2 ) {
			S1.insert( node );
		}
	}

};




// ################################################## RUNNER ################################################## //

/* 
PREPROCESS 
Inserts in the worklist of active nodes all thoose nodes that have more than one outgoing edge.
*/
struct FromDagNodes {
	
	aig::Graph & graph;
	Galois::InsertBag< aig::GNode > & workList;
  
	FromDagNodes( aig::Graph & graph, Galois::InsertBag< aig::GNode > & workList ) : graph( graph ), workList( workList ) { }
  
	void operator()( aig::GNode node ) const {
			
		aig::NodeData & nodeData = graph.getData( node, Galois::MethodFlag::READ );
		int fanout = std::distance( graph.out_edges( node ).begin(), graph.out_edges( node ).end() );

		if ( fanout > 1 ) {
			workList.push( node );
		}
	}
};

/* 
This function starts the execution for windowing computation 
There are tow alternatives:
(1) Run a preprocessing that will collect all nodes with fanout greater than one. 
In other words, all nodes that have more than one outgoing edge will be considered active nodes.
(2) Directly visit all nodes of the graph by using iterator from graph.begin() to greaph.end().
*/
void run( int nInputs, int nOutputs, int nLevels ) {

	// HOW TO USE IT?
	//Galois::reportPageAlloc("MeminfoPRE");


	Galois::InsertBag< aig::GNode > workList;

	// PREPROCESS //
	Galois::do_all( this->aig.getGraph().begin(), this->aig.getGraph().end(), FromDagNodes( this->aig.getGraph(), workList ) ); // Starting from DAG nodes

	// PROCESS //

	// Alternative (1)
	Galois::for_each( workList.begin(), workList.end(), RDWindowing( this->aig.getGraph(), nInputs, nOutputs, nLevels ) ); // Active nodes from Worklist

	// Alternative (2)
	//Galois::for_each( this->aig.getGraph().begin(), this->aig.getGraph().end(), RDWindowing( this->aig.getGraph(), nInputs, nOutputs, nLevels ) ); // Graph Topology-Driven


	// HOW TO USE IT?
	//Galois::reportPageAlloc("MeminfoPOS");
}

} /* namespace algorithm */



