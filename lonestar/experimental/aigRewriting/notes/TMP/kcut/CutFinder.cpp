#include "CutFinder.h"
#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Timer.h"

#include <iostream>

namespace algorithm {

CutFinder::CutFinder( aig::Aig & aig ) : aig( aig ) { }
CutFinder::~CutFinder() { }

void registerAndPruneCuts( std::set< aig::GNode > & cut, std::vector< std::set< aig::GNode > > & cutVector );
void combineCuts( aig::Graph & graph, aig::GNode node, int k, int c );

struct Preprocess {
	
	aig::Graph & graph;
	galois::InsertBag< aig::GNode > & workList;
  
	Preprocess( aig::Graph & graph, galois::InsertBag< aig::GNode > & workList ) : graph( graph ), workList( workList ) { }
  
	void operator()( aig::GNode pi ) const {
			
		aig::NodeData & piData = graph.getData( pi );
		std::set< aig::GNode > cut;
		cut.insert( pi );
		piData.cuts.push_back( cut );
		piData.bestCut = cut;

		for ( auto edge : graph.out_edges( pi ) ) {
		
			aig::GNode node = graph.getEdgeDst( edge );
			aig::NodeData & nodeData = graph.getData( node );
			nodeData.counter += 1;

			if ( nodeData.counter == 2 ) {
				workList.push( node );
			}
		}
	}
};

struct Process {

	aig::Graph & graph;
	int k;
	int c;

	Process( aig::Graph & graph, int k, int c ) : graph( graph ), k( k ), c( c ) { }

	void operator()( aig::GNode node, galois::UserContext<aig::GNode> & ctx ) const {
	
		aig::NodeData & nodeData = graph.getData( node );

		if ( nodeData.type == aig::NodeType::AND ) {

			// Combine Cuts
			combineCuts( graph, node, k, c );

			// Trivial Cut
			std::set< aig::GNode > cut;
			cut.insert( node );
			nodeData.cuts.push_back( cut );

			// NextNodes
			for ( auto edge : graph.out_edges( node ) ) {
			
				aig::GNode nextNode = graph.getEdgeDst( edge );
				aig::NodeData & nextNodeData = graph.getData( nextNode );
				nextNodeData.counter += 1;

				if ( nextNodeData.counter == 2 ) {
					ctx.push( nextNode );
				}
			}
		}
	}

	void combineCuts( aig::Graph & graph, aig::GNode node, int k, int c ) const {

		auto inEdgeIt = graph.in_edge_begin( node );
		aig::GNode lhsNode = graph.getEdgeDst( inEdgeIt );
		inEdgeIt++;
		aig::GNode rhsNode = graph.getEdgeDst( inEdgeIt );
			
		aig::NodeData & lhsData = graph.getData( lhsNode );
		aig::NodeData & rhsData = graph.getData( rhsNode );

		std::vector< std::set< aig::GNode > > resultingCuts;

		for ( auto lhs : lhsData.cuts ) {
			for ( auto rhs : rhsData.cuts ) {
				// Union between lhs and rhs cuts
				std::set< aig::GNode > newCut = lhs;
				for ( auto element : rhs ) {
					newCut.insert( element );
				}
				if ( newCut.size() <= k ) {
					registerAndPruneCuts( newCut, resultingCuts );
				}
			}
		}

		// Just for test
		if ( ( c > 0 ) && ( resultingCuts.size() > c ) ) {
			resultingCuts.erase( resultingCuts.begin()+c, resultingCuts.end() );
		}

		aig::NodeData & nodeData = graph.getData( node );
		nodeData.cuts = resultingCuts;
		// Best Cut
		// nodeData.bestCut = ??;
	}

	void registerAndPruneCuts( std::set< aig::GNode > & cut, std::vector< std::set< aig::GNode > > & cutVector ) const {

		bool dominated, isNewCutDominated = false;
		std::vector< int > dominatedCutsIndices;

		for ( unsigned i = 0; i < cutVector.size(); i++ ) {
			if ( cut.size() < cutVector[i].size() ) {
				dominated = true;
				for ( aig::GNode element : cut ) {
					if ( cutVector[i].find( element ) == cutVector[i].end() ) {
						dominated = false;
						break;
					}
				}
				if ( dominated ) {
					dominatedCutsIndices.push_back( i );
				}
			}
			else {
				dominated = true;
				for ( aig::GNode element : cutVector[i] ) {
					if ( cut.find( element ) == cut.end() ) {
						dominated = false;
						break;
					}
				}
				if ( dominated ) {
					isNewCutDominated = true;
					break;
				}
			}
		}

		if ( isNewCutDominated == false ) {
			for ( unsigned i = 0; i < dominatedCutsIndices.size(); i++ ) {
				int index = dominatedCutsIndices[i];
				cutVector.erase( cutVector.begin()+(index-i) );
			}
			cutVector.push_back( cut );
		}
	}

};

void CutFinder::run( int k, int c ) {

	galois::InsertBag< aig::GNode > workList;
	
	galois::do_all( this->aig.getInputNodes().begin(), this->aig.getInputNodes().end(), Preprocess( this->aig.getGraph(), workList ) );
	galois::for_each_local( workList, Process( this->aig.getGraph(), k, c ), galois::loopname("K-Cuts") );
	// galois::for_each( workList.begin(), workList.end(), Process( this->aig.getGraph(), k, c ) );
	
}

} /* namespace algorithm */

