#include "parsers/AigParser.h"
#include "writers/AigWriter.h"

#include "subjectgraph/aig/Aig.h"
#include "algorithms/CutManager.h"
#include "algorithms/NPNManager.h"
#include "algorithms/RewriteManager.h"
#include "algorithms/PreCompGraphManager.h"
#include "algorithms/ReconvDrivenCut.h"

#include "galois/Galois.h"
#include "galois/Timer.h"

#include <chrono>
#include <iostream>
#include <sstream>

using namespace std::chrono;

void mergeAIGs();

std::string getFileName( std::string path );

void limitFanout( aig::Aig & aig, int limit );

int main( int argc, char * argv[] ) {

	galois::SharedMemSys G; // shared-memory system object initializes global variables for galois
	
	if ( argc < 5 ) {
		std::cout << "<nThreads> <verbose (-vX, X={0,1,2,3})> <fileType (-aig or -aag)> <AigInputFile>" << std::endl;
		exit(1);
	}

	const int nTHREADS = atoi( argv[1] );
	int numThreads = galois::setActiveThreads( nTHREADS );

	std::string verbosity( argv[2] );
	int verbose;
	if ( verbosity.compare( "-v0" ) == 0 ) {
		verbose = 0;
	}
	else {
		if ( verbosity.compare( "-v1" ) == 0 ) {
			verbose = 1;
		}
		else {
			if ( verbosity.compare( "-v2" ) == 0 ) {
				verbose = 2;
			}
			else {
				verbose = 3;
			}
		}
	}

	galois::Timer T;
	T.start();	
	aig::Aig aig;
	std::string fileType( argv[3] );
	std::string path( argv[4] );
	AigParser aigParser( path, aig );

	if ( fileType.compare( "-aig" ) == 0 ) {
		aigParser.parseAig();
	}
	else {
		if ( fileType.compare( "-aag" ) == 0 ) {
			aigParser.parseAag();
		}
		else {
			std::cout << " Unknow input file type!" << std::endl;
		}
	}
	T.stop();

	std::string fileName = getFileName( path ); 

	if ( verbose == 2 ) {
		std::cout << "############## AIG REWRITING ##############" << std::endl;
		std::cout << "Design Name: " << fileName << std::endl;
		std::cout << "|Nodes|: " << aig.getGraph().size() << std::endl;
		std::cout << "|I|: " << aigParser.getI() << std::endl;
		std::cout << "|L|: " << aigParser.getL() << std::endl;
		std::cout << "|O|: " << aigParser.getO() << std::endl;
		std::cout << "|A|: " << aigParser.getA() << std::endl;
		std::cout << "|E|: " << aigParser.getE() << " (outgoing edges)" << std::endl;
		std::cout << "Parser run time: " << T.get() << " milliseconds" << std::endl << std::endl;
	}

	if ( verbose == 3 ) {
		std::cout << fileName << std::endl;
	}

	if ( verbose == 1 ) {
		std::cout << fileName << ";" << aigParser.getI() << ";" << aigParser.getO() << ";" << aigParser.getA() << ";";
	}

	// FANOUT LIMIT
	//limitFanout( aig, 30000 );



/*
	// RECONVERGENCE DRIVE CUT
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	algorithm::ReconvDrivenCut rdcMan( aig );
	rdcMan.run( 4 );

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	std::cout << numThreads << ";" << duration_cast<microseconds>( t2 - t1 ).count() << std::endl;
*/


	// AIG REWRITING
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	std::vector< int > levelHistogram = aigParser.getLevelHistogram();
/*	
	int i = 0;
	for( int value : levelHistogram ) {
		std::cout << i++ << ": " << value << std::endl;
	}
*/

	int K = 4, C = 500;
	bool compTruth = true;
	algorithm::CutManager cutMan( aig, K, C, numThreads, compTruth );
	//algorithm::runKCutOperator( cutMan );
	//cutMan.printAllCuts();
	//cutMan.printCutStatistics();
	//cutMan.printRuntimes();

	algorithm::NPNManager npnMan;

	algorithm::PreCompGraphManager pcgMan( npnMan );
	pcgMan.loadPreCompGraphFromArray();
	pcgMan.processDecompositionGraphs();

	int triesNGraphs = 500;
	bool useZeros = false;
	bool updateLevel = false;

	if ( verbose >= 2 ) {
		std::cout << "############### Configurations ################ " << std::endl;
		std::cout << "K: " << K << std::endl;
		std::cout << "C: " << C << std::endl;
		std::cout << "TriesNGraphs: " << triesNGraphs << std::endl;
		std::cout << "CompTruth: " << ( compTruth ? "yes" : "no" ) << std::endl;
		std::cout << "UseZeroCost: " << ( useZeros ? "yes" : "no" ) << std::endl;
		std::cout << "UpdateLevel: " << ( updateLevel ? "yes" : "no" ) << std::endl;
		std::cout << "nThreads: " << numThreads << std::endl;
	}

	algorithm::RewriteManager rwtMan( aig, cutMan, npnMan, pcgMan, triesNGraphs, useZeros, updateLevel );
	algorithm::runRewriteOperator( rwtMan, levelHistogram );
	//cutMan.printAllCuts();
	//cutMan.printCutStatistics();
	//cutMan.printRuntimes();
	
	long int nPushes = rwtMan.nPushes.reduce();
	std::cout << "nPushes: " << nPushes << std::endl;


	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	long double rewriteTime = duration_cast<microseconds>( t2 - t1 ).count();


	if ( verbose == 0 ) {
		std::cout << fileName << ";" << numThreads << ";" << aig.getNumAnds() << ";" << rewriteTime << std::endl;
	}


	if ( verbose >= 2 ) {
		std::cout << "AND Nodes: " << aig.getNumAnds() << std::endl;
		std::cout << "Runtime (us): " << rewriteTime << std::endl;
		std::cout << "Writing final AIG file..." << std::endl;
	}

	AigWriter aigWriter( fileName + "_rewritten.aig" );
	aigWriter.writeAig( aig );

	//aig.writeDot( fileName + "_rewritten.dot", aig.toDot() );
}


/*
int main( int argc, char * argv[] ) {

	mergeAIGs();
	
	return 0;

}
*/

void mergeAIGs() {

	aig::Aig aig1;
	std::string path1( "/workspace/vnpossani/benchmarks/mtm/mtm2320orig.aig");
	AigParser aigParser1( path1, aig1 );
	aigParser1.parseAig();

	aig::Aig aig2;
	std::string path2( "/workspace/vnpossani/benchmarks/mtm/sixteen.aig");
	AigParser aigParser2( path2, aig2 );
	aigParser2.parseAig();

	aig::Graph & aigGraph1 = aig1.getGraph(); 
	aig::Graph & aigGraph2 = aig2.getGraph(); 


	std::cout << "Step 1" << std::endl;

	// Redefine IDs for AIG2 to be beyond the range of IDs ofAIG1
	std::stack< aig::GNode > stack;
	aig2.computeTopologicalSortForAnds( stack );
	int currentId = aig1.getNodes().size();

	while ( !stack.empty() ) {
		aig::GNode node = stack.top();
		stack.pop();
		aig::NodeData & nodeData = aigGraph2.getData( node, galois::MethodFlag::UNPROTECTED );
		nodeData.id = currentId++;
	}

	for ( aig::GNode po : aig2.getOutputNodes() ) {
		aig::NodeData & poData = aigGraph2.getData( po, galois::MethodFlag::UNPROTECTED );
		poData.id = currentId++;
	}
	// End of IDs redefinition


	std::cout << "Step 2" << std::endl;

	// Redefine the size of resultant AIG
	int newNumNodes = aig1.getNodes().size() + ( aig2.getNodes().size() - aig2.getInputNodes().size() - 1 );
	int newNumPOs = aig1.getOutputNodes().size() + aig2.getOutputNodes().size();
	aig1.getNodes().resize( newNumNodes );	

	// Clone all ANDs and POs from AIG2 to AIG1
	for ( aig::GNode node : aig2.getNodes() ) {

		aig::NodeData & nodeData = aigGraph2.getData( node, galois::MethodFlag::UNPROTECTED );

		if ( nodeData.type == aig::NodeType::AND ) {
			aig::NodeData cloneNodeData = nodeData;
			aig::GNode cloneNode = aigGraph1.createNode( cloneNodeData );
			aigGraph1.addNode( cloneNode );
			aig1.getNodes()[ cloneNodeData.id ] = cloneNode;

			auto inEdge = aigGraph2.in_edge_begin( node );
			aig::GNode lhsNode = aigGraph2.getEdgeDst( inEdge );
			aig::NodeData & lhsNodeData = aigGraph2.getData( lhsNode, galois::MethodFlag::UNPROTECTED );
			bool lhsPol = aigGraph2.getEdgeData( inEdge );
			inEdge++;
			aig::GNode rhsNode = aigGraph2.getEdgeDst( inEdge );
			aig::NodeData & rhsNodeData = aigGraph2.getData( rhsNode, galois::MethodFlag::UNPROTECTED );
			bool rhsPol = aigGraph2.getEdgeData( inEdge );

        	aigGraph1.getEdgeData( aigGraph1.addMultiEdge( aig1.getNodes()[ lhsNodeData.id ], cloneNode, galois::MethodFlag::UNPROTECTED ) ) = lhsPol;
        	aigGraph1.getEdgeData( aigGraph1.addMultiEdge( aig1.getNodes()[ rhsNodeData.id ], cloneNode, galois::MethodFlag::UNPROTECTED ) ) = rhsPol;
		}

		if ( nodeData.type == aig::NodeType::PO ) {
			aig::NodeData cloneNodeData = nodeData;
			aig::GNode cloneNode = aigGraph1.createNode( cloneNodeData );
			aigGraph1.addNode( cloneNode );
			aig1.getNodes()[ cloneNodeData.id ] = cloneNode;
			aig1.getOutputNodes().push_back( cloneNode );

			auto inEdge = aigGraph2.in_edge_begin( node );
			aig::GNode lhsNode = aigGraph2.getEdgeDst( inEdge );
			aig::NodeData & lhsNodeData = aigGraph2.getData( lhsNode, galois::MethodFlag::UNPROTECTED );
			bool lhsPol = aigGraph2.getEdgeData( inEdge );

        	aigGraph1.getEdgeData( aigGraph1.addMultiEdge( aig1.getNodes()[ lhsNodeData.id ], cloneNode, galois::MethodFlag::UNPROTECTED ) ) = lhsPol; 
		}
	}
	// End of node cloning

	std::cout << "Step 3" << std::endl;
		
	// Reset all AND IDs in the topological sort
	aig1.resetAndIds();
	
	std::cout << "Step 4" << std::endl;

	// Redefine IDs and names for all POs of AIG1
	currentId = newNumNodes - newNumPOs;
	for ( aig::GNode po : aig1.getOutputNodes() ) {
		aig::NodeData & poData = aigGraph1.getData( po, galois::MethodFlag::WRITE );
		poData.id = currentId++;
		aig1.getNodes()[ poData.id ] = po;
	}

	aig1.getOutputNames().resize( newNumPOs );
	for ( int i = 0; i < aig1.getOutputNames().size(); i++ ) {
		std::stringstream name;
		name << "o" << i;
		aig1.getOutputNames()[i] = name.str();
	}
	// End of IDs redefinition for POs


	std::cout << "Step 5" << std::endl;

	AigWriter aigWriter( "mtm232016orig.aig" );
	aigWriter.writeAig( aig1 );
}


void limitFanout( aig::Aig & aig, int limit ) {

	aig::Graph & aigGraph = aig.getGraph();
	int idCounter = aig.getNodes().size();
	int size = aig.getInputNodes().size();

	//for ( aig::GNode node : aigGraph ) {
	for ( int i = 0; i < size; i++ ) {

		aig::GNode node = aig.getInputNodes()[i];

		aig::NodeData & nodeData = aigGraph.getData( node, galois::MethodFlag::UNPROTECTED );
	
		if ( nodeData.nFanout > limit ) {
/*			
			aig::GNode lhsNode;
			aig::GNode rhsNode;
			bool lhsPol;
			bool rhsPol;

			if ( nodeData.type == aig::NodeType::AND ) {
				auto inEdgeIt = aigGraph.in_edge_begin( node );
				lhsNode = aigGraph.getEdgeDst( inEdgeIt );
				lhsPol = aigGraph.getEdgeData( inEdgeIt );
				GinEdgeIt++;
				rhsNode = aigGraph.getEdgeDst( inEdgeIt );
				rhsPol = aigGraph.getEdgeData( inEdgeIt );
			}
*/
			auto outEdgeIt = aigGraph.edge_begin( node );
			auto outEdgeEnd = aigGraph.edge_end( node );
			std::advance( outEdgeIt, limit );
			
			int counter = limit;
			aig::GNode newNode = nullptr;
			std::vector< aig::GNode > toRemoveEdges;

			while ( outEdgeIt != outEdgeEnd ) {
				
				if ( counter == limit ) {
					aig::NodeData newNodeData;
					newNodeData.id = idCounter++;
					newNodeData.type = nodeData.type;
					newNodeData.level = nodeData.level;
					newNodeData.nFanout = limit;
					newNode = aigGraph.createNode( newNodeData );
					aigGraph.addNode( newNode );
					aig.getNodes().push_back( newNode );
					counter = 0;

					/*
					if ( nodeData.type == aig::NodeType::AND ) {
						aigGraph.getEdgeData( aigGraph.addMultiEdge( lhsNode, newNode, galois::MethodFlag::UNPROTECTED ) ) = lhsPol;
            			aigGraph.getEdgeData( aigGraph.addMultiEdge( rhsNode, newNode, galois::MethodFlag::UNPROTECTED ) ) = rhsPol;
					}
					else {
					*/
						if ( nodeData.type == aig::NodeType::PI ) {
							aig.getInputNodes().push_back( newNode );
							std::stringstream name; 
							name << aig.getInputNames()[ (nodeData.id-1) ] << "_" << newNodeData.id;
							aig.getInputNames().push_back( name.str() );
							//std::cout << "node " << nodeData.id << " expanding for " << newNodeData.id << " with name " << name << std::endl;
						}
					//}
				}

				aig::GNode dstNode = aigGraph.getEdgeDst( outEdgeIt );
				toRemoveEdges.push_back( dstNode );
				bool outEdgePol = aigGraph.getEdgeData( outEdgeIt, galois::MethodFlag::UNPROTECTED );	
            	aigGraph.getEdgeData( aigGraph.addMultiEdge( newNode, dstNode, galois::MethodFlag::UNPROTECTED ) ) = outEdgePol;
				counter++;
				outEdgeIt++;
			}

			// The last newNode may have fanout smaller than limit
			aig::NodeData & lastNodeData = aigGraph.getData( newNode, galois::MethodFlag::UNPROTECTED );
			lastNodeData.nFanout = std::distance( aigGraph.edge_begin( newNode ), aigGraph.edge_end( newNode ) );

			// Remove the fanout edges beyond limit in the original node
			for ( aig::GNode fanoutNode : toRemoveEdges ) {	
				auto edge = aigGraph.findEdge( node, fanoutNode );
				aigGraph.removeEdge( node, edge );
				
				//int nFanin = std::distance( aigGraph.in_edge_begin( fanoutNode ), aigGraph.in_edge_end( fanoutNode ) );
				//int nFanout = std::distance( aigGraph.edge_begin( node ), aigGraph.edge_end( node ) );
				//assert( nFanin == 2 );
				//assert( nFanout > 0 );
			}
			nodeData.nFanout = limit;
		}	
	}
}


std::string getFileName( std::string path ) {

	std::size_t slash = path.find_last_of("/") + 1;
	std::size_t dot = path.find_last_of(".");
	std::string fileName = path.substr( slash, (dot-slash) );
	return fileName;
}


// ################### K-CUT ################### //
// T.start();
// algorithm::CutFinder cutFinder( aig );
// cutFinder.run( K, C );
// T.stop();

// aig::Graph & graph = aig.getGraph();
// for ( auto po : aig.getOutputNodes() ) {
// 	auto inEdgeIt = graph.in_edge_begin( po );
// 	aig::GNode inNode = graph.getEdgeDst( inEdgeIt );
// 	aig::NodeData & inNodeData = graph.getData( inNode );
// 	std::cout << "Node " << inNodeData.id << " -> Cuts = { ";
// 	for ( auto cut : inNodeData.cuts ) {
// 		std::cout << "{ ";
// 		for ( aig::GNode node : cut ) {
// 			aig::NodeData & nodeData = graph.getData( node );
// 			std::cout << nodeData.id << " ";
// 		}
// 		std::cout << "} ";
// 	}
// 	std::cout << " }" << std::endl;
// }
