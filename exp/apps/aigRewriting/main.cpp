#include "parsers/AigParser.h"
#include "writers/AigWriter.h"
#include "subjectgraph/aig/Aig.h"
#include "algorithms/Rewriting.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include <iostream>

#define nOUTPUTS 10			// Number of outputs of the window (not used yet)
#define nINPUTS 10			// Number of inputs of the window
#define nFANOUT 5			// Maximum number of outgoing edges accepted to a node become and active node
#define nLEVELS 6			// Number of level sweeped to build the TFO (Transitive Fanout) of the reference node
#define CUT_SIZE_LIMIT 8		// Maximum size of reconvergence-driven cuts

std::string getFileName( std::string path );
double getMemUsage();

int main( int argc, char * argv[] ) {
	
	if ( argc < 5 ) {
		std::cout << "<nThreads> <verbose (-vX, X={0,1,2,3})> <fileType (-aig or -aag)> <AigInputFile>" << std::endl;
		exit(1);
	}

	const int nTHREADS = atoi( argv[1] );
	int numThreads = Galois::setActiveThreads( nTHREADS );

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

	Galois::Timer T;
	aig::Aig aig;
	std::string fileType( argv[3] );
	std::string path( argv[4] );
	AigParser aigParser( path );
	T.start();	
	if ( fileType.compare( "-aig" ) == 0 ) {
		aigParser.parseAig( aig );
	}
	else {
		if ( fileType.compare( "-aag" ) == 0 ) {
			aigParser.parseAag( aig );
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
		std::cout << "|E|: " << aigParser.getE( aig ) << " (outgoing edges)" << std::endl;
		std::cout << "Parser run time: " << T.get() << " milliseconds" << std::endl << std::endl;
		std::cout << "Using " << numThreads << " threads" << std::endl;
		std::cout << "Windowing Constraints: " << std::endl;
		std::cout << "\tnOUTPUTS: " << nOUTPUTS << std::endl;
		std::cout << "\tnINPUTS: " << nINPUTS << std::endl;
		std::cout << "\tnFANOUT: " << nFANOUT << std::endl;
		std::cout << "\tnLEVELS: " << nLEVELS << std::endl;
		std::cout << "\tCUT_SIZE_LIMIT: " << CUT_SIZE_LIMIT << std::endl;
	}

	aig.writeDot( fileName + ".dot", aig.toDot() );

	if ( verbose == 3 ) {
		std::cout << fileName << std::endl;
	}

	if ( verbose == 1 ) {
		std::cout << fileName << ";" << aigParser.getI() << ";" << aigParser.getO() << ";" << aigParser.getA() << ";";
	}

	T.start();
	algorithm::Rewriting rewriting( aig );
	rewriting.run( nOUTPUTS, nINPUTS, nLEVELS, nFANOUT, CUT_SIZE_LIMIT, verbose );
	T.stop();
	
	if ( verbose <= 1 ) {
		std::cout << T.get() << std::endl;
	}
	else {
		if ( verbose == 2 ) {
			std::cout << "Windowing run time: " << T.get() << " milliseconds" << std::endl;
			std::cout << "Memory usage: " << getMemUsage() << " MB" << std::endl;
		}
	}
/*
	std::cout << "AAG " << std::endl;
	AigWriter aagWriter( fileName + "_rewrited.aag" );
	aagWriter.writeAag( aig );
*/
	std::cout << "Writing final AIG ..." << std::endl;
	AigWriter aigWriter( fileName + "_rewrited.aig" );
	aigWriter.writeAig( aig );

	aig.writeDot( fileName + "_rewrited.dot", aig.toDot() );

	std::cout << "Done." << std::endl;

	return 0;
}

std::string getFileName( std::string path ) {

	std::size_t slash = path.find_last_of("/") + 1;
	std::size_t dot = path.find_last_of(".");
	std::string fileName = path.substr( slash, (dot-slash) );
	return fileName;
}

double getMemUsage() {
	using std::ios_base;
	using std::ifstream;
	using std::string;

	double vm_usage = 0.0;

	ifstream stat_stream("/proc/self/stat", ios_base::in);

	string pid, comm, state, ppid, pgrp, session, tty_nr;
	string tpgid, flags, minflt, cminflt, majflt, cmajflt;
	string utime, stime, cutime, cstime, priority, nice;
	string O, itrealvalue, starttime;

	unsigned long vsize;

	stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
			>> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime
			>> stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue
			>> starttime >> vsize; // don't care about the rest

	stat_stream.close();

	vm_usage = (vsize / 1024.0) / 1024.0;
	return vm_usage;
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
