#include "Rewriting.h"
#include "Windowing.h"
#include "../functional/FunctionHandler.h"
#include "../functional/BitVectorPool.h"
#include "../functional/FunctionUtil.h"
#include "../factoring/EquationOptimizer.h"
#include "../subjectgraph/cmxaig/Cmxaig.h"
#include "../mixer/Mixer.h"
#include "../covering/CoveringFinder.h"
#include "../covering/CoveringChooser.h"
#include "../covering/CoveringHandler.h"
#include "../mixer/Mixer.h"
#include "../writers/AigWriter.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Timer.h"
#include "Galois/Accumulator.h"

#include <iostream>
#include <unordered_set>
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

typedef struct expSupInfo {
	std::string expression;
	std::unordered_set< int > support;
	bool needsOpt;

	expSupInfo() {
		needsOpt = true;
	}

} ExpSupInfo;



// ################################################## AUXILIARY STRUCTS AND CLASSES ################################################## //
galois::GAccumulator<size_t> windowHistogram[ WIN_BUCKET_NUM ];
galois::GAccumulator<size_t> fanoutHistogram[ FANOUT_BUCKET_NUM ];

typedef std::unordered_set< aig::GNode, std::hash<aig::GNode>, std::equal_to<aig::GNode>, galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeSet;
typedef std::deque< aig::GNode, galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeDeque;
typedef std::unordered_map< int, ExpSupInfo > ExpSupMap;
typedef Functional::BitVectorPool BitVectorPool;
typedef Functional::word word;

// ################################################## FUNCTION HEADEARS ################################################## //

void addInfoWindowHistogram( GNodeSet & window );
void addInfoFanoutHistogram( int fanout );

void optimize( aig::Aig & aig, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots, int nodeID );

void findWindowExpSup( aig::Graph & graph, GNodeSet & leaves, GNodeSet & roots, ExpSupMap & expSupMap );
void findExpSup( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, ExpSupMap & expSupMap );

void findWindowExpSupHist( aig::Graph & graph, GNodeSet & leaves, GNodeSet & roots, ExpSupMap & expSupMap );
void findExpSupHist( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::string & currentExp, std::unordered_set< int > & currentSup, std::unordered_map< std::string, int > & literalHi1st, bool propagPolarity );

void replace( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots, std::unordered_map< std::string, aig::GNode > & leaveMap, std::unordered_map< std::string, aig::GNode > & rootMap, SubjectGraph::Cmxaig * cmxaig );
aig::GNode createNodes( SubjectGraph::Node* currentNode, aig::Graph & graph, std::unordered_map< std::string, aig::GNode > & leaveMap, std::unordered_map< std::string, aig::GNode > & rootMap, std::unordered_map< int, aig::GNode > & visited, std::vector< int > & availableIDs );


void buildWindowFunctions( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots );
word * buildFunction( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::unordered_map< int, Functional::word* > & computedFunctions, BitVectorPool & functionPool, int nWords );
void dumpLogicFuntions( std::string hexaText, std::string label );


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

struct FindActiveNodes {
	
	aig::Graph & graph;
	galois::InsertBag< aig::GNode > & workList;
	int maxFanout;
  
	FindActiveNodes( aig::Graph & graph, galois::InsertBag< aig::GNode > & workList, int nFanout ) : graph( graph ), workList( workList ) { 
		maxFanout = nFanout;
	}
  
	void operator()( aig::GNode node ) const {
			
		aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::READ );
		int nodeFanout = std::distance( graph.out_edges( node ).begin(), graph.out_edges( node ).end() );

		//if ( (nodeData.type == aig::NodeType::AND) && (nodeFanout > 1) && (nodeFanout <= maxFanout) ) {
		//if ( (nodeFanout > 1) && (nodeFanout <= maxFanout) ) {
		//if ( nodeFanout <= maxFanout) {
		if ( (nodeData.type != aig::NodeType::PO) && (nodeFanout <= maxFanout) ) {
			workList.push( node );
		}
		
		//addInfoFanoutHistogram( nodeFanout );
	}
};

struct AigRewriting {
	typedef int tt_does_not_need_aborts;
	typedef int tt_does_not_need_push;
	typedef int tt_needs_per_iter_alloc;
	typedef galois::PerIterAllocTy Alloc;

	aig::Aig & aig;
	aig::Graph & graph;
	int nInputs;
	int nOutputs;
	int nLevels;
	int cutSizeLimit;

	AigRewriting( aig::Aig & aig, int nInputs, int nOutputs, int nLevels, int cutSizeLimit ) 
		    : aig( aig ), graph( aig.getGraph() ), nInputs( nInputs ), nOutputs( nOutputs ), nLevels( nLevels ), cutSizeLimit( cutSizeLimit ) { }
					
	void operator()( aig::GNode activeNode, galois::UserContext< aig::GNode > & ctx ) const {

		if ( !graph.containsNode( activeNode, galois::MethodFlag::WRITE ) ) {
			return;
		}

		galois::PerIterAllocTy & allocator = ctx.getPerIterAlloc();

		std::cout << "Windowing ..." << std::endl;
		algorithm::Windowing windowing( graph, allocator, nInputs, nOutputs, nLevels, cutSizeLimit );
		bool accepted = windowing.computeReconvDrivenWindow( activeNode );
		//bool accepted = windowing.computeWindowTEST( activeNode );

		if ( !accepted ) {
			std::cout << "Window does not meet constraints, discading window ..." << std::endl;
			return;
		}

		// FIXME : Locks all nodes in the window.
		for ( auto leaf : windowing.getLeaves() ) {
			if ( !graph.containsNode( leaf, galois::MethodFlag::WRITE ) ) { // The contaisNode method acquires the lock!
				std::cout << "Discarding Window." << std::endl;
				return;
			}
			aig::NodeData & leafData = graph.getData( leaf, galois::MethodFlag::WRITE );
		}
	
		for ( auto node : windowing.getWindow() ) {
			if ( !graph.containsNode( node, galois::MethodFlag::WRITE ) ) {  // The contaisNode method acquires the lock!
				std::cout << "Discarding Window." << std::endl;
				return;
			}
			aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
		}
	
		for ( auto root : windowing.getRoots() ) {
			for ( auto outEdge : graph.out_edges( root ) ) {
				aig::GNode outNode = graph.getEdgeDst( outEdge );
				if ( !graph.containsNode( outNode, galois::MethodFlag::WRITE ) ) { // The contaisNode method acquires the lock!
					std::cout << "Discarding Window." << std::endl;
					return;
				}
				aig::NodeData & outNodeData = graph.getData( outNode, galois::MethodFlag::WRITE );
			}
		}


		aig::NodeData & activeNodeData = graph.getData( activeNode, galois::MethodFlag::WRITE );
		
		//windowing.writeWindowDot( windowing.toDot(), std::to_string( activeNodeData.id) );

		addInfoWindowHistogram( windowing.getWindow() );

		optimize( aig, windowing.getWindow(), windowing.getLeaves(), windowing.getRoots(), activeNodeData.id );
	}
};

// ################################################## RUNNER ################################################## //

Rewriting::Rewriting( aig::Aig & aig ) : aig( aig ) { }

Rewriting::~Rewriting() { }

void Rewriting::run( int nOutputs, int nInputs, int nLevels, int nFanout, int cutSizeLimit, int verbose ) {
	
	galois::InsertBag< aig::GNode > workList;

	// PREPROCESS //
	galois::do_all( this->aig.getGraph().begin(), this->aig.getGraph().end(), FindActiveNodes( this->aig.getGraph(), workList, nFanout ) ); // Starting from DAG nodes

	if ( verbose == 1 ) {
		std::cout << std::distance(workList.begin(), workList.end()) << ";";
	}
	else {
		if ( verbose == 2 ) {
			std::cout << "Number of Active Nodes: " << std::distance(workList.begin(), workList.end()) << std::endl;
		}
	}

	// PROCESS //
	galois::for_each( workList.begin(), workList.end(), AigRewriting( this->aig, nInputs, nOutputs, nLevels, cutSizeLimit ) ); // Active nodes from Worklist

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


// ################################################## REWRITING ################################################## //
void optimize( aig::Aig & aig, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots, int nodeID ) {

	aig::Graph & graph = aig.getGraph();
	std::unordered_map< std::string, aig::GNode > leafMap;
	std::unordered_map< std::string, aig::GNode > rootMap;
	ExpSupMap expSupMap;

	findWindowExpSup( graph, leaves, roots, expSupMap );
	//findWindowExpSupHist( graph, leaves, roots, expSupMap );

	// MIXER
	int nVarsW = leaves.size();
	int nWordsW = Functional::wordNum( nVarsW );
	int poolSizeW = 1000;
	std::vector< std::string > varSetW;
	for ( auto leaf : leaves ) {
		aig::NodeData & leafData = graph.getData( leaf, galois::MethodFlag::WRITE );
		std::string inputName = ("i"+std::to_string( leafData.id ));
		varSetW.push_back( inputName );
		leafMap.insert( std::make_pair( inputName, leaf ) );
	}

	//std::sort( varSetW.begin(), varSetW.end() );

	SubjectGraph::Mixer mixer( nVarsW, poolSizeW, varSetW );
	
	// DEBUG
	//SubjectGraph::Mixer mixerDebug( nVarsW, poolSizeW, varSetW );
	// DEBUG

	for ( auto root : roots ) {
		aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::WRITE );
		std::string outputName = ("o"+std::to_string(rootData.id));
		rootMap.insert( std::make_pair( outputName, root ) );		

		ExpSupInfo & expSupInfo = expSupMap[ rootData.id ];
		std::string expression = expSupInfo.expression;	
		std::unordered_set< int > support = expSupInfo.support;

		std::cout << std::endl << "Original Solution for root " << rootData.id << ":\n\t" << expression << std::endl;
		mixer.mix( expression, outputName );
	
		// DEBUG	
		//mixerDebug.mix( expression, outputName );		
		// DEBUG
	
		if ( !expSupInfo.needsOpt ) {
			continue;
		}
	
		int nVarsF = support.size();
		int nWordsF = Functional::wordNum( nVarsF );
		int poolSizeF = 10000;
		int nSolutions = 10;
		std::vector< std::string > varSetF;
		for ( int element : support ) {
			std::string inputName = ("i"+std::to_string(element));
			varSetF.push_back( inputName );
		}

		//std::sort( varSetF.begin(), varSetF.end() );

		std::unordered_map< std::string, std::pair< Functional::word*, unsigned int > > literals;
		Functional::BitVectorPool functionPool( poolSizeF, nWordsF );
		Functional::createLiterals( varSetF, literals, functionPool );
		Functional::FunctionUtil parser( literals, functionPool, nVarsF, nWordsF );
		Functional::word * f = parser.parseExpression( expression );
		//std::cout << "Fin = " << Functional::toHex( f, nWordsF ) << std::endl;

		if ( !Functional::isConstZero( f, nVarsF ) && !Functional::isConstOne( f, nVarsF ) ) {
			// FACTORIZATION
			std::cout << "\tFactoring ..." << std::endl;
			Factoring::EquationOptimizer factoring( f, varSetF, literals, functionPool, nSolutions );
			factoring.optimizeEquation();

			if ( factoring.getSolutions().size() > 0 ) {

				std::cout << "\tMixing New Solutions: " << std::endl;
				auto & solutions = factoring.getSolutions();
				std::cout << "\t" << solutions[0].getEquation() << std::endl;
				mixer.mix( solutions[0].getEquation() );
				
				for ( int i = 1; i < solutions.size(); i++ ) {
					if ( (solutions[i].getLiterals() - solutions[0].getLiterals()) > 2 ) {
						break;
					}
					std::cout << "\t" <<  solutions[i].getEquation() << std::endl;
					mixer.mix( solutions[i].getEquation() );
				}
			}
		}
	}


	SubjectGraph::Cmxaig * cmxaig = mixer.getCmxaig();
/*
	std::cout << "################### MIXE ################### " << std::endl;
	for ( auto output : cmxaig->getOutputs() ) {	
		SubjectGraph::ChoiceNode * choiceNode = (SubjectGraph::ChoiceNode*) output->getInNodes()[0];
		bool polarity = output->getInEdges()[0];
		word * f = choiceNode->getFunctionPtr();
		std::cout << "polarity: " << polarity << std::endl;
		std::cout << "Fout = " << Functional::toHex( f, nWordsW ) << std::endl;
		word * fNeg = new word[ nWordsW ];
		Functional::NOT( fNeg, f, nWordsW );
		std::cout << "Fout = " << Functional::toHex( fNeg, nWordsW ) << std::endl;
	
	}
	std::cout << "############################################ " << std::endl;
*/
	std::string fileName = "/workspace/vnpossani/GALOIS_DUMPS/dots/graph_" + std::to_string( nodeID ) + "_mixed_" + std::to_string( window.size() );
	std::ofstream dotFileMixer( fileName+".dot" );
	dotFileMixer << cmxaig->toDot();
/*
	// DEBUG
	SubjectGraph::Cmxaig * cmxaigDebug = mixerDebug.getCmxaig();
	dotFileMixer.close();

	std::cout << "################# ORIGINAL ################# " << std::endl;
	for ( auto output : cmxaigDebug->getOutputs() ) {	
		SubjectGraph::ChoiceNode * choiceNode = (SubjectGraph::ChoiceNode*) output->getInNodes()[0];
		bool polarity = output->getInEdges()[0];
		word * f = choiceNode->getFunctionPtr();
		std::cout << "polarity: " << polarity << std::endl;
		std::cout << "Fout = " << Functional::toHex( f, nWordsW ) << std::endl;
		word * fNeg = new word[ nWordsW ];
		Functional::NOT( fNeg, f, nWordsW );
		std::cout << "Fout = " << Functional::toHex( fNeg, nWordsW ) << std::endl;
	
	}
	std::cout << "############################################ " << std::endl;

	std::string debugName = "/workspace/vnpossani/GALOIS_DUMPS/dots/graph_" + std::to_string( nodeID ) + "_debug_" + std::to_string( window.size() );
	std::ofstream dotDebug( debugName+".dot" );
	dotDebug << cmxaigDebug->toDot();
	dotDebug.close();
	delete cmxaigDebug;
	// DEBUG
*/
	Covering::coveringMananger coveringMananger;
	Covering::setCoveringMananger( cmxaig->getAndCounter(), cmxaig->getXorCounter(), cmxaig->getMuxCounter(), coveringMananger );
	Covering::CoveringVector_2D outputCoverings;
	int poolSizeC = 10000;
	BitVectorPool coveringPool( poolSizeC, coveringMananger.nWords );

	// COVERING FINDER
	std::cout << "Computing Coverings ..." << std::endl;
	Covering::CoveringFinder coveringFinder( *cmxaig, coveringPool, coveringMananger );
	bool abandon = coveringFinder.run( outputCoverings );

	if ( abandon || outputCoverings.empty() ) {
		std::cout << "Covering Not Found, Abandoning Window ..." << std::endl;
		delete cmxaig;
		return;
	}

	// COVERING CHOOSER
	std::cout << "Choosing a Covering ..." << std::endl;
	float alpha = 1, beta = 2, gamma = 3;
	std::string repport;
	Covering::CoveringChooser coveringChooser( *cmxaig, coveringPool, coveringMananger, outputCoverings );
	
	int nCoverings = 1;	
	for ( auto vec : outputCoverings ) {
		nCoverings *= vec.size();
	}
	
	if ( nCoverings < 10000 ) {
		std::cout << "Exact Covering for " << nCoverings << " coverings" << std::endl;
		repport = coveringChooser.run( alpha, beta, gamma );
	}
	else {
		int nBestCovPerOutput = 5; // choose the X best coverings per output (local) and then comput the global coverings.
		std::cout << "Heuristic Covering (" << nBestCovPerOutput << ") for " << nCoverings << " coverings" << std::endl;
		repport = coveringChooser.run( alpha, beta, gamma, nBestCovPerOutput );
	}
	//std::cout << repport << std::endl;

	cmxaig = &(coveringChooser.getCoveredCmxaig());

	if ( window.size() <= coveringChooser.getFinalNodeCount() ) {
		std::cout << "Window Improvement: 0" << std::endl;
	}
	else {
		std::cout << "Window Improvement: " << window.size() - coveringChooser.getFinalNodeCount() << std::endl;
		// REPLACE
		std::cout << "Replacing Gaph ..." << std::endl;
		replace( graph, window, leaves, roots, leafMap, rootMap, cmxaig );
	}

	std::ofstream dotFileCovering( fileName+"_covered.dot" );
	dotFileCovering << cmxaig->toDot();
	dotFileCovering.close();

	aig.writeDot( fileName+"_writed.dot", aig.toDot() );

	AigWriter aigWriter( "node_" + std::to_string( nodeID ) +  "_" + std::to_string( window.size() ) +"_rewrited.aig" );
	aigWriter.writeAig( aig );

	delete cmxaig;
	
	std::cout << "Window Rewriting Done." << std::endl;
}

void findWindowExpSup( aig::Graph & graph, GNodeSet & leaves, GNodeSet & roots, ExpSupMap & expSupMap ) {

	for ( aig::GNode leave : leaves ) {	
		aig::NodeData & leaveData = graph.getData( leave, galois::MethodFlag::WRITE );
		ExpSupInfo expSupInfo;
		expSupInfo.support.insert( leaveData.id );
		expSupInfo.expression = ("i"+std::to_string( leaveData.id ));
		expSupMap.insert( std::make_pair( leaveData.id, expSupInfo ) );
	}

	for ( aig::GNode root : roots ) {

		aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::WRITE );
		
		findExpSup( graph, leaves, root, expSupMap );

		if ( rootData.andBehavior == false ) {
			ExpSupInfo expSupInfo = expSupMap[ rootData.id ];
			expSupInfo.expression = "!" + expSupInfo.expression; 
			expSupMap[ rootData.id ] = expSupInfo;
		}
	
		if ( VERBOSE ) {
			std::cout << "ID: " << rootData.id << " -> Value: " << expSupMap[ rootData.id ].expression << std::endl;
			for ( auto id : expSupMap[ rootData.id ].support ) {
				std::cout << id << " ";
			}
			std::cout << std::endl;
		}
	}
}

void findExpSup( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, ExpSupMap & expSupMap ) {

	ExpSupInfo expSupInfo;

	for ( auto inEdge : graph.in_edges( currentNode ) ) {

		aig::GNode inNode = graph.getEdgeDst( inEdge );
		aig::NodeData & inNodeData = graph.getData( inNode, galois::MethodFlag::WRITE );
		bool inEdgeData = graph.getEdgeData( inEdge, galois::MethodFlag::WRITE );

		auto it = expSupMap.find( inNodeData.id );
		if ( it == expSupMap.end() ) {
			findExpSup( graph, leaves, inNode, expSupMap );
		}

		auto element = expSupMap[ inNodeData.id ];

		if ( inEdgeData == true ) {
			if ( expSupInfo.expression.empty() ) {
				expSupInfo.expression = element.expression;
			}
			else {
				expSupInfo.expression = "(" + expSupInfo.expression + "*" + element.expression + ")";
			}
		}
		else {
			if ( expSupInfo.expression.empty() ) {
				expSupInfo.expression = "!" + element.expression;
			}
			else {
				expSupInfo.expression = "(" + expSupInfo.expression + "*!" + element.expression + ")";
			}	
		}

		expSupInfo.support.insert( element.support.begin(), element.support.end() );
	}

	aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::WRITE );
	expSupMap.insert( std::make_pair( currentNodeData.id, expSupInfo ) );
}



// HISTOGRAM
void findWindowExpSupHist( aig::Graph & graph, GNodeSet & leaves, GNodeSet & roots, ExpSupMap & expSupMap ) {

	for ( aig::GNode root : roots ) {

		aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::WRITE );
		
		std::string expression = "";
		std::unordered_set< int > support;
		std::unordered_map< std::string, int > literalHist; 

		findExpSupHist( graph, leaves, root, expression, support, literalHist, rootData.andBehavior );

		ExpSupInfo expSupInfo;
		rootData.andBehavior ? expSupInfo.expression = expression : expSupInfo.expression = "!" + expression;
		expSupInfo.support = support;
		expSupInfo.needsOpt = false;
	
		// EVALUATE THE LITERAL HISTOGRAM IN ORDER TO DETERMIN IF THE EXPRESSION NEEDS TO BE OPTIMIZED OR NOT.
		for ( auto entry : literalHist ) {
			//std::cout << entry.first << " : " << entry.second << std::endl;
			if ( entry.second > 1 ) {
				expSupInfo.needsOpt = true;
			}
		}

		// REGISTER SOLUTION
		expSupMap.insert( std::make_pair( rootData.id, expSupInfo ) );

		if ( VERBOSE ) {
			std::cout << "ID: " << rootData.id << " -> Value: " << expSupMap[ rootData.id ].expression << std::endl;
			for ( auto id : expSupMap[ rootData.id ].support ) {
				std::cout << id << " ";
			}
			std::cout << std::endl;
		}
	}
}

void findExpSupHist( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::string & currentExp, std::unordered_set< int > & currentSup,
		     std::unordered_map< std::string, int > & literalHist, bool propagPolarity ) {

	if ( leaves.count( currentNode ) > 0 ) {
		aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::WRITE );
		currentSup.insert( currentNodeData.id );
		currentExp = ("i"+std::to_string( currentNodeData.id ));
		std::string lit;

		if ( propagPolarity ) {
			lit = currentExp;
		}
		else {
			lit = ("!"+currentExp);
		}			
	
		auto litIt = literalHist.find( lit );

		if ( litIt == literalHist.end() ) {
			literalHist.insert( std::make_pair( lit, 1) );
		}
		else {
			literalHist[ lit ] = literalHist[ lit ] + 1;
		}

		return;
	}
	
	for ( auto inEdge : graph.in_edges( currentNode ) ) {
	
		aig::GNode inNode = graph.getEdgeDst( inEdge );
		aig::NodeData & inNodeData = graph.getData( inNode, galois::MethodFlag::WRITE );
		bool inEdgeData = graph.getEdgeData( inEdge, galois::MethodFlag::WRITE );
		std::string nextExp = "";
		std::unordered_set< int > nextSup;

		findExpSupHist( graph, leaves, inNode, nextExp, nextSup, literalHist, (inEdgeData == propagPolarity) );

		if ( inEdgeData == true ) {
			if ( currentExp.empty() ) {
				currentExp = nextExp;
			}
			else {
				currentExp = "(" + currentExp + "*" + nextExp + ")";
			}
		}
		else {
			if ( currentExp.empty() ) {
				currentExp = "!" + nextExp;
			}
			else {
				currentExp = "(" + currentExp + "*!" + nextExp + ")";
			}	
		}

		currentSup.insert( nextSup.begin(), nextSup.end() );
	}
}



void replace( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots, std::unordered_map< std::string, aig::GNode > & leaveMap, std::unordered_map< std::string, aig::GNode > & rootMap, SubjectGraph::Cmxaig * cmxaig ) {

	// Remove nodes
	std::vector< int > availableIDs;
	for ( auto node : window ) {
		if ( roots.count( node ) == 0 ) {
			aig::NodeData & nodeData = graph.getData( node, galois::MethodFlag::WRITE );
			availableIDs.push_back( nodeData.id );
			graph.removeNode( node, galois::MethodFlag::WRITE );
		}
	}

	for ( auto root : roots ) {
		aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::WRITE );
		for ( auto inEdge : graph.in_edges( root ) ) {
			aig::GNode inNode = graph.getEdgeDst( inEdge );
			aig::NodeData & inNodeData = graph.getData( inNode, galois::MethodFlag::WRITE ); 
			std::cout << std::endl << "Removing edge between " << inNodeData.id << " -> " << rootData.id << std::endl;
			auto edgeIt = graph.findEdge( inNode, root );

			if ( edgeIt == graph.edge_end( inNode ) ) {
				std::cout << "Edge not Found!" << std::endl;
			}
	
			graph.removeEdge( inNode, edgeIt, galois::MethodFlag::WRITE );
			
			edgeIt = graph.findEdge( inNode, root );
			if ( edgeIt == graph.edge_end( inNode ) ) {
				std::cout << "The edge was removed!" << std::endl;
			}
			else { 
				std::cout << "The edge was NOT removed!" << std::endl;
			}
		}
	}
/*
	std::unordered_map< int, aig::GNode > visited;
	std::vector< SubjectGraph::Node* > topAnds;
	
	for ( auto output : cmxaig->getOutputs() ) {
		SubjectGraph::Node * choiceNode = output->getInNodes()[0];
		SubjectGraph::Node * andNode = choiceNode->getInNodes()[0];

		aig::GNode root = rootMap[ output->getName() ];
		rootMap.erase( output->getName() );
		rootMap.insert( std::make_pair( andNode->getName(), root ) );
	}

	for ( auto output : cmxaig->getOutputs() ) {	
		SubjectGraph::Node * choiceNode = output->getInNodes()[0];
		SubjectGraph::Node * andNode = choiceNode->getInNodes()[0];

		SubjectGraph::Node * lhsIn = andNode->getInNodes()[0];
		SubjectGraph::Node * rhsIn = andNode->getInNodes()[1];

		aig::GNode lhsAnd = createNodes( lhsIn, graph, leaveMap, rootMap, visited, availableIDs );
		aig::GNode rhsAnd = createNodes( rhsIn, graph, leaveMap, rootMap, visited, availableIDs );

		bool lhsPol, rhsPol;

		if ( lhsIn->isChoiceNode() ) {
			if ( lhsIn->getInEdges()[0] == andNode->getInEdges()[0] ) {
				//lhsPol = andNode->getInEdges()[0];
				lhsPol = true;
			}
			else {
				lhsPol = false;
			}
		}
		else {
			lhsPol = andNode->getInEdges()[0];
		}
		
		if ( rhsIn->isChoiceNode() ) {
			if ( rhsIn->getInEdges()[0] == andNode->getInEdges()[1] ) {
				//rhsPol = andNode->getInEdges()[1];
				rhsPol = true;
			}
			else {
				rhsPol = false;
			}	
		}
		else {
			rhsPol = andNode->getInEdges()[1];
		}

		aig::GNode root = rootMap[ andNode->getName() ];	
		aig::NodeData & rootData = graph.getData( root, galois::MethodFlag::WRITE );

		// Add edges
		graph.getEdgeData( graph.addEdge( lhsAnd, root ), galois::MethodFlag::WRITE ) = lhsPol;
		graph.getEdgeData( graph.addEdge( rhsAnd, root ), galois::MethodFlag::WRITE ) = rhsPol;

		// Update outputEdges of the topAND (root)
		bool choicePolarity;

		if ( output->getInEdges()[0] == choiceNode->getInEdges()[0] ) {
			//choicePolarity = output->getInEdges()[0];
			choicePolarity = true;
		}
		else {
			choicePolarity = false;
		}

		if ( rootData.andBehavior != choicePolarity ) {
			for ( auto outEdge : graph.out_edges( root ) ) {
				bool & edgeData = graph.getEdgeData( outEdge, galois::MethodFlag::WRITE );
				edgeData = !edgeData;
			}
			rootData.andBehavior = !rootData.andBehavior;
		}
	}
*/
}

aig::GNode createNodes( SubjectGraph::Node* currentNode, aig::Graph & graph, std::unordered_map< std::string, aig::GNode > & leaveMap, std::unordered_map< std::string, aig::GNode > & rootMap, std::unordered_map< int, aig::GNode > & visited, std::vector< int > & availableIDs ) {

	auto itV = visited.find( currentNode->getId() );
	if ( itV != visited.end() ) {
		return itV->second;
	}

	if ( currentNode->isInputNode() ) {
		std::string inputName = currentNode->getName();
		return leaveMap[ inputName ];
	}

	auto itR = rootMap.find( currentNode->getName() );
	if ( itR != rootMap.end() ) {
		return itR->second;
	}

	if ( currentNode->isChoiceNode() ) {
		return createNodes( currentNode->getInNodes()[0], graph, leaveMap, rootMap, visited, availableIDs );
	}

	aig::GNode newNode;

	if ( currentNode->isAndNode() ) {
		SubjectGraph::Node * lhsIn = currentNode->getInNodes()[0];
		SubjectGraph::Node * rhsIn = currentNode->getInNodes()[1];
		aig::GNode lhsAnd = createNodes( lhsIn, graph, leaveMap, rootMap, visited, availableIDs );
		aig::GNode rhsAnd = createNodes( rhsIn, graph, leaveMap, rootMap, visited, availableIDs );
	
		bool lhsPol, rhsPol;

		if ( lhsIn->isChoiceNode() ) {
			if ( lhsIn->getInEdges()[0] == currentNode->getInEdges()[0] ) {
				//lhsPol = currentNode->getInEdges()[0];
				lhsPol = true;
			}
			else {
				lhsPol = false;
			}
		}
		else {
			lhsPol = currentNode->getInEdges()[0];
		}
		
		if ( rhsIn->isChoiceNode() ) {
			if ( rhsIn->getInEdges()[0] == currentNode->getInEdges()[1] ) {
				//rhsPol = currentNode->getInEdges()[1];
				rhsPol = true;
			}
			else {
				rhsPol = false;
			}	
		}
		else {
			rhsPol = currentNode->getInEdges()[1];
		}

		// Add nodes
		aig::NodeData newNodeData;
		newNodeData.id = availableIDs.back();
		availableIDs.pop_back();
		newNodeData.counter = 0;  // FIXME
		newNodeData.type = aig::NodeType::AND;
		newNode = graph.createNode( newNodeData );
		graph.addNode( newNode, galois::MethodFlag::WRITE );

		// Add edges
		graph.getEdgeData( graph.addEdge( lhsAnd, newNode ), galois::MethodFlag::WRITE ) = lhsPol;
		graph.getEdgeData( graph.addEdge( rhsAnd, newNode ), galois::MethodFlag::WRITE ) = rhsPol;
	}
/*
	if ( currentNode->isXorNode() ) {
		SubjectGraph::Node * lhsIn = currentNode->getInNodes()[0];
		SubjectGraph::Node * rhsIn = currentNode->getInNodes()[1];
		aig::GNode lhsAnd = createNodes( lhsIn, graph, leaveMap, rootMap, visited, availableIDs );
		aig::GNode rhsAnd = createNodes( rhsIn, graph, leaveMap, rootMap, visited, availableIDs );
	
		bool lhsPol, rhsPol;

		if ( lhsIn->isChoiceNode() ) {
			if ( lhsIn->getInEdges()[0] == currentNode->getInEdges()[0] ) {
				//lhsPol = currentNode->getInEdges()[0];
				lhsPol = true;
			}
			else {
				lhsPol = false;
			}

			SubjectGraph::Node * choiceIn = lhsIn->getInNodes[0];

			if ( choiceIn->isXor() && (lhsPol == true) ) {
				lhsPol = false;
			}
			
			if ( choiceIn->isXor() && (lhsPol == false) ) {
				lhsPol = true;
			}
		}
		else {
			lhsPol = currentNode->getInEdges()[0];
		}
		
		if ( rhsIn->isChoiceNode() ) {
			if ( rhsIn->getInEdges()[0] == currentNode->getInEdges()[1] ) {
				//rhsPol = currentNode->getInEdges()[1];
				rhsPol = true;
			}
			else {
				rhsPol = false;
			}	

			SubjectGraph::Node * choiceIn = rhsIn->getInNodes[0];

			if ( choiceIn->isXor() && (rhsPol == true) ) {
				rhsPol = false;
			}
			
			if ( choiceIn->isXor() && (rhsPol == false) ) {
				rhsPol = true;
			}
		}
		else {
			rhsPol = currentNode->getInEdges()[1];
		}

		// Add nodes
		aig::NodeData auxLhsNodeData;
		auxLhsNodeData.id = availableIDs.back();
		availableIDs.pop_back();
		auxLhsNodeData.counter = 0;  // FIXME
		auxLhsNodeData.type = aig::NodeType::AND;
		auxLhsNode = graph.createNode( auxLhsNodeData );
		graph.addNode( auxLhsNode, galois::MethodFlag::WRITE );
	
		graph.getEdgeData( graph.addEdge( lhsAnd, auxLhsNode ), galois::MethodFlag::WRITE ) = lhsPol;
		graph.getEdgeData( graph.addEdge( rhsAnd, auxLhsNode ), galois::MethodFlag::WRITE ) = !rhsPol;

		aig::NodeData auxRhsNodeData;
		auxRhsNodeData.id = availableIDs.back();
		availableIDs.pop_back();
		auxRhsNodeData.counter = 0;  // FIXME
		auxRhsNodeData.type = aig::NodeType::AND;
		auxRhsNode = graph.createNode( auxRhsNodeData );
		graph.addNode( auxRhsNode, galois::MethodFlag::WRITE );

		graph.getEdgeData( graph.addEdge( lhsAnd, auxRhsNode ), galois::MethodFlag::WRITE ) = !lhsPol;
		graph.getEdgeData( graph.addEdge( rhsAnd, auxRhsNode ), galois::MethodFlag::WRITE ) = rhsPol;

		aig::NodeData newNodeData;
		newNodeData.id = availableIDs.back();
		availableIDs.pop_back();
		newNodeData.counter = 0;  // FIXME
		newNodeData.type = aig::NodeType::AND;
		newNode = graph.createNode( newNodeData );
		graph.addNode( newNode, galois::MethodFlag::WRITE );

		// Add edges
		graph.getEdgeData( graph.addEdge( auxLhsNode, newNode ), galois::MethodFlag::WRITE ) = false;
		graph.getEdgeData( graph.addEdge( auxRhsNode, newNode ), galois::MethodFlag::WRITE ) = false;

		// FIXME Set output polarity
	}
*/

/*
	if ( currentNode->isMuxNode() ) {

	}
*/

	visited.insert( std::make_pair( currentNode->getId(), newNode ) );
	return newNode;
}
// ################################################################################################################# //


void buildWindowFunctions( aig::Graph & graph, GNodeSet & window, GNodeSet & leaves, GNodeSet & roots ) {

	std::vector< std::string > varSet;
	std::unordered_map< std::string, std::pair< word*, unsigned int > > literals;
	std::unordered_map< int, word* > computedFunctions;

	for ( aig::GNode leave : leaves ) {
		aig::NodeData & leaveData = graph.getData( leave, galois::MethodFlag::READ );
		varSet.push_back( "i"+std::to_string( leaveData.id ) );
	}

	int nWords = Functional::wordNum( varSet.size() );
	int nFunctions = 1000; // FIXME
	BitVectorPool functionPool( nWords, nFunctions );
	Functional::createLiterals( varSet, literals, functionPool );

	for ( auto leave : leaves ) {
		aig::NodeData & leaveData = graph.getData( leave, galois::MethodFlag::READ );
		auto it = literals.find( ("i"+std::to_string( leaveData.id )) );
		computedFunctions.insert( std::make_pair( leaveData.id, it->second.first ) );
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

word * buildFunction( aig::Graph & graph, GNodeSet & leaves, aig::GNode & currentNode, std::unordered_map< int, Functional::word* > & computedFunctions, BitVectorPool & functionPool, int nWords ) {

	aig::NodeData & currentNodeData = graph.getData( currentNode, galois::MethodFlag::READ );
	auto it = computedFunctions.find( currentNodeData.id );
	
	if ( it != computedFunctions.end() ) {
		return it->second;
	}

	if ( currentNodeData.type == aig::NodeType::AND ) {
	
		auto edgeIt = graph.in_edges( currentNode ).begin();
		aig::GNode lhsNode = graph.getEdgeDst( *edgeIt );
		bool lhsEdge = graph.getEdgeData( *edgeIt, galois::MethodFlag::WRITE );
		edgeIt++;
		aig::GNode rhsNode = graph.getEdgeDst( *edgeIt );
		bool rhsEdge = graph.getEdgeData( *edgeIt, galois::MethodFlag::WRITE );

		
		word * lhsFunction = buildFunction( graph, leaves, lhsNode, computedFunctions, functionPool, nWords );
		word * rhsFunction = buildFunction( graph, leaves, rhsNode, computedFunctions, functionPool, nWords );

		if ( lhsEdge == false ) {
			word * aux = functionPool.getMemory();
			Functional::NOT( aux, lhsFunction, nWords );
			lhsFunction = aux;
		}

		if ( rhsEdge == false ) {
			word * aux = functionPool.getMemory();
			Functional::NOT( aux, rhsFunction, nWords );
			rhsFunction = aux;
		}

		word * result = functionPool.getMemory();
		Functional::AND( result, lhsFunction, rhsFunction, nWords );
		computedFunctions.insert( std::make_pair( currentNodeData.id, result ) );

		return computedFunctions[ currentNodeData.id ];
	}
	else {
		if ( currentNodeData.type == aig::NodeType::PO ) {

			auto edgeIt = graph.in_edges( currentNode ).begin();
			aig::GNode inNode = graph.getEdgeDst( *edgeIt );
			bool inEdge = graph.getEdgeData( *edgeIt, galois::MethodFlag::WRITE );

			word * inFunction = buildFunction( graph, leaves, inNode, computedFunctions, functionPool, nWords );
	
			if ( inEdge == false ) {
				word * aux = functionPool.getMemory();
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

void dumpLogicFuntions( std::string hexaText, std::string label ) {

	std::stringstream path;
	path << "/workspace/vnpossani/GALOIS_DUMPS/functions/" << label << "-inputs.txt";
	std::ofstream functionFile;
	functionFile.open( path.str(), std::ios_base::app );
	functionFile << hexaText;
	functionFile.close();
}

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


} /* namespace algorithm */



