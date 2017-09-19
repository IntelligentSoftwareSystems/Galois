#ifndef WINDOWING_H_
#define WINDOWING_H_

#include "Aig.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"

#include <iostream>
#include <unordered_set>
#include <deque>
#include <limits>
#include <sstream>
#include <string>
#include <fstream>

namespace algorithm {

typedef Galois::PerIterAllocTy Alloc;
typedef std::unordered_set< aig::GNode, std::hash<aig::GNode>, std::equal_to<aig::GNode>, Galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeSet;
typedef std::deque< aig::GNode, Galois::PerIterAllocTy::rebind< aig::GNode >::other > GNodeDeque;

class Windowing {

private:

	aig::Graph & graph;
	Galois::PerIterAllocTy & allocator;

	GNodeSet window;
	GNodeSet leaves;
	GNodeSet roots;
	GNodeSet dagNodes;

	int nInputs;
	int nOutputs;
	int nLevels;
	int cutSizeLimit;

	void collectNodesTFO( GNodeSet & sources );
	void collectNodesTFOrec( aig::GNode & node, int levelCounter );
	void collectNodesTFO_iter( GNodeSet & sources );
	void collectRoots();

	void reconvDrivenCut( GNodeSet & sources );
	void constructCut( GNodeSet & currentLeaves, GNodeSet & currentWindow );
	void reconvDrivenCut_iter( GNodeSet & sources );
	int leafCost( aig::GNode & node, GNodeSet & currentWindow );

	void checkRoots();
	void checkLeaves();
	void checkWindow();

	void unification( GNodeSet & S1, GNodeSet & S2 );

public:

	Windowing( aig::Graph & graph, Galois::PerIterAllocTy & allocator, int nInputs, int nOutputs, int nLevels, int cutSizeLimit );

	virtual ~Windowing();

	bool computeReconvDrivenWindow( aig::GNode & node );
	bool computeWindowTEST( aig::GNode & node );

	GNodeSet & getWindow();
	GNodeSet & getRoots();
	GNodeSet & getLeaves();
	GNodeSet & getDagNodes();

	void printSet( aig::Graph & graph, GNodeSet & S, std::string label );
	std::string toDot();
	void writeWindowDot( std::string dotText, std::string label );

};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* WINDOWING_H_ */
