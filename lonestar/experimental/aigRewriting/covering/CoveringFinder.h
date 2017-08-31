/*
 * CoveringFinder.h
 *
 *  Created on: 31/03/2015
 *      Author: possani
 */

#ifndef COVERINGFINDER_H_
#define COVERINGFINDER_H_

#include "../functional/BitVectorPool.h"
#include "../subjectgraph/cmxaig/Cmxaig.h"
#include "../subjectgraph/cmxaig/nodes/Node.h"
#include "../covering/CoveringHandler.h"

#include <vector>
#include <unordered_set>

namespace Covering {

typedef SubjectGraph::Cmxaig Cmxaig;
typedef SubjectGraph::Node Node;

class CoveringFinder {

	Cmxaig & cmxaig;
	BitVectorPool & coveringPool;
	CoveringMananger & covMan;

	std::unordered_set< int > alreadVisited;

	bool computeCoverings( Node* currentNode, CoveringVector_1D & currentVector, CoveringVector_2D & alreadyComputed );

	int getTypeShift( Node * node );

	//CoveringVector_1D * computeMatrices( Node* currentNode, int maxNumMatrices );

public:

	CoveringFinder( Cmxaig & cmxaig, BitVectorPool & coveringPool, CoveringMananger & covMan );

	virtual ~CoveringFinder();

	bool run( CoveringVector_2D & outputCoverings );

	//CoveringVector_2D* run( int maxNumMatrices );
};

}  // End namespace Covering

#endif /* COVERINGFINDER_H_ */
