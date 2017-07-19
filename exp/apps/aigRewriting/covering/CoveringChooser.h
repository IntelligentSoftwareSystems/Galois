/*
 * CoveringChooser.h
 *
 *  Created on: 31/03/2015
 *      Author: possani
 */

#ifndef COVERINGCHOOSER_H_
#define COVERINGCHOOSER_H_

#include "../subjectgraph/cmxaig/Cmxaig.h"
#include "../covering/CoveringHandler.h"

#include <vector>
#include <utility>
#include <vector>
#include <unordered_set>

namespace Covering {

typedef SubjectGraph::Cmxaig Cmxaig;
typedef SubjectGraph::Node Node;
typedef std::pair< float, word* > Cost;
typedef std::vector< Cost > CostVector_1D;
typedef std::vector< CostVector_1D > CosVector_2D;

class CoveringChooser {

	Cmxaig & cmxaig;
	BitVectorPool & coveringPool;
	CoveringMananger & covMan;
	CoveringVector_2D & outputCoverings;
	CoveringVector_1D graphCoverings;

	word * finalCovering;
	float finalCost;
	unsigned finalNodeCount;
	unsigned andCount;
	unsigned xorCount;
	unsigned muxCount;

	float computeCost( word * covering, float alpha, float beta, float gamma) ;

	int computeUse( word * covering, int begin, int end );

	int popcount ( unsigned long x );

	bool cleanCmxaig ( Node * currentNode, std::unordered_set< unsigned > & visited );

	int getTypeShift( Node * node );

	std::string makeReport( unsigned m, unsigned x, unsigned a, float alpha, float beta, float gamma );

public:

	CoveringChooser( Cmxaig & cmxaig, BitVectorPool & coveringPool, CoveringMananger & covMan, CoveringVector_2D & outputCoverings );
	
	virtual ~CoveringChooser();

	std::string run( float alpha, float beta, float gamma );

	std::string run( float alpha, float beta, float gamma, float maxNumSoutions );

	Cmxaig & getCoveredCmxaig();

	void showAllCoverings();

	/******** Getters and Setters ********/

	CoveringVector_1D & getGraphCovering();

	word * getFinalCovering();

	float getFinalCost();

	unsigned getFinalNodeCount();

	unsigned getFinalAndCount();

	unsigned getFinalXorCount();

	unsigned getFinalMuxCount();

};

} // End namespace Covering

#endif /* COVERINGCHOOSER_H_ */
