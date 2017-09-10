/*
 * CoveringChooser.cpp
 *
 *  Created on: 31/03/2015
 *      Author: possani
 */

#include "../covering/CoveringChooser.h"
#include "../subjectgraph/cmxaig/nodes/Node.h"

#include <iostream>
#include <sstream>
#include <set>
#include <algorithm>

namespace Covering {

CoveringChooser::CoveringChooser( Cmxaig & cmxaig, BitVectorPool & coveringPool, CoveringMananger & covMan, CoveringVector_2D & outputCoverings ) :
								  cmxaig( cmxaig ), coveringPool( coveringPool ), covMan( covMan ), outputCoverings( outputCoverings ) {

	this->finalCovering = nullptr;
	this->finalCost = std::numeric_limits<float>::max();
	this->finalNodeCount = 0;
	this->andCount = 0;
	this->xorCount = 0;
	this->muxCount = 0;

	auto it = outputCoverings.begin(); 
	while ( it != outputCoverings.end() ) {
		if ( it->empty() ) {
			outputCoverings.erase( it );
		}
		else {
			it++;
		}
	}
}

CoveringChooser::~CoveringChooser() {
}

/* EXACT COVERING: Combine all possible solutions of each output node */
std::string CoveringChooser::run( float alpha, float beta, float gamma ) {

	float currentCost;

	if ( this->graphCoverings.empty() ) {
		//std::cout << "Combining Output Coverings ..." << std::endl;
		Covering::combineCoverigns( this->graphCoverings, this->outputCoverings, this->coveringPool, covMan.nWords );
	}

	for ( auto cov : graphCoverings ) {
		currentCost = computeCost( cov, alpha, beta, gamma );
		if ( this->finalCost > currentCost ) {
			this->finalCost = currentCost;
			this->finalCovering = cov;
		}
	}

	if ( this->finalCovering == nullptr ) {
		//std::cout << "Covering not found!" << std::endl; //FIXME Insert an Exception
	}

	this->andCount = computeUse( this->finalCovering, this->covMan.andBegin, this->covMan.andEnd );
	this->xorCount = computeUse( this->finalCovering, this->covMan.xorBegin, this->covMan.xorEnd );
	this->muxCount = computeUse( this->finalCovering, this->covMan.muxBegin, this->covMan.muxEnd );

	this->finalNodeCount = this->muxCount + this->xorCount + this->andCount;

	return makeReport( this->muxCount, this->xorCount, this->andCount, alpha, beta, gamma );
}

/* This function is used to sort a vector of pair<unsigned, Matrix*>, according to the cost (first element) */
bool pairComparator( const std::pair< float, word* > & lhs, const std::pair< float, word* >& rhs ) {
	return lhs.first < rhs.first;
}

/*HEURISTC COVERING: Combine only the nSolutions (with smallest cost) of each output node */
std::string CoveringChooser::run( float alpha, float beta, float gamma, float nSoutions ) {

	CosVector_2D outputCosts;
	float currentCost;

	/* For each output, computes the pairs <cost, matrix> of each possible solution*/
	for ( auto covVec : this->outputCoverings ) {

		CostVector_1D currentCostVector;

		for ( auto cov : covVec ) {

			currentCost = computeCost( cov, alpha, beta, gamma );

			currentCostVector.push_back( std::make_pair( currentCost, cov ) );
		}

		std::sort( currentCostVector.begin(), currentCostVector.end(), pairComparator ); // Sort according to the cost

		outputCosts.push_back( currentCostVector );
	}

	this->outputCoverings.clear();
	unsigned counter;

	/* For each output, selects the fist "nSolutions" (solutions with smallest costs) */
	for ( auto costVec : outputCosts ) {

		CoveringVector_1D currentCoveringVector;
		counter = 0;

		for ( auto costPair : costVec ) {
			currentCoveringVector.push_back( costPair.second );
			counter++;
			if ( counter >= nSoutions ) {
				break;
			}
		}

		this->outputCoverings.push_back( currentCoveringVector );
	}

	/*
	std::cout << "nOutputs: " << this->outputCoverings.size() << std::endl;
	int size = 0;
	for ( auto vec : this->outputCoverings ) {
		std::cout << vec.size() << " * ";
		size += vec.size();
	}
	std::cout << " = " << size << std::endl;
	*/

	//std::cout << "Combining Output Coverings ..." << std::endl;
	Covering::combineCoverigns( this->graphCoverings, this->outputCoverings, this->coveringPool, this->covMan.nWords );

	for ( auto cov : this->graphCoverings ) {

		currentCost = computeCost( cov, alpha, beta, gamma );

		if ( this->finalCost > currentCost ) {
			this->finalCost = currentCost;
			this->finalCovering = cov;
		}
	}

	if ( this->finalCovering == nullptr ) {
		//std::cout << "Covering not found!" << std::endl; //FIXME Insert an Exception
	}

	this->andCount = computeUse( this->finalCovering, this->covMan.andBegin, this->covMan.andEnd );
	this->xorCount = computeUse( this->finalCovering, this->covMan.xorBegin, this->covMan.xorEnd );
	this->muxCount = computeUse( this->finalCovering, this->covMan.muxBegin, this->covMan.muxEnd );

	this->finalNodeCount = this->muxCount + this->xorCount + this->andCount;

	return makeReport( this->muxCount, this->xorCount, this->andCount, alpha, beta, gamma );
}

float CoveringChooser::computeCost( word * covering, float alpha, float beta, float gamma) {

	unsigned currentAndCount = computeUse(covering, this->covMan.andBegin, this->covMan.andEnd );
	unsigned currentXorCount = computeUse( covering, this->covMan.xorBegin, this->covMan.xorEnd );
	unsigned currentMuxCout = computeUse( covering, this->covMan.muxBegin, this->covMan.muxEnd );

	return alpha * currentAndCount + beta * currentXorCount + gamma * currentMuxCout;
}

int CoveringChooser::computeUse( word * covering, int begin, int end ) {

	int count = 0;

	for ( int i = begin; i < end; i++ ) {
		count += popcount( covering[i] );
	}

	return count;
}

//This is better when most bits in x are 0. It uses 3 arithmetic operations and one comparison/branch per "1" bit in x.
int CoveringChooser::popcount(unsigned long x) {

	int count;

	for (count=0; x; count++) {
		x &= x-1;
	}

	return count;
}

Cmxaig & CoveringChooser::getCoveredCmxaig() {

	std::unordered_set <unsigned > visited;

	//std::cout << "Cleanning CMXAIG ..." << std::endl;

	for ( auto output : this->cmxaig.getOutputs() ) {
		cleanCmxaig( output, visited );
	}

	for ( auto input : this->cmxaig.getInputs() ) {
		if ( input->getOutNodes().empty()) {
			this->cmxaig.removeNode( input );
		}
	}

	return this->cmxaig;
}

bool CoveringChooser::cleanCmxaig( Node * currentNode, std::unordered_set< unsigned> & visited ) {

	if( currentNode->isInputNode() || (visited.find( currentNode->getId() ) != visited.end()) ) {
		return false;
	}

	visited.insert( currentNode->getId() );

	std::vector<bool> enableRemoval;
	std::vector<Node*> nodes;

	for ( auto in : currentNode->getInNodes() ) {
		enableRemoval.push_back( cleanCmxaig( in, visited ) );
		nodes.push_back( in );
	}

	for ( int i = 0; i < nodes.size() ; i++ ) {
		if ( enableRemoval[i] ) {
			this->cmxaig.removeNode( nodes[i] );
		}
	}

	if ( currentNode->isChoiceNode() && currentNode->getInNodes().empty() ) {
		return true;
	}

	int typeShift = getTypeShift( currentNode );
	if ( (currentNode->isMuxNode() || currentNode->isXorNode() || currentNode->isAndNode()) && ( Covering::contaisNode( this->finalCovering, currentNode->getTypeId(), typeShift ) == false) ) {
		return true;
	}

	return false;
}

int CoveringChooser::getTypeShift( Node * node ) {

	if ( node->isAndNode() ) {
		return this->covMan.andBegin;
	}

	if ( node->isXorNode() ) {
		return this->covMan.xorBegin;
	}

	if ( node->isMuxNode() ) {
		return this->covMan.muxBegin;
	}

	return 0;
}

void CoveringChooser::showAllCoverings() {

	std::cout << "\n *********************************** CIRCUIT COVERINGS ********************************** \n";
	for ( auto cov : this->graphCoverings ) {
		std::cout << Covering::toString( cov, this->covMan ) << std::endl << std::endl;
	}
	std::cout << "\n **************************************************************************************** \n";
}

/* Getters and Setters */

CoveringVector_1D & CoveringChooser::getGraphCovering() {
	return this->graphCoverings;
}

float CoveringChooser::getFinalCost() {
	return this->finalCost;
}

word * CoveringChooser::getFinalCovering() {
	return this->finalCovering;
}

unsigned CoveringChooser::getFinalNodeCount() {
	return this->finalNodeCount;
}

unsigned CoveringChooser::getFinalAndCount() {
	return this->andCount;
}

unsigned CoveringChooser::getFinalXorCount() {
	return this->xorCount;
}

unsigned CoveringChooser::getFinalMuxCount() {
	return this->muxCount;
}

std::string CoveringChooser::makeReport( unsigned muxCount, unsigned xorCount, unsigned andCount, float alpha, float beta, float gamma ) {

	std::stringstream report;
	report << "Number of Possible Solutions: " << this->graphCoverings.size() << " matrices\n";
	report << "# AND: " << andCount << "\tAlpha: " << alpha << "\n";
	report << "# XOR: " << xorCount << "\tBeta: " << beta << "\n";
	report << "# MUX: " << muxCount << "\tGamma: " << gamma << "\n";
	report << "Final Cost: " << this->getFinalCost() << "\n";
	report << "Final Node Count: " << this->getFinalNodeCount() << "\n";
	report << "Final Covering:\n" << Covering::toString( this->finalCovering, this->covMan ) << "\n";

	return report.str();
}

} // End namespace Covering
