/*
 * CoveringFinder.cpp
 *
 *  Created on: 31/03/2015
 *      Author: possani
 */

#include "../covering/CoveringFinder.h"
#include <iostream>

namespace Covering {

CoveringFinder::CoveringFinder( Cmxaig & cmxaig, BitVectorPool & coveringPool, CoveringMananger & covMan ) : cmxaig( cmxaig ), coveringPool( coveringPool ), covMan( covMan ) {

}

CoveringFinder::~CoveringFinder() {
}

/* EXACT COVERING */
bool CoveringFinder::run( CoveringVector_2D & outputCoverings ) {

	CoveringVector_2D alreadyComputed( this->cmxaig.getChoiceCounter() );
	bool abandon;
	for ( auto output : this->cmxaig.getOutputs() ) {
		CoveringVector_1D coveringVector;
		abandon = computeCoverings( output->getInNodes()[0], coveringVector, alreadyComputed );

		if ( abandon ) {
			return true;
		}

		outputCoverings.push_back( coveringVector );
	}

	return false;
}

/* EXACT COVERING */
bool CoveringFinder::computeCoverings( Node* currentNode, CoveringVector_1D & currentVector, CoveringVector_2D & alreadyComputed ) {

	if ( ( this->alreadVisited.count( currentNode->getId() ) > 0 ) && ( ( currentNode->isChoiceNode() ) && ( alreadyComputed[ currentNode->getTypeId() ].empty() ) ) ) {
		std::cout << "Cycle" << std::endl;
		return true;
	}

	this->alreadVisited.insert( currentNode->getId() );

	CoveringVector_2D currentCoverings;
	word * covering;

	if ( ( currentNode->isChoiceNode() ) && ( alreadyComputed[ currentNode->getTypeId() ].empty() == false ) ) {
		int choiceID = currentNode->getTypeId();
		currentVector.insert( currentVector.begin(),  alreadyComputed[ choiceID ].begin(), alreadyComputed[ choiceID ].end() );
		return false;
	}

	if ( currentNode->isInputNode() ) {
		return false;
	}

	bool abandon;
	for ( auto inNode : currentNode->getInNodes() ) {
		CoveringVector_1D nextVector;
		abandon = computeCoverings( inNode, nextVector, alreadyComputed );

		if ( abandon ) {
			return true;
		}

		if ( !nextVector.empty() ) {
			currentCoverings.push_back( nextVector );
		}
	}

	if ( currentNode->isMuxNode() || currentNode->isXorNode() || currentNode->isAndNode() ) {
		int typeShift = this->getTypeShift( currentNode );
		int constraint = 2;

		if ( currentNode->isMuxNode() ) {
			constraint = 3;
		}

		if ( currentCoverings.size() < constraint ) {
			covering = this->coveringPool.getCleanMemory();
			Covering::registerNode( covering, currentNode->getTypeId(), typeShift );
			currentVector.push_back( covering );
			currentCoverings.push_back( currentVector );
		}

		if ( currentCoverings.size() >= constraint ) {
			currentVector.clear();
			Covering::combineCoverigns( currentVector, currentCoverings, this->coveringPool, this->covMan.nWords );
			for ( auto cov : currentVector ) {
				Covering::registerNode( cov, currentNode->getTypeId(), typeShift );
			}
		}
	}
	else {
		if ( currentNode->isChoiceNode() ) {
			for ( auto vec : currentCoverings ) {
				currentVector.insert( currentVector.end(), vec.begin(), vec.end() );
			}
			alreadyComputed[ currentNode->getTypeId() ] = currentVector;
		}
	}

	return false;
}

///* HEURISTC COVERING */
//CoveringVector_2D* CoveringFinder::run(int maxNumMatrices) {
//
//	CoveringVector_1D* matrixVector;
//
//	for(unsigned i=0; i < this->cmxaig->getOutputs().size(); i++) {
//		matrixVector = computeMatrices(this->cmxaig->getOutputs()[i], maxNumMatrices);
//		this->outputCoverings->push_back(matrixVector);
//	}
//
//	return this->outputCoverings;
//}
//
///*  HEURISTC COVERING */
//CoveringVector_1D* CoveringFinder::computeMatrices(Node* currentNode, int maxNumMatrices) {
//
//	CoveringVector_2D* currentCoverings = new CoveringVector_2D();
//	CoveringVector_1D* matrixVector = new CoveringVector_1D();
//	Matrix* matrix;
//	unsigned numSolutions;
//	unsigned counter;
//
//	if(currentNode->isInputNode()) {
//		return matrixVector;
//	}
//
//	for(unsigned i=0; i < currentNode->getInNodes().size(); i++) {
//
//		matrixVector = computeMatrices(currentNode->getInNodes()[i], maxNumMatrices);
//
//		if(!matrixVector->empty()) {
//			currentCoverings->push_back(matrixVector);
//		}
//	}
//
//	if(currentCoverings->empty()) {
//		matrix = new Matrix(this->matrixLength);
//		matrix->registerNode(currentNode);
//		matrixVector->push_back(matrix);
//	}
//	else {
//		if( currentNode->isMuxNode() || currentNode->isXorNode() || currentNode->isAndNode() ) {
//
//			matrixVector = CombinationGenerator::combineCoverigns(currentCoverings);
//
//			for(unsigned i=0; i < matrixVector->size(); i++) {
//				matrix = (*matrixVector)[i];
//				matrix->registerNode(currentNode);
//			}
//		}
//		else {
//			if(currentNode->isChoiceNode()) {
//
//				matrixVector = new CoveringVector_1D();
//
//				if(currentCoverings->size() < maxNumMatrices) {
//					numSolutions = maxNumMatrices/currentCoverings->size();
//					for(CoveringVector_2D::iterator vectorIt = currentCoverings->begin(); vectorIt != currentCoverings->end(); vectorIt++) {
//						counter = 0;
//						for(CoveringVector_1D::iterator matricesIt = (*vectorIt)->begin(); matricesIt != (*vectorIt)->end(); matricesIt++) {
//							if(counter <= numSolutions) {
//								matrixVector->push_back( (*matricesIt) );
//								counter++;
//							}
//							else {
//								break;
//							}
//						}
//					}
//				}
//				else {
//					for(CoveringVector_2D::iterator vectorIt = currentCoverings->begin(); vectorIt != currentCoverings->end(); vectorIt++) {
//						if(matrixVector->size() <= maxNumMatrices) {
//							matrixVector->push_back( (*(*vectorIt)->begin()) );
//						}
//						else {
//							break;
//						}
//					}
//				}
//			}
//		}
//	}
//
//	return matrixVector;
//}

int CoveringFinder::getTypeShift( Node * node ) {

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


} // End namespace Covering
