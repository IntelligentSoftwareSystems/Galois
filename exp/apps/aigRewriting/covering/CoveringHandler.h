/*
 * CoveringHandler.h
 *
 *  Created on: 24/05/2017
 *      Author: Vincius Possani
 */

#ifndef COVERING_HANDLER_H_
#define COVERING_HANDLER_H_

#include "../functional/BitVectorPool.h"

#include <vector>
#include <string>
#include <iostream>

namespace Covering {

typedef unsigned long int word;
typedef std::vector< word* > CoveringVector_1D;
typedef std::vector< CoveringVector_1D > CoveringVector_2D;
typedef Functional::BitVectorPool BitVectorPool;

typedef struct coveringMananger {

	int andBegin;
	int andEnd;
	int xorBegin;
	int xorEnd;
	int muxBegin;
	int muxEnd;
	int nWords;

} CoveringMananger;

static void setCoveringMananger( int nAND, int nXOR, int nMUX, CoveringMananger & covMan );
static bool equals( word * lhs, word * rhs, int nWords );
static void copy( word * result, word * original, int nWords );
static void AND( word * result, word * lhs, word * rhs, int nWords );
static void OR( word * result, word * lhs, word * rhs, int nWords );
static void XOR( word * result, word * lhs, word * rhs, int nWords );
static void registerNode( word * covering, int typeID, int typeShift );
static bool contaisNode( word * covering, int typeID, int typeShift );
static void combineCoverigns( CoveringVector_1D & resultantVector, CoveringVector_2D & coverigns, BitVectorPool & coveringPool, int nWords );
static void combineCoveringVectors( CoveringVector_1D & currentResultVector, CoveringVector_1D & firstVector, CoveringVector_1D & secondVector, BitVectorPool & coveringPool, int nWords );
static std::string toString( word * covering, CoveringMananger & covMan );
static std::string decimalToBinaryString( word * covering, int begin, int end );

void setCoveringMananger( int nAND, int nXOR, int nMUX, CoveringMananger & covMan ) {

	covMan.nWords = 0;

	// ######## AND ######### //
	covMan.andBegin = 0;

	if ( nAND > 64 ) {
		covMan.nWords = (nAND / 64);
		if ( (nAND % 64) != 0 ) {
			covMan.nWords += 1;
		}
		covMan.andEnd = covMan.nWords;
	}
	else {
		if ( nAND != 0 ) {
			covMan.nWords = 1;
			covMan.andEnd = covMan.nWords;
		}
		else {
			covMan.andEnd = 0;
		}
	}

	// ######## XOR ######### //
	covMan.xorBegin = covMan.nWords;

	if ( nXOR > 64 ) {
		covMan.nWords += (nXOR / 64);
		if ( (nXOR % 64) != 0 ) {
			covMan.nWords += 1;
		}
		covMan.xorEnd = covMan.nWords;
	}
	else {
		if ( nXOR != 0 ) {
			covMan.nWords += 1;
			covMan.xorEnd = covMan.nWords;
		}
		else {
			covMan.xorBegin = 0;
			covMan.xorEnd = 0;
		}
	}

	// ######## MUX ######### //
	covMan.muxBegin = covMan.nWords;

	if ( nMUX > 64 ) {
		covMan.nWords += (nMUX / 64);
		if ( (nMUX % 64) != 0 ) {
			covMan.nWords += 1;
		}
		covMan.muxEnd = covMan.nWords;
	}
	else {
		if ( nMUX != 0 ) {
			covMan.nWords += 1;
			covMan.muxEnd = covMan.nWords;
		}
		else {
			covMan.muxBegin = 0;
			covMan.muxEnd = 0;
		}
	}
}

bool equals( word * lhs, word * rhs, int nWords ) {

	if ( ( lhs == nullptr ) || ( rhs == nullptr ) ) {
		return false;
	}

	for ( int i = 0; i < nWords; i++ ) {
		if ( lhs[i] != rhs[i] ) {
			return false;
		}
	}

	return true;
}

void copy( word * result, word * original, int nWords ) {

	if ( ( result == nullptr ) || ( original == nullptr ) ) {
		std::cout << "CoveringHandler::copy : At least one operand is NULL." << std::endl;
		exit(1);
	}

	for ( int i = 0; i < nWords; i++ ) {
		result[i] = original[i];
	}
}

void AND( word * result, word * lhs, word * rhs, int nWords ) {

	if ( ( lhs == nullptr ) || ( rhs == nullptr ) ) {
		std::cout << "CoveringHandler::AND: At least one operand is NULL." << std::endl;
		exit(1);
	}

	for ( int i = 0; i < nWords; i++ ) {
		result[i] = lhs[i] & rhs[i];
	}
}

void OR( word * result, word * lhs, word * rhs, int nWords ) {

	if ( ( lhs == nullptr ) || ( rhs == nullptr ) ) {
		std::cout << "CoveringHandler::OR: At least one operand is NULL." << std::endl;
		exit(1);
	}

	for ( int i = 0; i < nWords; i++ ) {
		result[i] = lhs[i] | rhs[i];
	}
}

void XOR( word * result, word * lhs, word * rhs, int nWords ) {

	if ( ( lhs == nullptr ) || ( rhs == nullptr ) ) {
		std::cout << "CoveringHandler::XOR: At least one operand is NULL." << std::endl;
		exit(1);
	}

	for ( int i = 0; i < nWords; i++ ) {
		result[i] = lhs[i] ^ rhs[i];
	}
}

void registerNode( word * covering, int typeID, int typeShift ) {

	if ( covering == nullptr ) {
		std::cout << "CoveringHandler::registerNode: covering is NULL." << std::endl;
		exit(1);
	}

	int index, bit;
	word mask = 1;
	index = typeShift + (typeID / 64);
	bit = typeID % 64;
	mask = mask << bit;
	covering[ index ] = covering[ index ] | mask;
}

bool contaisNode( word * covering, int typeID, int typeShift ) {

	if ( covering == nullptr ) {
		std::cout << "CoveringHandler::containsNode: covering is NULL." << std::endl;
		exit(1);
	}

	int index, bit;
	word mask = 1;
	index = typeShift + (typeID / 64);
	bit = typeID % 64;
	mask = mask << bit;
	return ( (covering[ index ] & mask) != 0 );
}

void combineCoverigns( CoveringVector_1D & resultantVector, CoveringVector_2D & coverings, BitVectorPool & coveringPool, int nWords ) {

	//CoveringVector_1D hardCopyFirstVector = coverigns.back();
	//CoveringVector_1D & firstVector = hardCopyFirstVector;
	CoveringVector_1D firstVector = coverings.back();
	coverings.pop_back();

	while ( !coverings.empty() ) {
		CoveringVector_1D & secondVector = coverings.back();
		CoveringVector_1D currentResultVector;
		combineCoveringVectors( currentResultVector, firstVector, secondVector, coveringPool, nWords );
		firstVector = currentResultVector;
		coverings.pop_back();
	}

	resultantVector = firstVector;

	//for ( auto element : firstVector ) {
	//	resultantVector.push_back( element );
	//}
}

void combineCoveringVectors( CoveringVector_1D & currentResultVector, CoveringVector_1D & firstVector, CoveringVector_1D & secondVector, BitVectorPool & coveringPool, int nWords ) {

	for ( int i = 0; i < firstVector.size(); i++ ) {
		for ( int j = 0; j < secondVector.size(); j++ ) {
			word * resultantCovering = coveringPool.getMemory();
			Covering::OR( resultantCovering, firstVector[i], secondVector[j], nWords );
			currentResultVector.push_back( resultantCovering );
		}
	}
}

std::string toString( word * covering, CoveringMananger & covMan ) {

	std::string binaryString = "";

	binaryString += "And: " + decimalToBinaryString( covering, covMan.andBegin, covMan.andEnd ) + "\n";
	binaryString += "Xor: " + decimalToBinaryString( covering, covMan.xorBegin, covMan.xorEnd ) + "\n";
	binaryString += "Mux: " + decimalToBinaryString( covering, covMan.muxBegin, covMan.muxEnd ) + "\n";

	return binaryString;
}

std::string decimalToBinaryString( word * covering, int begin, int end )  {

	std::string binaryString = "";

	for ( int i = end-1; i >= begin; i-- ) {
		for(int j=63; j >= 0; j--) {
			if ( (covering[i] >> j) & 1) {
				binaryString += ("1");
			}
			else {
				binaryString += ("0");
			}
		}
	}

	return binaryString;
}

} // End namespace Covering

#endif /* COVERING_HANDLER_H_ */
