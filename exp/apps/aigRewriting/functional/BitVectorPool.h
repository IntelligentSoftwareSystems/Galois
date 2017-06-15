/*
 * FunctionPool.h
 *
 *  Created on: 27/03/2017
 *      Author: viniciuspossani
 */

#ifndef SRC_MAIN_FUNCTIONPOOL_H_
#define SRC_MAIN_FUNCTIONPOOL_H_

#include <vector>

namespace Functional {

typedef unsigned long word;

class BitVectorPool {

	int blockSize;
	int nWords;
	int index;
	word ** currentBlock;
	std::vector< word** > blocks;

	void alloc();

public:

	BitVectorPool( int nElements, int nWords );

	virtual ~BitVectorPool();

	word * getMemory();

	word * getCleanMemory();

	void giveBackMemory();

	int getNumBlocks();
};

} /* namespace Functional */

#endif /* SRC_MAIN_FUNCTIONPOOL_H_ */
