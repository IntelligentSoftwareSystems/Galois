/*

 Possani K-Cuts August 29, 2017.
 ABC-based implementation on Galois.

*/

#ifndef CUTMANANGER_H_
#define CUTMANANGER_H_

#include "Aig.h"
#include "CutPool.h"
#include "../functional/FunctionHandler32.h"
#include "galois/Reduction.h"

namespace algorithm {

typedef struct cutList_ {
        
	Cut ** head;
	Cut ** tail;
   
	cutList_( int K ) {
		head = new Cut*[ K+1 ];
		for ( int i = 0; i < K+1; i++ ) {
			head[i] = nullptr;
		}
	  
		tail = new Cut*[ K+1 ];
		for ( int i = 0; i < K+1; i++ ) {
			tail[i] = nullptr;
		}
	}

	~cutList_() {
		delete [] head;
		delete [] tail;
	}
                                                                                   
} CutList;


typedef struct auxTruth_ {

	unsigned int * truth[4];

	auxTruth_( int nWords ) {
		for ( int i = 0; i < 4; i++ ) {
			truth[i] = new unsigned int[ nWords ];
		}
	}

	~auxTruth_() {
		for ( int i = 0; i < 4; i++ ) {
			delete [] truth[i];
		}
	}
	
} AuxTruth;

typedef galois::substrate::PerThreadStorage< CutPool > PerThreadCutPool;
typedef galois::substrate::PerThreadStorage< CutList > PerThreadCutList;
typedef galois::substrate::PerThreadStorage< AuxTruth > PerThreadAuxTruth;

class CutMananger {

private:

	aig::Aig & aig;
	int K;
	int C;
	int nWords;
	int nNodes;	
	int nThreads;
	long int cutPoolSize;
	bool compTruth;
	long double kcutTime;

	PerThreadCutPool perThreadCutPool;
	PerThreadCutList perThreadCutList;
	PerThreadAuxTruth perThreadAuxTruth;
	Cut ** nodeCuts;

	// Cuts Statistics //
	galois::GAccumulator< long int > nCuts;
	galois::GAccumulator< long int > nTriv;
	galois::GAccumulator< long int > nFilt;
	galois::GAccumulator< long int > nSatu;

	// Runtime Statistics //
	galois::GAccumulator< long int > mergeTime;
	galois::GAccumulator< long int > filterTime;
	galois::GAccumulator< long int > procTwoTime;
	galois::GAccumulator< long int > compTime;
	galois::GAccumulator< long int > scheduleTime;


	void computeCutsRec( aig::GNode node, CutPool * cutPool, CutList * cutList, AuxTruth * auxTruth );

	inline bool processTwoCuts( CutPool * cutPool, CutList * cutList, AuxTruth * auxTruth, Cut * lhsCut, Cut * rhsCut, bool lhsPolarity, bool rhsPolarity, int nodeId, int & currentNumCuts );

	Cut * mergeCuts( CutPool * cutPool, Cut * lhsCut, Cut * rhsCut );

	inline bool cutFilter( CutPool * cutPool, CutList * cutList, Cut * resCut, int & currentNumCuts );

	inline bool checkCutDominance( Cut * smallerCut, Cut * largerCut );

	inline void commitCuts( int nodeId, CutList * cutList );

	void computeTruth( AuxTruth * auxTruth, Cut * resCut, Cut * lhsCut, Cut * rhsCut, bool lhsPolarity, bool rhsPolarity );

	inline unsigned truthPhase( Cut * resCut, Cut * inCut );

public:

	CutMananger( aig::Aig & aig, int K, int C, int nThreads, bool compTruth );

	~CutMananger();

	void computeCuts( CutPool * cutPool, CutList * cutList, AuxTruth * auxTruth, int nodeId, int lhsId, int rhsId, bool lhsPolarity, bool rhsPolarity );

	void computeCutsRecursively( aig::GNode node );

	unsigned int * readTruth( Cut * cut );
	void recycleNodeCuts( int nodeId );
	void printNodeCuts( int nodeId, long int & counter );
	void printAllCuts();
	void printCutStatistics();
	void printRuntimes();

	aig::Aig & getAig();
	int getK();
	int getC();
	int getNWords();
	int getNThreads();
	bool getCompTruthFlag();
	long double getKcutTime();
	void setKcutTime( long double time );
	PerThreadCutPool & getPerThreadCutPool();
	PerThreadCutList & getPerThreadCutList();
	PerThreadAuxTruth & getPerThreadAuxTruth();
	Cut ** getNodeCuts();

};

// Function that runs the KCut operator define in the end of file CutMananger.cpp //
void runKCutOperator( CutMananger & cutMan );

} /* namespace algorithm */

#endif /* CUTMANANGERC_H_ */
