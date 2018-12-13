/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/*

 @Vinicius Possani
 Parallel Parallel LUT-Based Tech Mapping October 16, 2018.
 ABC-based implementation on Galois.

*/

#ifndef PRIORITYCUTMANAGER_H_
#define PRIORITYCUTMANAGER_H_

#include "Aig.h"
#include "PriorityCutPool.h"
#include "CutManager.h"
#include "../functional/FunctionHandler32.h"
#include "galois/Reduction.h"
#include <unordered_map>

namespace algorithm {

typedef struct pricutList_ {

  PriCut ** array;
	int 			nCuts;

	pricutList_(int C) {
		array = new PriCut*[C+1];
		nCuts	= 0;
	}

	~pricutList_() {
		delete array;
	}

} PriCutList;

typedef struct LUT_ {

	PriCut* bestCut;
	int rootId;
	int level;
	
	LUT_() {
		bestCut = nullptr;
		rootId = 0;
		level = 0;
	}

} LUT;

typedef galois::substrate::PerThreadStorage<PriCutPool> PerThreadPriCutPool;
typedef galois::substrate::PerThreadStorage<PriCutList> PerThreadPriCutList;
typedef galois::substrate::PerThreadStorage<AuxTruth> PerThreadAuxTruth;

typedef std::unordered_map< int, LUT > Covering;

class PriCutManager {

private:
  aig::Aig& aig;
  int K;
  int C;
  int nWords;
  int nNodes;
  int nThreads;
  long int cutPoolSize;
  bool compTruth;
	bool fPower;
	int sortMode;
	float fEpsilon;
  long double kcutTime;

  PerThreadPriCutPool perThreadPriCutPool;
  PerThreadPriCutList perThreadPriCutList;
  PerThreadAuxTruth perThreadAuxTruth;
  PriCut** nodePriCuts;

	Covering covering;
	int nLUTs;
	int nLevels;

  // Cuts Statistics //
  galois::GAccumulator<long int> nCuts;
  galois::GAccumulator<long int> nTriv;
  galois::GAccumulator<long int> nFilt;
  galois::GAccumulator<long int> nSatu;

  // Runtime Statistics //
  galois::GAccumulator<long int> mergeTime;
  galois::GAccumulator<long int> filterTime;
  galois::GAccumulator<long int> procTwoTime;
  galois::GAccumulator<long int> compTime;
  galois::GAccumulator<long int> scheduleTime;

  void computePriCutsRec(aig::GNode node, PriCutPool* cutPool, PriCutList* cutList, AuxTruth* auxTruth);
  PriCut* mergeCuts(PriCutPool* cutPool, PriCut* lhsCut, PriCut* rhsCut);
  inline bool cutFilter(PriCutPool* cutPool, PriCutList* cutList, PriCut* resCut);
  inline bool checkCutDominance(PriCut* smallerCut, PriCut* largerCut);
  inline void commitCuts(int nodeId, PriCutList* cutList);
  void computeTruth(AuxTruth* auxTruth, PriCut* resCut, PriCut* lhsCut, PriCut* rhsCut, bool lhsPolarity, bool rhsPolarity);
  inline unsigned truthPhase(PriCut* resCut, PriCut* inCut);
	float cutAreaRef( PriCut * cut );
	float cutAreaDeref( PriCut * cut );
	void cutSort(PriCutPool * cutPool, PriCutList * cutList, PriCut * resCut);
	int sortCompare(PriCut * lhsCut, PriCut * rhsCut);
	float cutAreaFlow( PriCut * cut );
	float cutAreaDerefed( PriCut * cut );
	float cutDelay( PriCut * cut );
	float cutEdgeFlow( PriCut * cut );
public:

  PriCutManager(aig::Aig& aig, int K, int C, int nThreads, bool compTruth);

  ~PriCutManager();

  void computePriCuts(PriCutPool* cutPool, PriCutList* cutList, AuxTruth* auxTruth,
                   int nodeId, int lhsId, int rhsId, bool lhsPolarity,
                   bool rhsPolarity);

  void computePriCutsRecursively(aig::GNode node);

  unsigned int* readTruth(PriCut* cut);
  void recycleNodeCuts(int nodeId);
	PriCut * getBestCut( int nodeId );
	void computeCovering();
	int computeCoveringRec( aig::Graph & aigGraph, aig::GNode node );
	void printCovering();
  void printNodeCuts(int nodeId, long int& counter);
  void printAllCuts();
	void printNodeBestCut(int nodeId);
	void printBestCuts();
  void printCutStatistics();
  void printRuntimes();

  aig::Aig& getAig();
	int getNumLUTs();
	int getNumLevels();
  int getK();
  int getC();
  int getNWords();
  int getNThreads();
  bool getCompTruthFlag();
  long double getKcutTime();
  void setKcutTime(long double time);
  PerThreadPriCutPool& getPerThreadPriCutPool();
  PerThreadPriCutList& getPerThreadPriCutList();
  PerThreadAuxTruth& getPerThreadAuxTruth();
  PriCut** getNodePriCuts();
	Covering & getCovering();
};

// Function that runs the KCut operator define in the end of file CutManager.cpp
// //
void runKPriCutOperator(PriCutManager& cutMan);

} /* namespace algorithm */

#endif /* PRIORITYCUTMANAGERC_H_ */
