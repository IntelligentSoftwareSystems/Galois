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
#include <unordered_set>

namespace algorithm {

typedef std::unordered_map< int, PriCut* > Covering;

// #### THREAD LOCAL #### //
typedef struct pricutList_ {

  PriCut ** array;
	int 			nCuts;

	pricutList_(int C) {
		array = new PriCut*[C+2];
		nCuts	= 0;
	}

	~pricutList_() {
		delete array;
	}

} PriCutList;

typedef galois::PerIterAllocTy::rebind< std::pair<const int, int> >::other MapAlloc;
typedef std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, MapAlloc> RefMap;

typedef struct threadLocalData_ {

	PriCutPool cutPool;
	PriCutList cutList;
	AuxTruth auxTruth;

	threadLocalData_( int cutPoolSize, int K, bool compTruth, int C, int nWords) :
									  cutPool( cutPoolSize, K, compTruth ), cutList( C ), auxTruth( nWords ) { } 

} ThreadLocalData; 

typedef galois::substrate::PerThreadStorage< ThreadLocalData > PerThreadData;


// #### CONTROL TYPES #### //
enum SortMode { DELAY, DELAY_OLD, AREA };
enum CostMode { AREA_FLOW, LOCAL_AREA };
enum RefMode  { STANDARD, MAP };

class PriCutManager {

private:
  aig::Aig& aig;
	aig::Graph& aigGraph;
  int K;
  int C;
  int nWords;
  int nNodes;
  int nThreads;
  long int cutPoolSize;
  bool compTruth;
	bool deterministic;
	bool verbose;
	int passId;
	int sortMode;
	int costMode;
	int refMode;
	int nLUTs;
	int nLevels;
	bool fPower;
	float fEpsilon;
  long double kcutTime;

	PerThreadData perThreadData;
  
	PriCut** nodePriCuts;
	Covering covering;
	std::vector<aig::GNode> sortedNodes;

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

  void computePriCutsRec(aig::GNode node, ThreadLocalData* thData, RefMap& refMap);

  PriCut* mergeCuts(PriCutPool& cutPool, PriCut* lhsCut, PriCut* rhsCut);
  inline bool cutFilter(PriCutPool& cutPool, PriCutList& cutList, PriCut* resCut);
  inline bool checkCutDominance(PriCut* smallerCut, PriCut* largerCut);
  
	inline void commitCuts(PriCutPool& cutPool, PriCutList& cutList, int nodeId);

  inline void recycleNodeCuts(PriCutPool& cutPool, int nodeId);
	inline void cleanupCutList(PriCutPool& cutPool, PriCutList& cutList);

  void computeTruth(AuxTruth& auxTruth, PriCut* resCut, PriCut* lhsCut, PriCut* rhsCut, bool lhsPolarity, bool rhsPolarity);
  inline unsigned truthPhase(PriCut* resCut, PriCut* inCut);
	
	void cutSort(PriCutPool& cutPool, PriCutList& cutList, PriCut* resCut);
	int sortCompare(PriCut* lhsCut, PriCut* rhsCut);

	void increaseCutReferences(PriCut* cut);
	void decreaseCutReferences(PriCut* cut);

	// ################### Start of the NewCuts Cost Functions ###################### //	
	inline float cutDelay(PriCut* cut);
	// STANDARD VERSIONS
	void cutFlowCosts(PriCut* cut);
	void cutDerefedCosts(PriCut* cut);
	void cutRefCosts(PriCut* cut, float& area, float& edge);
	void cutDerefCosts(PriCut* cut, float& area, float& edge);
	// REFMAP RERSIONS
	void cutDerefedCosts(PriCut* cut, RefMap& refMap);
	void cutRefCosts(PriCut* cut, float& area, float& edge, RefMap& refMap);
	void cutDerefCosts(PriCut* cut, float& area, float& edge, RefMap& refMap);
	// ################### End of the NewCuts Cost Functions ###################### //

	inline void copyCut(PriCut* dest, PriCut* source);

public:

  PriCutManager(aig::Aig& aig, int K, int C, int nThreads, bool compTruth, bool deterministic, bool verbose);

  ~PriCutManager();

  void computePriCuts(ThreadLocalData* thData, RefMap& refMap, aig::NodeData& nodeData, int lhsId, int rhsId, bool lhsPolarity, bool rhsPolarity);

	void mapChoices(ThreadLocalData* thData, RefMap& refMap, aig::NodeData & nodeData);

  void computePriCutsRecursively(aig::GNode node, RefMap& refMap);

  unsigned int* readTruth(PriCut* cut);
	inline void switchToFirstDelayMode();
	inline void switchToSecondDelayMode();
	inline void switchToAreaFlowMode();
	inline void switchToLocalAreaMode();

	void resetNodeCountersFanout();
	void resetNodeCountersZero();
	void resetNodeCountersOnly();
	void computeReferenceCounters();
	void computeCoveringReferenceCounters();
	void computeRequiredTimes();
	void computeCovering();

	void printCovering();
  void printNodeCuts(int nodeId, long int& counter);
  void printAllCuts();
	void printNodeBestCut(int nodeId);
	void printBestCuts();
  void printCutStatistics();
  void printRuntimes();

	int getNumLUTs();
	int getNumLevels();
  int getK();
  int getC();
  int getNWords();
  int getNThreads();
	bool isDeterministic();
  bool getCompTruthFlag();
  bool getVerboseFlag();
  long double getKcutTime();
  void setKcutTime(long double time);

  inline aig::Aig& getAig();
	inline PriCut* getBestCut( int nodeId );
  PriCut** getNodePriCuts();
	Covering & getCovering();

  PerThreadData& getPerThreadData();
};

// Function that runs the KCut operator define in the end of file CutManager.cpp //
void runKPriCutOperator(PriCutManager& cutMan);

} /* namespace algorithm */

#endif /* PRIORITYCUTMANAGERC_H_ */

