/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#ifndef CUTMANAGER_H_
#define CUTMANAGER_H_

#include "Aig.h"
#include "CutPool.h"
#include "../functional/FunctionHandler32.h"
#include "galois/Reduction.h"

namespace algorithm {

typedef struct cutList_ {

  Cut** head;
  Cut** tail;

  cutList_(int K) {
    head = new Cut*[K + 1];
    for (int i = 0; i < K + 1; i++) {
      head[i] = nullptr;
    }

    tail = new Cut*[K + 1];
    for (int i = 0; i < K + 1; i++) {
      tail[i] = nullptr;
    }
  }

  ~cutList_() {
    delete[] head;
    delete[] tail;
  }

} CutList;

typedef struct auxTruth_ {

  unsigned int* truth[4];

  auxTruth_(int nWords) {
    for (int i = 0; i < 4; i++) {
      truth[i] = new unsigned int[nWords];
    }
  }

  ~auxTruth_() {
    for (int i = 0; i < 4; i++) {
      delete[] truth[i];
    }
  }

} AuxTruth;

typedef galois::substrate::PerThreadStorage<CutPool> PerThreadCutPool;
typedef galois::substrate::PerThreadStorage<CutList> PerThreadCutList;
typedef galois::substrate::PerThreadStorage<AuxTruth> PerThreadAuxTruth;

class CutManager {

private:
  aig::Aig& aig;
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
  Cut** nodeCuts;

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

  void computeCutsRec(aig::GNode node, CutPool* cutPool, CutList* cutList,
                      AuxTruth* auxTruth);

  inline bool processTwoCuts(CutPool* cutPool, CutList* cutList,
                             AuxTruth* auxTruth, Cut* lhsCut, Cut* rhsCut,
                             bool lhsPolarity, bool rhsPolarity,
                             int& currentNumCuts);

  Cut* mergeCuts(CutPool* cutPool, Cut* lhsCut, Cut* rhsCut);

  inline bool cutFilter(CutPool* cutPool, CutList* cutList, Cut* resCut,
                        int& currentNumCuts);

  inline bool checkCutDominance(Cut* smallerCut, Cut* largerCut);

  inline void commitCuts(int nodeId, CutList* cutList);

  void computeTruth(AuxTruth* auxTruth, Cut* resCut, Cut* lhsCut, Cut* rhsCut,
                    bool lhsPolarity, bool rhsPolarity);

  inline unsigned truthPhase(Cut* resCut, Cut* inCut);

public:
  CutManager(aig::Aig& aig, int K, int C, int nThreads, bool compTruth);

  ~CutManager();

  void computeCuts(CutPool* cutPool, CutList* cutList, AuxTruth* auxTruth,
                   int nodeId, int lhsId, int rhsId, bool lhsPolarity,
                   bool rhsPolarity);

  void computeCutsRecursively(aig::GNode node);

  unsigned int* readTruth(Cut* cut);
  void recycleNodeCuts(int nodeId);
  void printNodeCuts(int nodeId, long int& counter);
  void printAllCuts();
  void printCutStatistics();
  void printRuntimes();

  aig::Aig& getAig();
  int getK();
  int getC();
  int getNWords();
  int getNThreads();
  bool getCompTruthFlag();
  long double getKcutTime();
  void setKcutTime(long double time);
  PerThreadCutPool& getPerThreadCutPool();
  PerThreadCutList& getPerThreadCutList();
  PerThreadAuxTruth& getPerThreadAuxTruth();
  Cut** getNodeCuts();
};

// Function that runs the KCut operator define in the end of file CutManager.cpp
// //
void runKCutOperator(CutManager& cutMan);

} /* namespace algorithm */

#endif /* CUTMANAGERC_H_ */
