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
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#include "CutManager.h"
#include "galois/Galois.h"
#include "galois/Bag.h"

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <assert.h>

using namespace std::chrono;

namespace algorithm {

CutManager::CutManager(aig::Aig& aig, int K, int C, int nThreads,
                       bool compTruth)
    :

      aig(aig), K(K), C(C),
      nWords(Functional32::wordNum(K)),
      nNodes(std::distance(aig.getGraph().begin(), aig.getGraph().end()) -
             aig.getNumOutputs()),
      nThreads(nThreads),
      cutPoolSize(nNodes / nThreads),
      compTruth(compTruth),
      perThreadCutPool(cutPoolSize, K, compTruth), perThreadCutList(K),
      perThreadAuxTruth(nWords) {

  kcutTime = 0;
  nodeCuts = new Cut*[nNodes + 1];
  for (int i = 0; i < nNodes + 1; i++) {
    nodeCuts[i] = nullptr;
  }
}

CutManager::~CutManager() { delete[] nodeCuts; }

void CutManager::computeCuts(CutPool* cutPool, CutList* cutList,
                             AuxTruth* auxTruth, int nodeId, int lhsId,
                             int rhsId, bool lhsPolarity, bool rhsPolarity) {

  int i;
  int currentNumCuts = 0;

  // start with the elementary cut
  Cut* trivialCut       = cutPool->getMemory();
  trivialCut->leaves[0] = nodeId;
  trivialCut->nLeaves++;
  trivialCut->sig = (1U << (nodeId % 31));
  if (this->compTruth) {
    unsigned* cutTruth = readTruth(trivialCut);
    for (int i = 0; i < this->nWords; i++) {
      cutTruth[i] = 0xAAAAAAAA;
    }
  }
  cutList->head[1] = trivialCut;
  cutList->tail[1] = trivialCut;
  currentNumCuts++;
  nCuts += 1;
  nTriv += 1;

  // std::chrono::high_resolution_clock::time_point t1 =
  // std::chrono::high_resolution_clock::now();

  Cut* lhsLargeCutsBegin;
  for (lhsLargeCutsBegin = this->nodeCuts[lhsId]; lhsLargeCutsBegin != nullptr;
       lhsLargeCutsBegin = lhsLargeCutsBegin->nextCut) {
    if (lhsLargeCutsBegin->nLeaves == this->K) {
      break;
    }
  }

  Cut* rhsLargeCutsBegin;
  for (rhsLargeCutsBegin = this->nodeCuts[rhsId]; rhsLargeCutsBegin != nullptr;
       rhsLargeCutsBegin = rhsLargeCutsBegin->nextCut) {
    if (rhsLargeCutsBegin->nLeaves == this->K) {
      break;
    }
  }

  // small by small
  for (Cut* lhsCut = this->nodeCuts[lhsId]; lhsCut != lhsLargeCutsBegin;
       lhsCut      = lhsCut->nextCut) {
    for (Cut* rhsCut = this->nodeCuts[rhsId]; rhsCut != rhsLargeCutsBegin;
         rhsCut      = rhsCut->nextCut) {
      if (processTwoCuts(cutPool, cutList, auxTruth, lhsCut, rhsCut,
                         lhsPolarity, rhsPolarity, currentNumCuts)) {
        commitCuts(nodeId, cutList);
        return; // The Maximum number of cuts per node was reached
      }
    }
  }

  // small by large
  for (Cut* lhsCut = this->nodeCuts[lhsId]; lhsCut != lhsLargeCutsBegin;
       lhsCut      = lhsCut->nextCut) {
    for (Cut* rhsCut = rhsLargeCutsBegin; rhsCut != nullptr;
         rhsCut      = rhsCut->nextCut) {
      if ((lhsCut->sig & rhsCut->sig) != lhsCut->sig) {
        continue;
      }
      if (processTwoCuts(cutPool, cutList, auxTruth, lhsCut, rhsCut,
                         lhsPolarity, rhsPolarity, currentNumCuts)) {
        commitCuts(nodeId, cutList);
        return; // The Maximum number of cuts per node was reached
      }
    }
  }

  // small by large
  for (Cut* rhsCut = this->nodeCuts[rhsId]; rhsCut != rhsLargeCutsBegin;
       rhsCut      = rhsCut->nextCut) {
    for (Cut* lhsCut = lhsLargeCutsBegin; lhsCut != nullptr;
         lhsCut      = lhsCut->nextCut) {
      if ((lhsCut->sig & rhsCut->sig) != rhsCut->sig) {
        continue;
      }
      if (processTwoCuts(cutPool, cutList, auxTruth, lhsCut, rhsCut,
                         lhsPolarity, rhsPolarity, currentNumCuts)) {
        commitCuts(nodeId, cutList);
        return; // The Maximum number of cuts per node was reached
      }
    }
  }

  // large by large
  for (Cut* lhsCut = lhsLargeCutsBegin; lhsCut != nullptr;
       lhsCut      = lhsCut->nextCut) {
    for (Cut* rhsCut = rhsLargeCutsBegin; rhsCut != nullptr;
         rhsCut      = rhsCut->nextCut) {
      if (lhsCut->sig != rhsCut->sig) {
        continue;
      }
      for (i = 0; i < this->K; i++) {
        if (lhsCut->leaves[i] != rhsCut->leaves[i]) {
          break;
        }
      }
      if (i < this->K) {
        continue;
      }
      if (processTwoCuts(cutPool, cutList, auxTruth, lhsCut, rhsCut,
                         lhsPolarity, rhsPolarity, currentNumCuts)) {
        commitCuts(nodeId, cutList);
        return; // The Maximum number of cuts per node was reached
      }
    }
  }

  // Copy from currentCutList to the nodeCuts
  commitCuts(nodeId, cutList);

  // std::chrono::high_resolution_clock::time_point t2 =
  // std::chrono::high_resolution_clock::now(); compTime +=
  // std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
}

void CutManager::computeCutsRecursively(aig::GNode node) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  if (this->nodeCuts[nodeData.id] == nullptr) {

    auto inEdgeIt          = aigGraph.in_edge_begin(node);
    aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& lhsData = aigGraph.getData(lhsNode);
    bool lhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    inEdgeIt++;
    aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& rhsData = aigGraph.getData(rhsNode);
    bool rhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    CutPool* cutPool   = this->perThreadCutPool.getLocal();
    CutList* cutList   = this->perThreadCutList.getLocal();
    AuxTruth* auxTruth = this->perThreadAuxTruth.getLocal();

    computeCutsRec(lhsNode, cutPool, cutList, auxTruth);
    computeCutsRec(rhsNode, cutPool, cutList, auxTruth);

    computeCuts(cutPool, cutList, auxTruth, nodeData.id, lhsData.id, rhsData.id,
                lhsPolarity, rhsPolarity);
  }
}

void CutManager::computeCutsRec(aig::GNode node, CutPool* cutPool,
                                CutList* cutList, AuxTruth* auxTruth) {

  aig::Graph& aigGraph    = this->aig.getGraph();
  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  if (this->nodeCuts[nodeData.id] == nullptr) {

    auto inEdgeIt          = aigGraph.in_edge_begin(node);
    aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& lhsData = aigGraph.getData(lhsNode);
    bool lhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    inEdgeIt++;
    aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& rhsData = aigGraph.getData(rhsNode);
    bool rhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

    computeCutsRec(lhsNode, cutPool, cutList, auxTruth);
    computeCutsRec(rhsNode, cutPool, cutList, auxTruth);

    computeCuts(cutPool, cutList, auxTruth, nodeData.id, lhsData.id, rhsData.id,
                lhsPolarity, rhsPolarity);
  }
}

inline bool CutManager::processTwoCuts(CutPool* cutPool, CutList* cutList,
                                       AuxTruth* auxTruth, Cut* lhsCut,
                                       Cut* rhsCut, bool lhsPolarity,
                                       bool rhsPolarity,
                                       int& currentNumCuts) {

  // std::chrono::high_resolution_clock::time_point t0 =
  // std::chrono::high_resolution_clock::now();

  Cut* resCut;

  // merge the cuts
  // std::chrono::high_resolution_clock::time_point t1 =
  // std::chrono::high_resolution_clock::now();
  if (lhsCut->nLeaves >= rhsCut->nLeaves) {
    resCut = mergeCuts(cutPool, lhsCut, rhsCut);
  } else {
    resCut = mergeCuts(cutPool, rhsCut, lhsCut);
  }
  // std::chrono::high_resolution_clock::time_point t2 =
  // std::chrono::high_resolution_clock::now(); mergeTime +=
  // std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

  if (resCut == nullptr) {
    return false;
  }

  // assert( (resCut->nLeaves > 1) && (resCut->nLeaves <= K) );

  // set the signature
  resCut->sig = lhsCut->sig | rhsCut->sig;

  // std::chrono::high_resolution_clock::time_point t3 =
  // std::chrono::high_resolution_clock::now();
  // check containment
  if (cutFilter(cutPool, cutList, resCut, currentNumCuts)) {
    return false;
  }
  // std::chrono::high_resolution_clock::time_point t4 =
  // std::chrono::high_resolution_clock::now(); filterTime +=
  // std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();

  if (this->compTruth) {
    computeTruth(auxTruth, resCut, lhsCut, rhsCut, lhsPolarity, rhsPolarity);
    // printf( "%x\n", (*readTruth( resCut )) );
  }

  // add to the list
  if (cutList->head[resCut->nLeaves] == nullptr) {
    cutList->head[resCut->nLeaves] = resCut;
  } else {
    cutList->tail[resCut->nLeaves]->nextCut = resCut;
  }
  cutList->tail[resCut->nLeaves] = resCut;
  currentNumCuts++;
  nCuts += 1;

  // std::chrono::high_resolution_clock::time_point t5 =
  // std::chrono::high_resolution_clock::now(); procTwoTime +=
  // std::chrono::duration_cast<std::chrono::microseconds>( t5 - t0 ).count();

  // return status (0 if okay; 1 if exceeded the limit)

  if (currentNumCuts >= this->C) {
    nSatu += 1;
    return true;
  }

  return false;
}

Cut* CutManager::mergeCuts(CutPool* cutPool, Cut* lhsCut, Cut* rhsCut) {

  int i, j, l;

  // assert( lhsCut->nLeaves >= rhsCut->nLeaves );

  Cut* resCut;

  // the case of the largest cut sizes
  if (lhsCut->nLeaves == this->K && rhsCut->nLeaves == this->K) {
    for (i = 0; i < lhsCut->nLeaves; i++) {
      if (lhsCut->leaves[i] != rhsCut->leaves[i]) {
        return nullptr;
      }
    }
    resCut = cutPool->getMemory();
    for (i = 0; i < lhsCut->nLeaves; i++) {
      resCut->leaves[i] = lhsCut->leaves[i];
    }
    resCut->nLeaves = lhsCut->nLeaves;
    return resCut;
  }

  // the case when one of the cuts is the largest
  if (lhsCut->nLeaves == this->K) {
    for (i = 0; i < rhsCut->nLeaves; i++) {
      for (j = lhsCut->nLeaves - 1; j >= 0; j--) {
        if (lhsCut->leaves[j] == rhsCut->leaves[i]) {
          break;
        }
      }
      if (j == -1) { // did not find
        return nullptr;
      }
    }
    resCut = cutPool->getMemory();
    for (i = 0; i < lhsCut->nLeaves; i++) {
      resCut->leaves[i] = lhsCut->leaves[i];
    }
    resCut->nLeaves = lhsCut->nLeaves;
    return resCut;
  }

  // compare two cuts with different numbers
  resCut = cutPool->getMemory();
  i      = 0;
  j      = 0;
  for (l = 0; l < this->K; l++) {
    if (j == rhsCut->nLeaves) {
      if (i == lhsCut->nLeaves) {
        resCut->nLeaves = l;
        return resCut;
      }
      resCut->leaves[l] = lhsCut->leaves[i++];
      continue;
    }

    if (i == lhsCut->nLeaves) {
      if (j == rhsCut->nLeaves) {
        resCut->nLeaves = l;
        return resCut;
      }
      resCut->leaves[l] = rhsCut->leaves[j++];
      continue;
    }

    if (lhsCut->leaves[i] < rhsCut->leaves[j]) {
      resCut->leaves[l] = lhsCut->leaves[i++];
      continue;
    }

    if (lhsCut->leaves[i] > rhsCut->leaves[j]) {
      resCut->leaves[l] = rhsCut->leaves[j++];
      continue;
    }

    resCut->leaves[l] = lhsCut->leaves[i++];
    j++;
  }

  if (i < lhsCut->nLeaves || j < rhsCut->nLeaves) {
    cutPool->giveBackMemory(resCut);
    return nullptr;
  }

  resCut->nLeaves = l;
  return resCut;
}

inline bool CutManager::cutFilter(CutPool* cutPool, CutList* cutList,
                                  Cut* resCut, int& currentNumCuts) {

  // check if this cut is filtered out by smaller cuts
  for (int i = 2; i <= resCut->nLeaves; i++) {

    for (Cut* cut = cutList->head[i]; cut != nullptr; cut = cut->nextCut) {

      // skip the non-contained cuts
      if ((cut->sig & resCut->sig) != cut->sig) {
        continue;
      }
      // check containment seriously
      if (checkCutDominance(cut, resCut)) {
        // Recycle Cut
        cutPool->giveBackMemory(resCut);
        nFilt += 1;
        return true; // resCut is dominated
      }
    }
  }

  // filter out other cuts using this one
  for (int i = resCut->nLeaves + 1; i <= this->K; i++) {

    Cut* prevCut  = nullptr;
    Cut* toRemove = nullptr;
    Cut* cut      = cutList->head[i];

    while (cut != nullptr) {

      // sKip the non-contained cuts
      if ((cut->sig & resCut->sig) != resCut->sig) {
        prevCut = cut;
        cut     = cut->nextCut;
        continue;
      }
      // check containment seriously
      if (checkCutDominance(resCut, cut)) {

        currentNumCuts--;
        nCuts -= 1;
        nFilt += 1;

        // when the cut to be removed is the first of the list
        if (cut == cutList->head[i]) {
          cutList->head[i] = cut->nextCut;
          toRemove         = cut;
          cut              = cut->nextCut;
          // Recycle Cut
          cutPool->giveBackMemory(toRemove);
          continue;
        }

        // when the cut to be removed is in the middle or in the end of the list
        if (prevCut != nullptr) {
          prevCut->nextCut = cut->nextCut;
          toRemove         = cut;
          cut              = cut->nextCut;
          // Recycle Cut
          cutPool->giveBackMemory(toRemove);
        } else {
          std::cout << "Bug cut removal!" << std::endl;
          exit(1);
        }
      } else {
        prevCut = cut;
        cut     = cut->nextCut;
      }
    }

    cutList->tail[i] = prevCut;
  }

  return false;
}

inline bool CutManager::checkCutDominance(Cut* smallerCut, Cut* largerCut) {

  int i, j;

  for (i = 0; i < smallerCut->nLeaves; i++) {
    for (j = 0; j < largerCut->nLeaves; j++) {
      if (smallerCut->leaves[i] == largerCut->leaves[j]) {
        break;
      }
    }
    if (j == largerCut->nLeaves) { // node i in smallerCut is not contained in largerCut
      return false;
    }
  }
  // every node in smallerCut is contained in largerCut
  return true;
}

void CutManager::commitCuts(int nodeId, CutList* cutList) {

  // Copy from currentCutList to the nodeCuts
  this->nodeCuts[nodeId] = cutList->head[1];
  Cut* lastCut           = cutList->head[1];
  cutList->head[1]       = nullptr;
  for (int i = 2; i < this->K + 1; i++) {
    if (cutList->head[i] == nullptr) {
      continue;
    }
    lastCut->nextCut = cutList->head[i];
    lastCut          = cutList->tail[i];
    cutList->head[i] = nullptr;
    cutList->tail[i] = nullptr;
  }
}

void CutManager::computeTruth(AuxTruth* auxTruth, Cut* resCut, Cut* lhsCut,
                              Cut* rhsCut, bool lhsPolarity, bool rhsPolarity) {

  // permute the first table
  if (lhsPolarity) {
    Functional32::copy(auxTruth->truth[0], readTruth(lhsCut), this->nWords);
  } else {
    Functional32::NOT(auxTruth->truth[0], readTruth(lhsCut), this->nWords);
  }
  Functional32::truthStretch(auxTruth->truth[2], auxTruth->truth[0],
                             lhsCut->nLeaves, this->K,
                             truthPhase(resCut, lhsCut));

  // permute the second table
  if (rhsPolarity) {
    Functional32::copy(auxTruth->truth[1], readTruth(rhsCut), this->nWords);
  } else {
    Functional32::NOT(auxTruth->truth[1], readTruth(rhsCut), this->nWords);
  }
  Functional32::truthStretch(auxTruth->truth[3], auxTruth->truth[1],
                             rhsCut->nLeaves, this->K,
                             truthPhase(resCut, rhsCut));

  // produce the resulting table. In this first version we are not considering
  // the cut->fCompl flag. It may be considerer in further versions according to
  // the demand.
  // if ( cut->fCompl ) {
  //	Functional32::NAND( readTruth( cut ) , auxTruth[2], auxTruth[3], K );
  //}
  // else {
  Functional32::AND(readTruth(resCut), auxTruth->truth[2], auxTruth->truth[3],
                    this->nWords);
  //}
}

inline unsigned CutManager::truthPhase(Cut* resCut, Cut* inCut) {

  unsigned phase = 0;
  int i, j;
  for (i = j = 0; i < resCut->nLeaves; i++) {

    if (j == inCut->nLeaves) {
      break;
    }
    if (resCut->leaves[i] < inCut->leaves[j]) {
      continue;
    }

    assert(resCut->leaves[i] == inCut->leaves[j]);
    phase |= (1 << i);
    j++;
  }

  return phase;
}

unsigned int* CutManager::readTruth(Cut* cut) {
  return (unsigned*)(cut->leaves + this->K);
}

/*
 *     This method gives the cut's memory back to current thread cutPool.
 *     However, the memory can be allocated by the cutPool of one thread
 *     and returned to cutPool of another thread.
 */
void CutManager::recycleNodeCuts(int nodeId) {

  CutPool* cutPool = this->perThreadCutPool.getLocal();
  Cut* cut = this->nodeCuts[nodeId];

	while ( cut != nullptr ) {
		Cut* nextCut = cut->nextCut;
    cutPool->giveBackMemory(cut);
		cut = nextCut;
  }

  this->nodeCuts[nodeId] = nullptr;
}

void CutManager::printNodeCuts(int nodeId, long int& counter) {

  std::cout << "Node " << nodeId << ": { ";
  for (Cut* currentCut = this->nodeCuts[nodeId]; currentCut != nullptr;
       currentCut      = currentCut->nextCut) {
    counter++;
    std::cout << "{ ";
    for (int i = 0; i < currentCut->nLeaves; i++) {
      std::cout << currentCut->leaves[i] << " ";
    }
    std::cout << "} ";
  }
  std::cout << "}" << std::endl;
}

void CutManager::printAllCuts() {

  long int counter     = 0;
  aig::Graph& aigGraph = this->aig.getGraph();

  std::cout << std::endl << "########## All K-Cuts ###########" << std::endl;
  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    if ((nodeData.type == aig::NodeType::AND) ||
        (nodeData.type == aig::NodeType::PI)) {
      printNodeCuts(nodeData.id, counter);
    }
  }
  std::cout << "#################################" << std::endl;
}

void CutManager::printCutStatistics() {

  long int nCutsRed = nCuts.reduce();
  nCutsRed += this->aig.getNumInputs();

  long int nTrivRed = nTriv.reduce();
  nTrivRed += this->aig.getNumInputs();

  long int nFiltRed = nFilt.reduce();

  long int nSatuRed = nSatu.reduce();


  std::cout << std::endl << "############## Cut Statistics #############" << std::endl;
  std::cout << "nCuts: " << nCutsRed << std::endl;
  std::cout << "nTriv: " << nTrivRed << std::endl;
  std::cout << "nFilt: " << nFiltRed << std::endl;
  std::cout << "nSatu: " << nSatuRed << std::endl;
  std::cout << "nCutPerNode: " << (((double)nCutsRed) / this->nNodes) << std::endl;
  std::cout << "###########################################" << std::endl;
}

void CutManager::printRuntimes() {

  std::cout << std::endl << "#### Runtimes in microsecond ####" << std::endl;
  // std::cout << "Merge: " << mergeTime << std::endl;
  // std::cout << "Filter: " << filterTime << std::endl;
  // std::cout << "ProcTwo: " << procTwoTime << std::endl;
  // std::cout << "Compute: " << compTime << std::endl;
  // std::cout << "Schedule: " << scheduleTime << std::endl;
  std::cout << "Total: " << this->kcutTime << std::endl;
  std::cout << "#################################" << std::endl;
}

aig::Aig& CutManager::getAig() { return this->aig; }

int CutManager::getK() { return this->K; }

int CutManager::getC() { return this->C; }

int CutManager::getNWords() { return this->nWords; }

int CutManager::getNThreads() { return this->nThreads; }

bool CutManager::getCompTruthFlag() { return this->compTruth; }

long double CutManager::getKcutTime() { return this->kcutTime; }

void CutManager::setKcutTime(long double time) { this->kcutTime = time; }

PerThreadCutPool& CutManager::getPerThreadCutPool() {
  return this->perThreadCutPool;
}

PerThreadCutList& CutManager::getPerThreadCutList() {
  return this->perThreadCutList;
}

PerThreadAuxTruth& CutManager::getPerThreadAuxTruth() {
  return this->perThreadAuxTruth;
}

Cut** CutManager::getNodeCuts() { return this->nodeCuts; }

// ######################## BEGIN OPERATOR ######################## //
struct KCutOperator {

  CutManager& cutMan;

  KCutOperator(CutManager& cutMan) : cutMan(cutMan) {}

  void operator()(aig::GNode node, galois::UserContext<aig::GNode>& ctx) {

    aig::Aig& aig        = cutMan.getAig();
    aig::Graph& aigGraph = aig.getGraph();

    aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

    if (nodeData.type == aig::NodeType::AND) {

      // Touching outgoing neighobors to acquire their locks
      aigGraph.out_edges(node);

      // Combine Cuts
      auto inEdgeIt          = aigGraph.in_edge_begin(node);
      aig::GNode lhsNode     = aigGraph.getEdgeDst(inEdgeIt);
      aig::NodeData& lhsData = aigGraph.getData(lhsNode);
      bool lhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

      inEdgeIt++;
      aig::GNode rhsNode     = aigGraph.getEdgeDst(inEdgeIt);
      aig::NodeData& rhsData = aigGraph.getData(rhsNode);
      bool rhsPolarity       = aigGraph.getEdgeData(inEdgeIt);

      CutPool* cutPool   = cutMan.getPerThreadCutPool().getLocal();
      CutList* cutList   = cutMan.getPerThreadCutList().getLocal();
      AuxTruth* auxTruth = cutMan.getPerThreadAuxTruth().getLocal();

			//ctx.cautiousPoint();

      cutMan.computeCuts(cutPool, cutList, auxTruth, nodeData.id, lhsData.id,
                         rhsData.id, lhsPolarity, rhsPolarity);

      // NextNodes
      for (auto edge : aigGraph.out_edges(node)) {
        aig::GNode nextNode = aigGraph.getEdgeDst(edge);
        aig::NodeData& nextNodeData =
            aigGraph.getData(nextNode, galois::MethodFlag::WRITE);
        nextNodeData.counter += 1;
        if (nextNodeData.counter == 2) {
          ctx.push(nextNode);
        }
      }
    } else {
      if (nodeData.type == aig::NodeType::PI) {
        // Touching outgoing neighobors to acquire their locks and their fanin
        // node's locks.
        aigGraph.out_edges(node);

				//ctx.cautiousPoint();

        // Set the trivial cut
        nodeData.counter      = 3;
        CutPool* cutPool      = cutMan.getPerThreadCutPool().getLocal();
        Cut* trivialCut       = cutPool->getMemory();
        trivialCut->leaves[0] = nodeData.id;
        trivialCut->nLeaves++;
        trivialCut->sig = (1U << (nodeData.id % 31));
        if (cutMan.getCompTruthFlag()) {
          unsigned* cutTruth = cutMan.readTruth(trivialCut);
          for (int i = 0; i < cutMan.getNWords(); i++) {
            cutTruth[i] = 0xAAAAAAAA;
          }
        }
        cutMan.getNodeCuts()[nodeData.id] = trivialCut;

        // Schedule next nodes
        for (auto edge : aigGraph.out_edges(node)) {
          aig::GNode nextNode = aigGraph.getEdgeDst(edge);
          aig::NodeData& nextNodeData =
              aigGraph.getData(nextNode, galois::MethodFlag::WRITE);
          nextNodeData.counter += 1;
          if (nextNodeData.counter == 2) {
            ctx.push(nextNode);
          }
        }
      }
    }
  }
};

void runKCutOperator(CutManager& cutMan) {

  galois::InsertBag<aig::GNode> workList;
  typedef galois::worklists::PerSocketChunkBag<500> DC_BAG;
  // typedef galois::worklists::PerSocketChunkFIFO< 200 > DC_FIFO;
  // typedef galois::worklists::PerSocketChunkLIFO< 200 > DC_LIFO;
  // typedef galois::worklists::PerThreadChunkFIFO< 200 > AC_FIFO;

  for (auto pi : cutMan.getAig().getInputNodes()) {
    workList.push(pi);
  }

  // Galois Parallel Foreach
  galois::for_each(galois::iterate(workList.begin(), workList.end()),
                   KCutOperator(cutMan),
									 galois::wl<DC_BAG>(),
                   galois::loopname("KCutOperator"));


	//galois::wl<galois::worklists::Deterministic<>>(),
	//galois::wl<DC_BAG>(),
	
}
// ######################## END OPERATOR ######################## //

} /* namespace algorithm */
