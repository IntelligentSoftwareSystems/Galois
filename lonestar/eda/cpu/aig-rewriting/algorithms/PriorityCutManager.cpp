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
 Parallel Parallel LUT-Based Tech Mapping October 16, 2018.
 ABC-based implementation on Galois.

*/

#include "PriorityCutManager.h"
#include "galois/Galois.h"
#include "galois/Bag.h"

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <assert.h>

using namespace std::chrono;

namespace algorithm {

PriCutManager::PriCutManager(aig::Aig& aig, int K, int C, int nThreads,
                             bool compTruth, bool deterministic, bool verbose)
    : aig(aig), aigGraph(aig.getGraph()), K(K), C(C),
      nWords(Functional32::wordNum(K)),
      nNodes(std::distance(aig.getGraph().begin(), aig.getGraph().end()) -
             aig.getNumOutputs()),
      nThreads(nThreads), cutPoolSize(nNodes / nThreads), compTruth(compTruth),
      deterministic(deterministic), verbose(verbose),
      perThreadData(cutPoolSize, K, compTruth, C, nWords) {

  nLUTs   = 0;
  nLevels = 0;
  passId  = 0;

  sortMode = SortMode::DELAY;
  costMode = CostMode::AREA_FLOW;

  if (deterministic) {
    refMode = RefMode::MAP;
  } else {
    refMode = RefMode::STANDARD;
  }

  fPower   = false;
  fEpsilon = (float)0.005;
  kcutTime = 0;

  nodePriCuts = new PriCut*[nNodes + 1];
  for (int i = 0; i < nNodes + 1; i++) {
    nodePriCuts[i] = nullptr;
  }

  // iterating from 0 to N is reverse topological order
  // iterating from N to 0 is topological order
  aig.computeGenericTopologicalSortForAnds(this->sortedNodes);
}

PriCutManager::~PriCutManager() { delete[] nodePriCuts; }

void PriCutManager::computePriCutsRecursively(aig::GNode node, RefMap& refMap) {

  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  if (this->nodePriCuts[nodeData.id] == nullptr) {

    auto inEdgeIt      = aigGraph.in_edge_begin(node);
    aig::GNode lhsNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& lhsData =
        aigGraph.getData(lhsNode, galois::MethodFlag::READ);
    bool lhsPolarity = aigGraph.getEdgeData(inEdgeIt);

    inEdgeIt++;
    aig::GNode rhsNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& rhsData =
        aigGraph.getData(rhsNode, galois::MethodFlag::READ);
    bool rhsPolarity = aigGraph.getEdgeData(inEdgeIt);

    ThreadLocalData* thData = this->perThreadData.getLocal();

    computePriCutsRec(lhsNode, thData, refMap);
    computePriCutsRec(rhsNode, thData, refMap);

    computePriCuts(thData, refMap, nodeData, lhsData.id, rhsData.id,
                   lhsPolarity, rhsPolarity);
  }
}

void PriCutManager::computePriCutsRec(aig::GNode node, ThreadLocalData* thData,
                                      RefMap& refMap) {

  aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

  if (this->nodePriCuts[nodeData.id] == nullptr) {

    auto inEdgeIt      = aigGraph.in_edge_begin(node);
    aig::GNode lhsNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& lhsData =
        aigGraph.getData(lhsNode, galois::MethodFlag::READ);
    bool lhsPolarity = aigGraph.getEdgeData(inEdgeIt);

    inEdgeIt++;
    aig::GNode rhsNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& rhsData =
        aigGraph.getData(rhsNode, galois::MethodFlag::READ);
    bool rhsPolarity = aigGraph.getEdgeData(inEdgeIt);

    computePriCutsRec(lhsNode, thData, refMap);
    computePriCutsRec(rhsNode, thData, refMap);

    computePriCuts(thData, refMap, nodeData, lhsData.id, rhsData.id,
                   lhsPolarity, rhsPolarity);
  }
}

void PriCutManager::computePriCuts(ThreadLocalData* thData, RefMap& refMap,
                                   aig::NodeData& nodeData, int lhsId,
                                   int rhsId, bool lhsPolarity,
                                   bool rhsPolarity) {

  PriCut *trivialCut, *resCut, *oldBestCut = nullptr;

  cleanupCutList(thData->cutPool,
                 thData->cutList); // Ensure that the cutList is empty

  if ((this->passId > 1) && (this->nodePriCuts[nodeData.id] != nullptr)) {
    // Save a copy of the previous bestCut and recompute the cut's costs
    oldBestCut = thData->cutPool.getMemory();
    copyCut(oldBestCut, this->nodePriCuts[nodeData.id]);

    if (this->costMode == CostMode::AREA_FLOW) {
      cutFlowCosts(oldBestCut);
    } else {
      if (this->refMode == RefMode::MAP) {
        cutDerefedCosts(oldBestCut, refMap);
      } else {
        cutDerefedCosts(oldBestCut);
      }
    }

    if (this->deterministic == false) {
      decreaseCutReferences(oldBestCut);
    }
  }

  for (PriCut* lhsCut = this->nodePriCuts[lhsId]; lhsCut != nullptr;
       lhsCut         = lhsCut->nextCut) {
    for (PriCut* rhsCut = this->nodePriCuts[rhsId]; rhsCut != nullptr;
         rhsCut         = rhsCut->nextCut) {

      if (Functional32::countOnes(lhsCut->sig | rhsCut->sig) > this->K) {
        continue;
      }

      // merge the cuts
      if (lhsCut->nLeaves >= rhsCut->nLeaves) {
        resCut = mergeCuts(thData->cutPool, lhsCut, rhsCut);
      } else {
        resCut = mergeCuts(thData->cutPool, rhsCut, lhsCut);
      }

      if (resCut == nullptr) {
        continue;
      }

      // check containment
      if (cutFilter(thData->cutPool, thData->cutList, resCut)) {
        continue;
      }

      if (this->compTruth) {
        computeTruth(thData->auxTruth, resCut, lhsCut, rhsCut, lhsPolarity,
                     rhsPolarity);
        // std::cout << Functional32::toHex( readTruth( resCut ), getNWords() )
        // << std::endl;
      }

      if (this->costMode == CostMode::AREA_FLOW) {
        cutFlowCosts(resCut);
      } else {
        if (this->refMode == RefMode::MAP) {
          cutDerefedCosts(resCut, refMap);
        } else {
          cutDerefedCosts(resCut);
        }
      }

      // add to the sorted list
      cutSort(thData->cutPool, thData->cutList, resCut);
    }
  }

  if ((nodeData.nFanout > 0) && (nodeData.choiceList != nullptr)) {
    mapChoices(thData, refMap, nodeData);
  }

  // start with the elementary cut
  trivialCut            = thData->cutPool.getMemory();
  trivialCut->leaves[0] = nodeData.id;
  trivialCut->nLeaves++;
  trivialCut->sig = (1U << (nodeData.id % 31));
  if (this->compTruth) {
    unsigned* cutTruth = readTruth(trivialCut);
    for (int i = 0; i < this->nWords; i++) {
      cutTruth[i] = 0xAAAAAAAA;
    }
  }
  thData->cutList.array[thData->cutList.nCuts++] = trivialCut;
  nCuts += 1;
  nTriv += 1;

  // Copy from currentCutList to the nodeCuts
  commitCuts(thData->cutPool, thData->cutList, nodeData.id);

  if (oldBestCut != nullptr) {
    if (this->nodePriCuts[nodeData.id]->delay > nodeData.reqTime) {
      oldBestCut->nextCut =
          this->nodePriCuts[nodeData.id]; // Keep the oldBestCut as the best one
      this->nodePriCuts[nodeData.id] = oldBestCut;
    } else {
      thData->cutPool.giveBackMemory(oldBestCut);
    }
  }

  if (this->deterministic == false) {
    increaseCutReferences(this->nodePriCuts[nodeData.id]);
  }
}

void PriCutManager::mapChoices(ThreadLocalData* thData, RefMap& refMap,
                               aig::NodeData& nodeData) {

  aig::GNode nextChoice = nullptr;

  for (aig::GNode currChoice = nodeData.choiceList; currChoice != nullptr;
       currChoice            = nextChoice) {

    aig::NodeData& currChoiceData =
        aigGraph.getData(currChoice, galois::MethodFlag::READ);
    nextChoice = currChoiceData.choiceList;

    // std::cout << "Node " << currChoiceData.id << " has fanout " <<
    // currChoiceData.nFanout << std::endl;

    for (PriCut* chCut = this->nodePriCuts[currChoiceData.id]; chCut != nullptr;
         chCut         = chCut->nextCut) {

      // Discard trivial cuts
      if (chCut->nLeaves == 1) {
        continue;
      }

      PriCut* chCutCopy = thData->cutPool.getMemory();
      copyCut(chCutCopy, chCut);

      /*
      if (this->compTruth) { // FIXME treat complemented choices
          if(currChoiceData.isCompl) {
              Functional32::NOT(readTruth(chCutCopy), readTruth(chCutCopy),
      this->nWords);
          }
      }
      */

      // check containment
      if (cutFilter(thData->cutPool, thData->cutList, chCutCopy)) {
        continue;
      }

      if (this->costMode == CostMode::AREA_FLOW) {
        cutFlowCosts(chCutCopy);
      } else {
        if (this->refMode == RefMode::MAP) {
          cutDerefedCosts(chCutCopy, refMap);
          // cutDerefedCosts(chCutCopy, thData->refMap);
        } else {
          cutDerefedCosts(chCutCopy);
        }
      }

      // add to the sorted list
      cutSort(thData->cutPool, thData->cutList, chCutCopy);
    }
  }
}

PriCut* PriCutManager::mergeCuts(PriCutPool& cutPool, PriCut* lhsCut,
                                 PriCut* rhsCut) {

  // assert( lhsCut->nLeaves >= rhsCut->nLeaves );
  int i, j, l;
  PriCut* resCut;

  // the case of the largest cut sizes
  if (lhsCut->nLeaves == this->K && rhsCut->nLeaves == this->K) {
    for (i = 0; i < lhsCut->nLeaves; i++) {
      if (lhsCut->leaves[i] != rhsCut->leaves[i]) {
        return nullptr;
      }
    }
    resCut = cutPool.getMemory();
    for (i = 0; i < lhsCut->nLeaves; i++) {
      resCut->leaves[i] = lhsCut->leaves[i];
    }
    resCut->nLeaves = lhsCut->nLeaves;
    resCut->sig     = lhsCut->sig | rhsCut->sig; // set the signature
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
    resCut = cutPool.getMemory();
    for (i = 0; i < lhsCut->nLeaves; i++) {
      resCut->leaves[i] = lhsCut->leaves[i];
    }
    resCut->nLeaves = lhsCut->nLeaves;
    resCut->sig     = lhsCut->sig | rhsCut->sig; // set the signature
    return resCut;
  }

  // compare two cuts with different numbers
  resCut = cutPool.getMemory();
  i      = 0;
  j      = 0;
  for (l = 0; l < this->K; l++) {
    if (j == rhsCut->nLeaves) {
      if (i == lhsCut->nLeaves) {
        resCut->nLeaves = l;
        resCut->sig     = lhsCut->sig | rhsCut->sig; // set the signature
        return resCut;
      }
      resCut->leaves[l] = lhsCut->leaves[i++];
      continue;
    }

    if (i == lhsCut->nLeaves) {
      if (j == rhsCut->nLeaves) {
        resCut->nLeaves = l;
        resCut->sig     = lhsCut->sig | rhsCut->sig; // set the signature
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
    cutPool.giveBackMemory(resCut);
    return nullptr;
  }

  resCut->nLeaves = l;
  resCut->sig     = lhsCut->sig | rhsCut->sig; // set the signature
  return resCut;
}

inline bool PriCutManager::cutFilter(PriCutPool& cutPool, PriCutList& cutList,
                                     PriCut* resCut) {

  PriCut* cut;

  for (int i = 0; i < cutList.nCuts; i++) {

    cut = cutList.array[i];

    if (cut->nLeaves <= resCut->nLeaves) {
      // skip the non-contained cuts
      if ((cut->sig & resCut->sig) != cut->sig) {
        continue;
      }
      // check containment seriously
      if (checkCutDominance(cut, resCut)) {
        cutPool.giveBackMemory(resCut); // Recycle Cut
        nFilt += 1;
        return true; // resCut is dominated
      }
    } else {
      // sKip the non-contained cuts
      if ((cut->sig & resCut->sig) != resCut->sig) {
        continue;
      }
      // check containment seriously
      if (checkCutDominance(resCut, cut)) {
        nCuts -= 1;
        nFilt += 1;
        cutList.nCuts--;
        cutPool.giveBackMemory(cut); // Recycle Cut
        for (int j = i; j < cutList.nCuts; j++) {
          cutList.array[j] = cutList.array[j + 1];
        }
      }
    }
  }
  return false;
}

inline bool PriCutManager::checkCutDominance(PriCut* smallerCut,
                                             PriCut* largerCut) {

  int i, j;
  for (i = 0; i < smallerCut->nLeaves; i++) {
    for (j = 0; j < largerCut->nLeaves; j++) {
      if (smallerCut->leaves[i] == largerCut->leaves[j]) {
        break;
      }
    }
    if (j ==
        largerCut
            ->nLeaves) { // node i in smallerCut is not contained in largerCut
      return false;
    }
  }
  // every node in smallerCut is contained in largerCut
  return true;
}

inline void PriCutManager::cutSort(PriCutPool& cutPool, PriCutList& cutList,
                                   PriCut* resCut) {

  // cut structure is empty
  if (cutList.nCuts == 0) {
    cutList.array[cutList.nCuts++] = resCut;
    nCuts += 1;
    return;
  }

  // the cut will be added - find its place
  cutList.array[cutList.nCuts++] = resCut;

  for (int i = cutList.nCuts - 2; i >= 0; i--) {
    if (sortCompare(cutList.array[i], resCut) <= 0) {
      break;
    }
    cutList.array[i + 1] = cutList.array[i];
    cutList.array[i]     = resCut;
  }

  if (cutList.nCuts > this->C) {
    cutPool.giveBackMemory(cutList.array[--cutList.nCuts]);
  } else {
    nCuts += 1;
  }
}

int PriCutManager::sortCompare(PriCut* lhsCut, PriCut* rhsCut) {

  if (this->fPower) {
    if (this->sortMode == SortMode::AREA) { // area flow
      if (lhsCut->area < rhsCut->area - this->fEpsilon)
        return -1;
      if (lhsCut->area > rhsCut->area + this->fEpsilon)
        return 1;
      if (lhsCut->power < rhsCut->power - this->fEpsilon)
        return -1;
      if (lhsCut->power > rhsCut->power + this->fEpsilon)
        return 1;
      if (lhsCut->edge < rhsCut->edge - this->fEpsilon)
        return -1;
      if (lhsCut->edge > rhsCut->edge + this->fEpsilon)
        return 1;
      if (lhsCut->nLeaves < rhsCut->nLeaves)
        return -1;
      if (lhsCut->nLeaves > rhsCut->nLeaves)
        return 1;
      if (lhsCut->delay < rhsCut->delay - this->fEpsilon)
        return -1;
      if (lhsCut->delay > rhsCut->delay + this->fEpsilon)
        return 1;
      return 0;
    }
    if (this->sortMode == SortMode::DELAY) { // delay
      if (lhsCut->delay < rhsCut->delay - this->fEpsilon)
        return -1;
      if (lhsCut->delay > rhsCut->delay + this->fEpsilon)
        return 1;
      if (lhsCut->nLeaves < rhsCut->nLeaves)
        return -1;
      if (lhsCut->nLeaves > rhsCut->nLeaves)
        return 1;
      if (lhsCut->area < rhsCut->area - this->fEpsilon)
        return -1;
      if (lhsCut->area > rhsCut->area + this->fEpsilon)
        return 1;
      if (lhsCut->power < rhsCut->power - this->fEpsilon)
        return -1;
      if (lhsCut->power > rhsCut->power + this->fEpsilon)
        return 1;
      if (lhsCut->edge < rhsCut->edge - this->fEpsilon)
        return -1;
      if (lhsCut->edge > rhsCut->edge + this->fEpsilon)
        return 1;
      return 0;
    }
    assert(this->sortMode == SortMode::DELAY_OLD); // delay old, exact area
    if (lhsCut->delay < rhsCut->delay - this->fEpsilon)
      return -1;
    if (lhsCut->delay > rhsCut->delay + this->fEpsilon)
      return 1;
    if (lhsCut->power < rhsCut->power - this->fEpsilon)
      return -1;
    if (lhsCut->power > rhsCut->power + this->fEpsilon)
      return 1;
    if (lhsCut->edge < rhsCut->edge - this->fEpsilon)
      return -1;
    if (lhsCut->edge > rhsCut->edge + this->fEpsilon)
      return 1;
    if (lhsCut->area < rhsCut->area - this->fEpsilon)
      return -1;
    if (lhsCut->area > rhsCut->area + this->fEpsilon)
      return 1;
    if (lhsCut->nLeaves < rhsCut->nLeaves)
      return -1;
    if (lhsCut->nLeaves > rhsCut->nLeaves)
      return 1;
    return 0;
  } else {                                  // regular
    if (this->sortMode == SortMode::AREA) { // area
      if (lhsCut->area < rhsCut->area - this->fEpsilon)
        return -1;
      if (lhsCut->area > rhsCut->area + this->fEpsilon)
        return 1;
      if (lhsCut->edge < rhsCut->edge - this->fEpsilon)
        return -1;
      if (lhsCut->edge > rhsCut->edge + this->fEpsilon)
        return 1;
      if (lhsCut->power < rhsCut->power - this->fEpsilon)
        return -1;
      if (lhsCut->power > rhsCut->power + this->fEpsilon)
        return 1;
      if (lhsCut->nLeaves < rhsCut->nLeaves)
        return -1;
      if (lhsCut->nLeaves > rhsCut->nLeaves)
        return 1;
      if (lhsCut->delay < rhsCut->delay - this->fEpsilon)
        return -1;
      if (lhsCut->delay > rhsCut->delay + this->fEpsilon)
        return 1;
      return 0;
    }
    if (this->sortMode == SortMode::DELAY) { // delay
      if (lhsCut->delay < rhsCut->delay - this->fEpsilon)
        return -1;
      if (lhsCut->delay > rhsCut->delay + this->fEpsilon)
        return 1;
      if (lhsCut->nLeaves < rhsCut->nLeaves)
        return -1;
      if (lhsCut->nLeaves > rhsCut->nLeaves)
        return 1;
      if (lhsCut->area < rhsCut->area - this->fEpsilon)
        return -1;
      if (lhsCut->area > rhsCut->area + this->fEpsilon)
        return 1;
      if (lhsCut->edge < rhsCut->edge - this->fEpsilon)
        return -1;
      if (lhsCut->edge > rhsCut->edge + this->fEpsilon)
        return 1;
      if (lhsCut->power < rhsCut->power - this->fEpsilon)
        return -1;
      if (lhsCut->power > rhsCut->power + this->fEpsilon)
        return 1;
      return 0;
    }
    assert(this->sortMode == SortMode::DELAY_OLD);
    if (lhsCut->delay < rhsCut->delay - this->fEpsilon)
      return -1;
    if (lhsCut->delay > rhsCut->delay + this->fEpsilon)
      return 1;
    if (lhsCut->area < rhsCut->area - this->fEpsilon)
      return -1;
    if (lhsCut->area > rhsCut->area + this->fEpsilon)
      return 1;
    if (lhsCut->edge < rhsCut->edge - this->fEpsilon)
      return -1;
    if (lhsCut->edge > rhsCut->edge + this->fEpsilon)
      return 1;
    if (lhsCut->power < rhsCut->power - this->fEpsilon)
      return -1;
    if (lhsCut->power > rhsCut->power + this->fEpsilon)
      return 1;
    if (lhsCut->nLeaves < rhsCut->nLeaves)
      return -1;
    if (lhsCut->nLeaves > rhsCut->nLeaves)
      return 1;
    return 0;
  }
}

inline void PriCutManager::commitCuts(PriCutPool& cutPool, PriCutList& cutList,
                                      int nodeId) {

  assert(cutList.nCuts != 0);

  recycleNodeCuts(cutPool, nodeId);

  // Copy from currenti CutList to the nodePriCuts and clean up the cutList
  this->nodePriCuts[nodeId] = cutList.array[0];

  int i;
  for (i = 0; i < cutList.nCuts - 1; i++) {
    cutList.array[i]->nextCut = cutList.array[i + 1];
    cutList.array[i]          = nullptr;
  }
  cutList.array[i]->nextCut = nullptr;
  cutList.array[i]          = nullptr;
  cutList.nCuts             = 0;

  assert(this->nodePriCuts[nodeId] != nullptr);
}

/*
 *     This method gives the cut's memory back to current thread cutPool.
 *     However, the memory can be allocated by the cutPool of one thread
 *     and reused by a cutPool of another thread. But, the original cutPool
 *     will be responsible for dealocating the memory.
 */
inline void PriCutManager::recycleNodeCuts(PriCutPool& cutPool, int nodeId) {

  PriCut* cut = this->nodePriCuts[nodeId];

  while (cut != nullptr) {
    PriCut* nextCut = cut->nextCut;
    cutPool.giveBackMemory(cut);
    cut = nextCut;
  }

  this->nodePriCuts[nodeId] = nullptr;
}

inline void PriCutManager::cleanupCutList(PriCutPool& cutPool,
                                          PriCutList& cutList) {

  for (int i = 0; i < cutList.nCuts; i++) {
    cutPool.giveBackMemory(cutList.array[i]);
    cutList.array[i] = nullptr;
  }

  cutList.nCuts = 0;
}

inline void PriCutManager::copyCut(PriCut* dest, PriCut* source) {

  dest->area    = source->area;
  dest->edge    = source->edge;
  dest->power   = source->power;
  dest->delay   = source->delay;
  dest->sig     = source->sig;
  dest->nLeaves = source->nLeaves;
  dest->nextCut = nullptr;
  for (int i = 0; i < source->nLeaves; i++) {
    dest->leaves[i] = source->leaves[i];
  }
  if (this->compTruth) {
    unsigned int* destTruth   = readTruth(dest);
    unsigned int* sourceTruth = readTruth(source);
    Functional32::copy(destTruth, sourceTruth, this->nWords);
  }
}

void PriCutManager::computeTruth(AuxTruth& auxTruth, PriCut* resCut,
                                 PriCut* lhsCut, PriCut* rhsCut,
                                 bool lhsPolarity, bool rhsPolarity) {

  // permute the first table
  if (lhsPolarity) {
    Functional32::copy(auxTruth.truth[0], readTruth(lhsCut), this->nWords);
  } else {
    Functional32::NOT(auxTruth.truth[0], readTruth(lhsCut), this->nWords);
  }
  Functional32::truthStretch(auxTruth.truth[2], auxTruth.truth[0],
                             lhsCut->nLeaves, this->K,
                             truthPhase(resCut, lhsCut));

  // permute the second table
  if (rhsPolarity) {
    Functional32::copy(auxTruth.truth[1], readTruth(rhsCut), this->nWords);
  } else {
    Functional32::NOT(auxTruth.truth[1], readTruth(rhsCut), this->nWords);
  }
  Functional32::truthStretch(auxTruth.truth[3], auxTruth.truth[1],
                             rhsCut->nLeaves, this->K,
                             truthPhase(resCut, rhsCut));

  // produce the resulting table. In this first version we are not considering
  // the cut->fCompl flag. It may be considerer in further versions according to
  // the demand.
  // if ( cut->fCompl ) {
  //	Functional32::NAND( readTruth( cut ) , auxTruth[2], auxTruth[3], K );
  //}
  // else {
  Functional32::AND(readTruth(resCut), auxTruth.truth[2], auxTruth.truth[3],
                    this->nWords);
  //}
}

inline unsigned PriCutManager::truthPhase(PriCut* resCut, PriCut* inCut) {

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

unsigned int* PriCutManager::readTruth(PriCut* cut) {
  return (unsigned*)(cut->leaves + this->K);
}

void PriCutManager::increaseCutReferences(PriCut* cut) {

  int leafId;

  for (int i = 0; i < cut->nLeaves; i++) {
    leafId                  = cut->leaves[i];
    aig::GNode leaf         = this->aig.getNodes()[leafId];
    aig::NodeData& leafData = aigGraph.getData(leaf, galois::MethodFlag::WRITE);
    assert(leafData.nRefs >= 0);
    leafData.nRefs++;
  }
}

void PriCutManager::decreaseCutReferences(PriCut* cut) {

  int leafId;

  for (int i = 0; i < cut->nLeaves; i++) {
    leafId                  = cut->leaves[i];
    aig::GNode leaf         = this->aig.getNodes()[leafId];
    aig::NodeData& leafData = aigGraph.getData(leaf, galois::MethodFlag::WRITE);
    assert(leafData.nRefs > 0);
    --leafData.nRefs;
  }
}

// ################### Start of the New Cut's Cost Functions
// ###################### //
inline float PriCutManager::cutDelay(PriCut* cut) {

  int leafId;
  float currDelay, delay = std::numeric_limits<float>::min();

  for (int i = 0; i < cut->nLeaves; i++) {
    leafId    = cut->leaves[i];
    currDelay = getBestCut(leafId)->delay + 1.0;
    delay     = std::max(delay, currDelay);
  }
  return delay;
}

void PriCutManager::cutFlowCosts(PriCut* cut) {

  int leafId;
  float areaFlow = 1.0;
  float edgeFlow = cut->nLeaves;
  float currDelay, delay = std::numeric_limits<float>::min();
  PriCut* bestCut = nullptr;

  for (int i = 0; i < cut->nLeaves; i++) {

    leafId          = cut->leaves[i];
    aig::GNode leaf = this->aig.getNodes()[leafId];
    aig::NodeData& leafData =
        aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);
    bestCut = getBestCut(leafId);

    if ((leafData.nRefs == 0) || (leafData.type == aig::NodeType::CONSTZERO)) {
      areaFlow += bestCut->area;
      edgeFlow += bestCut->edge;
    } else {
      assert(leafData.nRefs > this->fEpsilon);
      areaFlow += bestCut->area / leafData.nRefs;
      edgeFlow += bestCut->edge / leafData.nRefs;
    }

    currDelay = bestCut->delay + 1.0;
    delay     = std::max(delay, currDelay);
  }

  cut->area  = areaFlow;
  cut->edge  = edgeFlow;
  cut->delay = delay;
}

// STANDARD VERSION
void PriCutManager::cutDerefedCosts(PriCut* cut) {

  float area1 = 0, area2 = 0, edge1 = 0, edge2 = 0;

  if (cut->nLeaves < 2) {
    cut->area = 0;
    cut->edge = cut->nLeaves;
    return;
  }

  cutRefCosts(cut, area1, edge1);
  cutDerefCosts(cut, area2, edge2);

  assert(area2 > area1 - this->fEpsilon);
  assert(area2 < area1 + this->fEpsilon);
  assert(edge2 > edge1 - this->fEpsilon);
  assert(edge2 < edge1 + this->fEpsilon);

  cut->area  = area2;
  cut->edge  = edge2;
  cut->delay = cutDelay(cut);
}

void PriCutManager::cutRefCosts(PriCut* cut, float& area, float& edge) {

  int leafId;
  area += 1.0;
  edge += cut->nLeaves;

  for (int i = 0; i < cut->nLeaves; i++) {

    leafId                  = cut->leaves[i];
    aig::GNode leaf         = this->aig.getNodes()[leafId];
    aig::NodeData& leafData = aigGraph.getData(leaf, galois::MethodFlag::WRITE);

    assert(leafData.nRefs >= 0);
    if ((leafData.nRefs++ > 0) || (leafData.type != aig::NodeType::AND))
      continue;

    cutRefCosts(getBestCut(leafId), area, edge);
  }
}

void PriCutManager::cutDerefCosts(PriCut* cut, float& area, float& edge) {

  int leafId;
  area += 1.0;
  edge += cut->nLeaves;

  for (int i = 0; i < cut->nLeaves; i++) {

    leafId                  = cut->leaves[i];
    aig::GNode leaf         = this->aig.getNodes()[leafId];
    aig::NodeData& leafData = aigGraph.getData(leaf, galois::MethodFlag::WRITE);

    assert(leafData.nRefs > 0);
    if (--leafData.nRefs > 0 || (leafData.type != aig::NodeType::AND))
      continue;

    cutDerefCosts(getBestCut(leafId), area, edge);
  }
}

// REFMAP VERSION

void PriCutManager::cutDerefedCosts(PriCut* cut, RefMap& refMap) {

  float area1 = 0, area2 = 0, edge1 = 0, edge2 = 0;

  if (cut->nLeaves < 2) {
    cut->area = 0;
    cut->edge = cut->nLeaves;
    return;
  }

  cutRefCosts(cut, area1, edge1, refMap);
  cutDerefCosts(cut, area2, edge2, refMap);

  assert(area2 > area1 - this->fEpsilon);
  assert(area2 < area1 + this->fEpsilon);
  assert(edge2 > edge1 - this->fEpsilon);
  assert(edge2 < edge1 + this->fEpsilon);

  cut->area  = area2;
  cut->edge  = edge2;
  cut->delay = cutDelay(cut);
}

void PriCutManager::cutRefCosts(PriCut* cut, float& area, float& edge,
                                RefMap& refMap) {

  int leafId;
  area += 1.0;
  edge += cut->nLeaves;

  for (int i = 0; i < cut->nLeaves; i++) {

    leafId          = cut->leaves[i];
    aig::GNode leaf = this->aig.getNodes()[leafId];
    aig::NodeData& leafData =
        aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);

    // Experimental
    auto it = refMap.find(leafId);
    if (it != refMap.end()) {
      assert(it->second >= 0);
      if ((it->second++ > 0) || (leafData.type != aig::NodeType::AND))
        continue;
    } else {
      assert(leafData.nRefs >= 0);
      refMap.insert({leafId, leafData.nRefs + 1});
      if ((leafData.nRefs > 0) || (leafData.type != aig::NodeType::AND))
        continue;
    }

    cutRefCosts(getBestCut(leafId), area, edge, refMap);
  }
}

void PriCutManager::cutDerefCosts(PriCut* cut, float& area, float& edge,
                                  RefMap& refMap) {

  int leafId;
  area += 1.0;
  edge += cut->nLeaves;

  for (int i = 0; i < cut->nLeaves; i++) {

    leafId          = cut->leaves[i];
    aig::GNode leaf = this->aig.getNodes()[leafId];
    aig::NodeData& leafData =
        aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);

    // Experimental
    auto it = refMap.find(leafId);
    if (it != refMap.end()) {
      assert(it->second > 0);
      if (--it->second > 0 || (leafData.type != aig::NodeType::AND))
        continue;
    } else {
      assert(leafData.nRefs > 0);
      refMap.insert({leafId, leafData.nRefs - 1});
      if ((leafData.nRefs - 1) > 0 || (leafData.type != aig::NodeType::AND))
        continue;
    }

    cutDerefCosts(getBestCut(leafId), area, edge, refMap);
  }
}
// ################### End of the NewCuts Cost Functions ######################
// //

void PriCutManager::resetNodeCountersFanout() {

  const float FLOAT_MAX = std::numeric_limits<float>::max();

  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    nodeData.counter = 0;
    nodeData.nRefs   = nodeData.nFanout;
    nodeData.reqTime = FLOAT_MAX;
  }

  // galois::do_all( galois::iterate( aigGraph ), ResetNodeCountersFanout{
  // aigGraph }, galois::loopname("ResetOperatorFanout"), galois::steal() );
}

void PriCutManager::resetNodeCountersZero() {

  const float FLOAT_MAX = std::numeric_limits<float>::max();

  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    nodeData.counter = 0;
    nodeData.nRefs   = 0;
    nodeData.reqTime = FLOAT_MAX;
  }

  // galois::do_all( galois::iterate( aigGraph ), ResetNodeCountersZero{
  // aigGraph }, galois::loopname("ResetOperatorZero"), galois::steal() );
}

void PriCutManager::resetNodeCountersOnly() {

  const float FLOAT_MAX = std::numeric_limits<float>::max();

  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    nodeData.counter = 0;
    nodeData.reqTime = FLOAT_MAX;
  }

  // galois::do_all( galois::iterate( aigGraph ), ResetNodeCountersOnly{
  // aigGraph }, galois::loopname("ResetOperatorOnly"), galois::steal() );
}

void PriCutManager::computeReferenceCounters() {

  PriCut* bestCut;
  int size = this->nNodes + 1;

  for (int i = 0; i < size; i++) {
    bestCut = this->nodePriCuts[i];
    if (bestCut == nullptr) {
      continue;
    }
    if (bestCut->nLeaves == 1) {
      continue; // skip trivial cuts
    }
    for (int j = 0; j < bestCut->nLeaves; j++) {
      aig::GNode leaf = this->aig.getNodes()[bestCut->leaves[j]];
      aig::NodeData& leafData =
          aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);
      leafData.nRefs++;
    }
  }
}

void PriCutManager::computeCoveringReferenceCounters() {

  computeCovering();

  PriCut* bestCut;

  for (auto entry : this->covering) {
    bestCut = entry.second;
    if (bestCut == nullptr) {
      continue;
    }
    if (bestCut->nLeaves == 1) {
      continue; // skip trivial cuts
    }
    for (int j = 0; j < bestCut->nLeaves; j++) {
      aig::GNode leaf = this->aig.getNodes()[bestCut->leaves[j]];
      aig::NodeData& leafData =
          aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);
      leafData.nRefs++;
    }
  }
}

void PriCutManager::computeRequiredTimes() {

  float maxDelay = 0;
  PriCut* bestCut;

  for (auto po : this->aig.getOutputNodes()) {
    auto inEdgeIt     = aigGraph.in_edge_begin(po);
    aig::GNode inNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& inNodeData =
        aigGraph.getData(inNode, galois::MethodFlag::UNPROTECTED);
    if (inNodeData.type == aig::NodeType::CONSTZERO) {
      continue;
    }
    bestCut = getBestCut(inNodeData.id);
    if (maxDelay < bestCut->delay - this->fEpsilon) {
      maxDelay = bestCut->delay;
    }
  }

  for (auto po : this->aig.getOutputNodes()) {
    auto inEdgeIt     = aigGraph.in_edge_begin(po);
    aig::GNode inNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& inNodeData =
        aigGraph.getData(inNode, galois::MethodFlag::UNPROTECTED);
    inNodeData.reqTime = maxDelay;
  }

  // iterating from 0 to N is reverse topological order
  // iterating from N to 0 is topological order
  for (auto node : this->sortedNodes) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);

    // Reset node data to prepare for the next mapping pass
    nodeData.counter = 0;
    if (this->deterministic) {
      nodeData.nRefs = 0;
    }

    bestCut = getBestCut(nodeData.id);
    for (int i = 0; i < bestCut->nLeaves; i++) {
      aig::GNode leaf = this->aig.getNodes()[bestCut->leaves[i]];
      aig::NodeData& leafData =
          aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);
      leafData.reqTime = std::min((nodeData.reqTime - 1), leafData.reqTime);
    }
  }
}

void PriCutManager::computeCovering() {

  PriCut* bestCut;
  aig::GNode leaf;
  int leafId, nodeId;
  std::vector<int> S;

  this->covering.clear();

  this->nLevels = -1;
  for (auto po : this->aig.getOutputNodes()) {
    auto inEdgeIt     = aigGraph.in_edge_begin(po);
    aig::GNode inNode = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& inNodeData =
        aigGraph.getData(inNode, galois::MethodFlag::UNPROTECTED);

    if ((inNodeData.type != aig::NodeType::PI) &&
        (inNodeData.type != aig::NodeType::LATCH) &&
        (inNodeData.type != aig::NodeType::CONSTZERO)) {
      S.push_back(inNodeData.id);
      bestCut = getBestCut(inNodeData.id);
      if (this->nLevels < bestCut->delay) {
        this->nLevels = bestCut->delay;
      }
    }
  }

  while (!S.empty()) {
    nodeId = S.back();
    S.pop_back();

    auto it = this->covering.find(nodeId);
    if (it != this->covering.end()) {
      continue;
    }

    bestCut = getBestCut(nodeId);
    this->covering.insert({nodeId, bestCut});

    for (int i = 0; i < bestCut->nLeaves; i++) {
      leafId = bestCut->leaves[i];
      leaf   = this->aig.getNodes()[leafId];
      aig::NodeData& leafData =
          aigGraph.getData(leaf, galois::MethodFlag::UNPROTECTED);
      leafData.nRefs++; // Update reference counters
      auto it = this->covering.find(leafId);
      if (it == this->covering.end()) {
        if ((leafData.type != aig::NodeType::PI) &&
            (leafData.type != aig::NodeType::LATCH) &&
            (leafData.type != aig::NodeType::CONSTZERO)) {
          S.push_back(leafId);
        }
      }
    }
  }
}

inline void PriCutManager::switchToFirstDelayMode() {
  this->passId++;
  this->sortMode = SortMode::DELAY;
  this->costMode = CostMode::AREA_FLOW;
}

inline void PriCutManager::switchToSecondDelayMode() {
  this->passId++;
  this->sortMode = SortMode::DELAY_OLD;
  this->costMode = CostMode::AREA_FLOW;
}

inline void PriCutManager::switchToAreaFlowMode() {
  this->passId++;
  this->sortMode = SortMode::AREA;
  this->costMode = CostMode::AREA_FLOW;
}

inline void PriCutManager::switchToLocalAreaMode() {
  this->passId++;
  this->sortMode = SortMode::AREA;
  this->costMode = CostMode::LOCAL_AREA;
}

inline aig::Aig& PriCutManager::getAig() { return this->aig; }

inline PriCut* PriCutManager::getBestCut(int nodeId) {
  return this->nodePriCuts[nodeId]; // the first cut is the best cut
}

int PriCutManager::getNumLUTs() {
  this->nLUTs = this->covering.size();
  return this->nLUTs;
}

int PriCutManager::getNumLevels() { return this->nLevels; }

int PriCutManager::getK() { return this->K; }

int PriCutManager::getC() { return this->C; }

int PriCutManager::getNWords() { return this->nWords; }

int PriCutManager::getNThreads() { return this->nThreads; }

bool PriCutManager::isDeterministic() { return this->deterministic; }

bool PriCutManager::getCompTruthFlag() { return this->compTruth; }

bool PriCutManager::getVerboseFlag() { return this->verbose; }

long double PriCutManager::getKcutTime() { return this->kcutTime; }

void PriCutManager::setKcutTime(long double time) { this->kcutTime = time; }

PerThreadData& PriCutManager::getPerThreadData() { return this->perThreadData; }

PriCut** PriCutManager::getNodePriCuts() { return this->nodePriCuts; }

Covering& PriCutManager::getCovering() { return this->covering; }

void PriCutManager::printCovering() {

  std::cout << std::endl
            << "########## Mapping Covering ###############" << std::endl;
  PriCut* bestCut;
  for (auto entry : this->covering) {
    std::cout << "Node " << entry.first << ": { ";
    bestCut = entry.second;
    for (int i = 0; i < bestCut->nLeaves; i++) {
      std::cout << bestCut->leaves[i] << " ";
    }
    std::cout << "}" << std::endl;
    // std::cout << "}[" << Functional32::toHex( readTruth( bestCut ),
    // this->nWords )  << "] " << std::endl;
  }
  std::cout << std::endl
            << "###########################################" << std::endl;
}

void PriCutManager::printNodeCuts(int nodeId, long int& counter) {

  std::cout << "Node " << nodeId << ": { ";
  for (PriCut* currentCut = this->nodePriCuts[nodeId]; currentCut != nullptr;
       currentCut         = currentCut->nextCut) {
    counter++;
    std::cout << "{ ";
    for (int i = 0; i < currentCut->nLeaves; i++) {
      std::cout << currentCut->leaves[i] << " ";
    }
    // std::cout << "} ";
    std::cout << "}(a" << currentCut->area << ") ";
    // std::cout << "}(a" << currentCut->area << ", e" << currentCut->edge << ")
    // "; std::cout << "}[" << Functional32::toHex( readTruth( currentCut ),
    // this->nWords )  << "] ";
  }
  std::cout << "}" << std::endl;
}

void PriCutManager::printAllCuts() {

  long int counter = 0;

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

void PriCutManager::printNodeBestCut(int nodeId) {

  PriCut* bestCut = getBestCut(nodeId);
  std::cout << "Node " << nodeId << ": { ";
  for (int i = 0; i < bestCut->nLeaves; i++) {
    std::cout << bestCut->leaves[i] << " ";
  }
  std::cout << "}(a" << bestCut->area << ")" << std::endl;
  // std::cout << "}(a" << bestCut->area << ", e" << bestCut->edge << ")" <<
  // std::endl; std::cout << "}[" << Functional32::toHex( readTruth( bestCut ),
  // this->nWords )  << "] " << std::endl;
}

void PriCutManager::printBestCuts() {

  std::cout << std::endl << "########## Best K-Cuts ###########" << std::endl;
  for (aig::GNode node : aigGraph) {
    aig::NodeData& nodeData =
        aigGraph.getData(node, galois::MethodFlag::UNPROTECTED);
    if ((nodeData.type == aig::NodeType::AND) ||
        (nodeData.type == aig::NodeType::PI)) {
      printNodeBestCut(nodeData.id);
    }
  }
  std::cout << "#################################" << std::endl;
}

void PriCutManager::printCutStatistics() {

  long int nCutsRed = nCuts.reduce();
  nCutsRed += this->aig.getNumInputs();

  long int nTrivRed = nTriv.reduce();
  nTrivRed += this->aig.getNumInputs();

  long int nFiltRed = nFilt.reduce();

  long int nSatuRed = nSatu.reduce();

  std::cout << std::endl
            << "############## Cut Statistics #############" << std::endl;
  std::cout << "nCuts: " << nCutsRed << std::endl;
  std::cout << "nTriv: " << nTrivRed << std::endl;
  std::cout << "nFilt: " << nFiltRed << std::endl;
  std::cout << "nSatu: " << nSatuRed << std::endl;
  std::cout << "nCutPerNode: " << (((double)nCutsRed) / this->nNodes)
            << std::endl;
  std::cout << "###########################################" << std::endl;
}

void PriCutManager::printRuntimes() {

  std::cout << std::endl << "#### Runtimes in microsecond ####" << std::endl;
  // std::cout << "Merge: " << mergeTime << std::endl;
  // std::cout << "Filter: " << filterTime << std::endl;
  // std::cout << "ProcTwo: " << procTwoTime << std::endl;
  // std::cout << "Compute: " << compTime << std::endl;
  // std::cout << "Schedule: " << scheduleTime << std::endl;
  std::cout << "Total: " << this->kcutTime << std::endl;
  std::cout << "#################################" << std::endl;
}

// ######################## BEGIN OPERATOR ######################## //
struct KPriCutOperator {

  const float FLOAT_MAX = std::numeric_limits<float>::max();
  PriCutManager& cutMan;

  KPriCutOperator(PriCutManager& cutMan) : cutMan(cutMan) {}

  void operator()(aig::GNode node, galois::UserContext<aig::GNode>& ctx) {

    aig::Aig& aig        = cutMan.getAig();
    aig::Graph& aigGraph = aig.getGraph();

    aig::NodeData& nodeData = aigGraph.getData(node, galois::MethodFlag::READ);

    if (nodeData.type == aig::NodeType::AND) {

      // Touching outgoing neighobors to acquire their locks
      aigGraph.out_edges(node);

      // Combine Cuts
      auto inEdgeIt      = aigGraph.in_edge_begin(node);
      aig::GNode lhsNode = aigGraph.getEdgeDst(inEdgeIt);
      aig::NodeData& lhsData =
          aigGraph.getData(lhsNode, galois::MethodFlag::READ);
      bool lhsPolarity = aigGraph.getEdgeData(inEdgeIt);

      inEdgeIt++;
      aig::GNode rhsNode = aigGraph.getEdgeDst(inEdgeIt);
      aig::NodeData& rhsData =
          aigGraph.getData(rhsNode, galois::MethodFlag::READ);
      bool rhsPolarity = aigGraph.getEdgeData(inEdgeIt);

      ThreadLocalData* thData = cutMan.getPerThreadData().getLocal();

      RefMap refMap(ctx.getPerIterAlloc());

      cutMan.computePriCuts(thData, refMap, nodeData, lhsData.id, rhsData.id,
                            lhsPolarity, rhsPolarity);

      // Mark node as processed
      nodeData.counter = nodeData.nFanout;
      nodeData.reqTime = FLOAT_MAX;

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

      // Delete cuts of previous nodes if possibles
      if (lhsData.type == aig::NodeType::AND) {
        if (--lhsData.counter == 0) {
          PriCut* cut =
              cutMan.getNodePriCuts()[lhsData.id]
                  ->nextCut; // skipe the first cut which is the best cut
          cutMan.getNodePriCuts()[lhsData.id]->nextCut = nullptr;
          while (cut != nullptr) {
            PriCut* nextCut = cut->nextCut;
            thData->cutPool.giveBackMemory(cut);
            cut = nextCut;
          }
        }
      }
      if (rhsData.type == aig::NodeType::AND) {
        if (--rhsData.counter == 0) {
          PriCut* cut =
              cutMan.getNodePriCuts()[rhsData.id]
                  ->nextCut; // skipe the first cut which is the best cut
          cutMan.getNodePriCuts()[rhsData.id]->nextCut = nullptr;
          while (cut != nullptr) {
            PriCut* nextCut = cut->nextCut;
            thData->cutPool.giveBackMemory(cut);
            cut = nextCut;
          }
        }
      }
    } else {
      if (nodeData.type == aig::NodeType::PI) {
        // Touching outgoing neighobors to acquire their locks and their fanin
        // node's locks.
        aigGraph.out_edges(node);

        if (cutMan.getNodePriCuts()[nodeData.id] == nullptr) {
          // Set the trivial cut
          nodeData.counter        = 3;
          ThreadLocalData* thData = cutMan.getPerThreadData().getLocal();
          PriCut* trivialCut      = thData->cutPool.getMemory();
          trivialCut->leaves[0]   = nodeData.id;
          trivialCut->nLeaves++;
          trivialCut->sig = (1U << (nodeData.id % 31));
          if (cutMan.getCompTruthFlag()) {
            unsigned* cutTruth = cutMan.readTruth(trivialCut);
            for (int i = 0; i < cutMan.getNWords(); i++) {
              cutTruth[i] = 0xAAAAAAAA;
            }
          }
          cutMan.getNodePriCuts()[nodeData.id] = trivialCut;
        }

        nodeData.counter = 0;
        nodeData.reqTime = FLOAT_MAX;

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
// ######################## END OPERATOR ######################## //

void runKPriCutOperator(PriCutManager& cutMan) {

  typedef galois::worklists::PerSocketChunkBag<1000> DC_BAG;
  bool verbose      = cutMan.getVerboseFlag();
  int nAreaRecovery = 2, nAreaFlow = 1, nLocalArea = 1;
  aig::Aig& aig = cutMan.getAig();
  //	aig::Graph & aigGraph = aig.getGraph();

  if (verbose) {
    std::cout << std::endl << "########## LUT Mapping ###########" << std::endl;
    std::cout << "Mapping in First Delay Mode" << std::endl;
  }
  cutMan.switchToFirstDelayMode();
  cutMan.resetNodeCountersFanout();
  // Galois Parallel Foreach
  galois::for_each(galois::iterate(aig.getInputNodes()),
                   KPriCutOperator(cutMan), galois::wl<DC_BAG>(),
                   galois::loopname("KPriCutOperator"),
                   galois::per_iter_alloc());

  if (verbose) {
    std::cout << "Mapping in Second Delay Mode" << std::endl;
  }
  cutMan.switchToSecondDelayMode();
  cutMan.computeRequiredTimes();
  if (cutMan.isDeterministic()) {
    cutMan.computeCovering();
  }
  // Galois Parallel Foreach
  galois::for_each(galois::iterate(aig.getInputNodes()),
                   KPriCutOperator(cutMan), galois::wl<DC_BAG>(),
                   galois::loopname("KPriCutOperator"),
                   galois::per_iter_alloc());

  for (int i = 1; i <= nAreaRecovery; i++) {

    for (int j = 1; j <= nLocalArea; j++) {
      if (verbose) {
        std::cout << "Mapping in Local Area Mode" << std::endl;
      }
      cutMan.switchToLocalAreaMode();
      cutMan.computeRequiredTimes();
      if (cutMan.isDeterministic()) {
        cutMan.computeCovering();
      }
      // Galois Parallel Foreach
      galois::for_each(galois::iterate(aig.getInputNodes()),
                       KPriCutOperator(cutMan), galois::wl<DC_BAG>(),
                       galois::loopname("KPriCutOperator"),
                       galois::per_iter_alloc());
    }

    for (int j = 1; j <= nAreaFlow; j++) {
      if (verbose) {
        std::cout << "Mapping in Area Flow Mode" << std::endl;
      }
      cutMan.switchToAreaFlowMode();
      cutMan.computeRequiredTimes();
      if (cutMan.isDeterministic()) {
        cutMan.computeCovering();
      }
      // Galois Parallel Foreach
      galois::for_each(galois::iterate(aig.getInputNodes()),
                       KPriCutOperator(cutMan), galois::wl<DC_BAG>(),
                       galois::loopname("KPriCutOperator"),
                       galois::per_iter_alloc());
    }
  }

  if (verbose) {
    std::cout << "Covering ..." << std::endl;
  }
  cutMan.computeCovering();
}

} /* namespace algorithm */
